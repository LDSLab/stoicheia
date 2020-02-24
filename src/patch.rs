use crate::{Axis, Fallible, Label, StoiError};
use itertools::Itertools;
use ndarray as nd;
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use num_traits::Zero;
use std::collections::HashMap;
use std::convert::TryInto;

/// A tensor with labeled axes
///
/// A patch has several interesting properties:
///
///   - A name, which is the tensor they are part of
///   - A set of axes, and labels of each of the components along that axis
///       - The axes must match the length and order of the dense array
///       - They are stored sorted internally
///       - They must be unique
///       - They might not be consecutive (it depends on the Quilt)
///       - They might overlap other patches (it depends on the Quilt)
///   - A regular array of 32-bit floats in the same order as the list of axes
///
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct Patch {
    /// Names and labels of the axes
    axes: Vec<Axis>,
    /// Tensor containing all the elements of this patch
    dense: ArrayD<f32>,
}

impl Patch {
    /// Create a new patch from an array and some labels
    ///
    /// The labels must be in sorted order (within the patch), if not they will be sorted.
    /// The axes dimensions must match the dense array's dimensions, otherwise it will error.
    pub fn new<A, E>(orig_axes: Vec<A>, dense: ArrayD<f32>)
        -> Fallible<Self>
        where
            A: TryInto<Axis, Error=E>,
            E: Into<StoiError>
        {
        // TODO: is there some way we can into() less than three times?
        let mut axes : Vec<Axis> = vec![];
        for ax in orig_axes {
            axes.push(ax.try_into().map_err(|e| e.into())?);
        }

        if axes.len() != dense.ndim() {
            return Err(StoiError::InvalidValue(
                "The number of labeled axes doesn't match the number of axes in the dense tensor.",
            ));
        }
        if !axes
            .iter()
            .zip(dense.axes())
            .all(|(ax, d_ax)| ax.len() == d_ax.len())
        {
            return Err(StoiError::InvalidValue(
                "The shape of the axis labels doesn't match the shape of the dense tensor.",
            ));
        }

        let dims = axes.iter().map(|a| a.len()).collect_vec();
        if dims.iter().product::<usize>() > 256 << 20 {
            return Err(StoiError::TooLarge(
                "Patches must be 256 million elements or less (1GB of 32bit floats)",
            ));
        }

        Ok(Self { axes, dense })
    }

    /// Create a new empty patch given some axes
    /// 
    /// Cloning axes small enough to be part of a patch is usually cheap, so this
    /// function uses several clones and conversions to make using it more convenient
    pub fn try_from_axes<A, E>(orig_axes: Vec<A>)
        -> Fallible<Self>
        where
        A: TryInto<Axis, Error=E>,
        E: Into<StoiError>
        {
        // TODO: is there some way we can into() less than three times?
        let mut axes : Vec<Axis> = vec![];
        for ax in orig_axes {
            axes.push(ax.try_into().map_err(|e| e.into())?);
        }
        
        let dims = axes.iter().map(|a| a.len()).collect_vec();
        if dims.iter().product::<usize>() > 256 << 20 {
            return Err(StoiError::TooLarge(
                "Patches must be 256 million elements or less (1GB of 32bit floats)",
            ));
        }
        Self::new(axes, ArrayD::from_elem(dims, std::f32::NAN))
    }

    /// How many dimensions in this patch
    pub fn ndim(&self) -> usize {
        self.dense.ndim()
    }

    /// Apply another patch to this one, changing `self` where it overlaps with `pat`.
    ///
    /// This is not the same as merging the patches, because this only changes `self` where it
    /// overlaps with `pat`, and won't allocate or expand either one.
    pub fn apply(&mut self, pat: &Patch) -> Fallible<()> {
        if self.axes.iter().map(|a| &a.name).sorted().collect_vec()
            != pat.axes.iter().map(|a| &a.name).sorted().collect_vec()
        {
            return Err(StoiError::InvalidValue("The axes of two patches don't match (broadcasting is not supported yet so they must match exactly)"));
        }
        if self.dense.is_empty() || pat.dense.is_empty() {
            // It's a no op either way
            return Ok(());
        }

        // TODO: Support broadcasting smaller patches
        // TODO: Fast path for axes with one consecutive, identical order overlap

        //
        // 1: Align the axes (cheap)
        //

        // For every axis in self..
        let axis_shuffle: Vec<usize> = self
            .axes
            .iter()
            .map(|self_axis|
            // Look it up in pat
            pat.axes.iter().position(|pat_axis| self_axis.name == pat_axis.name).unwrap())
            .collect();
        // Now roll the tensor if necessary
        let shard = pat.dense.view().permuted_axes(&axis_shuffle[..]);
        let shard_axes = axis_shuffle
            .iter()
            .map(|&ax_ix| &pat.axes[ax_ix])
            .collect_vec();
        // Get rid of the reference to pat because we will just confuse ourselves otherwise.
        // Because it's axes don't match self.
        std::mem::drop(pat);

        // Create a new box large enough to hold either patch or self
        let max_shape = self
            .dense
            .shape()
            .iter()
            .zip(shard.shape().iter())
            .map(|(&x, &y)| x.max(y))
            .collect_vec();
        let mut union = ArrayD::from_elem(&max_shape[..], std::f32::NAN);

        // 2: Precompute the axes so we don't search on every element
        //    This also helps for optimizing the rectangle to copy
        //
        // label_shuffles =
        //  for each axis:
        //      for each label on self's patch:
        //          If pat has a matching label:
        //              The index into pat of the corresponding label
        //              or else None
        let mut label_shuffles = vec![];
        for ax_ix in 0..self.ndim() {
            let pat_label_to_idx: HashMap<Label, usize> = shard_axes[ax_ix]
                .labels()
                .iter()
                .copied()
                .enumerate()
                .map(|(i, l)| (l, i))
                .collect();
            label_shuffles.push(
                self.axes[ax_ix]
                    .labels()
                    .iter()
                    .map(|l| *pat_label_to_idx.get(l).unwrap_or(&std::usize::MAX))
                    .collect::<Vec<usize>>(),
            );
        }

        // 3. Shuffle the intersection into self-space and apply the patch
        //  - Fill the union box with the incoming patch
        //  - For each axis
        //    - Copy data from one to the other in label-shuffle order
        //    - Swap the buffers
        //  - Erase the planes that aren't used in self, so that they don't mutate
        {
            Self::merge_slice(shard.view(), union.view_mut(), shard.shape(), |x, y| {
                *x = *y
            });

            union = Self::shuffle_pull_ndim(
                union,
                &label_shuffles[..],
            );

            for (ax_ix, label_shuffle) in label_shuffles.iter().enumerate() {
                for (self_idx, pat_idx) in label_shuffle.iter().enumerate() {
                    if *pat_idx == std::usize::MAX {
                        union.index_axis_mut(nd::Axis(ax_ix), self_idx).fill(std::f32::NAN);
                    }
                }
            }
        }

        // 5. Now that all labels on all axes match, apply the patch
        let sh = self.dense.shape().to_owned();
        Self::merge_slice(union.view(), self.dense.view_mut(), &sh[..], |a, b| {
            if !b.is_nan() {
                *a = *b;
            }
        });
        Ok(())
    }

    /// Copy an N-d rectangle at the origin between incongruent arrays
    ///
    /// Make congruent slices on both sides and then assign()'s
    fn merge_slice<F: Fn(&mut f32, &f32)>(
        read: ArrayViewD<f32>,
        write: ArrayViewMutD<f32>,
        size: &[usize],
        merge: F,
    ) {
        let read_slice = size
            .iter()
            .enumerate()
            .fold(read, |mut rd, (ax_ix, &width)| {
                rd.slice_axis_inplace(nd::Axis(ax_ix), (0..width).into());
                rd
            });
        let mut write_slice = size
            .iter()
            .enumerate()
            .fold(write, |mut wr, (ax_ix, &width)| {
                wr.slice_axis_inplace(nd::Axis(ax_ix), (0..width).into());
                wr
            });

        write_slice.zip_mut_with(&read_slice, merge);
    }

    /// Copy (N-1)D planes, with possible replication
    ///
    /// "pull" here means there is one index for each write plane.
    /// As a result, you can make multiple copies of the each plane if you want.
    ///
    /// Use std::usize::MAX to skip a plane
    fn shuffle_pull(
        read: &ArrayViewD<f32>,
        write: &mut ArrayViewMutD<f32>,
        axis: nd::Axis,
        shuffle: &[usize],
    ) {
        assert_eq!(shuffle.len(), write.len_of(axis));
        for write_index in 0..shuffle.len() {
            if shuffle[write_index] != std::usize::MAX {
                write
                    .index_axis_mut(axis, write_index)
                    .assign(&read.index_axis(axis, shuffle[write_index]));
            }
        }
    }

    /// Shuffle each plane of all tensor axes, with possible replication
    ///
    /// "pull" here means there is one index for each write plane.
    /// As a result, you can make multiple copies of the each plane if you want.
    ///
    /// Use std::usize::MAX to skip a plane
    ///
    /// The results will be in "original" after this function completes
    fn shuffle_pull_ndim(
        mut original: ArrayD<f32>,
        shuffle: &[Vec<usize>],
    ) -> ArrayD<f32> {
        assert_eq!(original.ndim(), shuffle.len());
        let mut scratch = original.clone();
        for ax_ix in 0..original.ndim() {
            Self::shuffle_pull(
                &original.view(),
                &mut scratch.view_mut(),
                nd::Axis(ax_ix),
                &shuffle[ax_ix],
            );
            std::mem::swap(&mut original, &mut scratch);
        }
        original
    }

    /// Possibly compact the patch, removing unused labels
    ///
    /// You can compact a source patch but not a target patch for an apply().
    /// This is because if you compact the target, any data that would have been in the cut areas
    /// will not be copied (which is probably not what you want).
    ///
    /// Compacting will only occur if it saves at least 25% of the space, to save on copies.
    /// For this reason it works in-place, so a copy is not always necessary.
    ///
    ///     use stoicheia::{Axis, Patch};
    ///     use ndarray::arr2;
    ///     let mut p = Patch::new(vec![
    ///         Axis::range("a", 0..2),
    ///         Axis::range("b", 0..3)
    ///     ], arr2(&[
    ///         [ 3, 0, 5],
    ///         [ 0, 0, 0]
    ///     ]).into_dyn()).unwrap();
    ///
    ///     p.compact();
    ///
    ///     assert_eq!(
    ///         p.to_dense(),
    ///         arr2(&[
    ///             [3, 5]
    ///         ]).into_dyn());
    pub fn compact(&mut self) {
        // TODO: Profile if it's better to do one complex pass or multiple simple ones

        // This is a ragged matrix, not a tensor
        // It's (ndim, len-of-that-dim), and represents if we are going to keep that slice
        let mut keep: Vec<Vec<bool>> = self
            .dense
            .shape()
            .iter()
            .map(|&len| vec![false; len])
            .collect();

        // Scan the tensor to check if it's empty
        for (point, value) in self.dense.indexed_iter() {
            for ax_ix in 0..self.ndim() {
                keep[ax_ix][point[ax_ix]] |= !value.is_zero();
            }
        }

        let keep_indices: Vec<Vec<usize>> = keep
            .iter()
            .map(|v| {
                v.iter()
                    .enumerate()
                    .filter_map(|(i, &k)| if k { Some(i) } else { None })
                    .collect_vec()
            })
            .collect();
        // The total number of elements in the new patch
        let mut keep_lens: Vec<(usize, usize)> = keep_indices
            .iter()
            .map(|indices| indices.len())
            .enumerate()
            .collect();
        let total_new_elements: usize = keep.iter().map(|indices| indices.len()).product();

        // If the juice is worth the squeeze
        if total_new_elements / 3 >= self.dense.len() / 4 {
            // Remove the most selective axes first
            keep_lens.sort_unstable_by_key(|&(ax_ix, ct)| ct / self.dense.len_of(nd::Axis(ax_ix)));
            for (ax_ix, _count) in keep_lens {
                // Delete labels
                self.axes[ax_ix] = Axis::new_unchecked(
                    &self.axes[ax_ix].name,
                    keep_indices[ax_ix]
                        .iter()
                        .map(|&i| self.axes[ax_ix].labels()[i])
                        .collect(),
                );

                // Delete elements
                self.dense = self.dense.select(nd::Axis(ax_ix), &keep_indices[ax_ix]);
            }
        }
    }

    /// Render the patch as a dense array. This always copies the data.
    pub fn to_dense(&self) -> nd::ArrayD<f32> {
        self.dense.clone()
    }

    /// Get a reference to the content
    pub fn content(&self) -> nd::ArrayViewD<f32> {
        self.dense.view()
    }

    /// Get a mutable reference to the content
    pub fn content_mut(&mut self) -> nd::ArrayViewMutD<f32> {
        self.dense.view_mut()
    }

    /// Get a shared reference to the axes within
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use ndarray as nd;

    #[test]
    fn patch_1d_apply() {
        let item_axis = Axis::new("item", vec![1, 3]).unwrap();

        // Set both elements
        let mut base = Patch::try_from_axes(vec![&item_axis]).unwrap();
        let revision =
            Patch::new(vec![item_axis.clone()], nd::arr1(&[100., 300.]).into_dyn()).unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert_eq!(modified[[1]], 300.);

        // Set one but miss the other
        let mut base = Patch::try_from_axes(vec![&item_axis]).unwrap();
        let revision = Patch::new(
            vec![Axis::new("item", vec![1, 2]).unwrap()],
            nd::arr1(&[100., 300.]).into_dyn(),
        )
        .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert!(modified[[1]].is_nan());

        // Miss both
        let mut base = Patch::try_from_axes(vec![&item_axis]).unwrap();
        let revision = Patch::new(
            vec![("item", vec![10, 30])],
            nd::arr1(&[100., 300.]).into_dyn(),
        )
        .unwrap();
        base.apply(&revision).unwrap();

        let modified = base.to_dense();
        assert!(modified[[0]].is_nan());
        assert!(modified[[1]].is_nan());

        // Unsorted labels
        let mut base = Patch::try_from_axes(vec![
            ("item", vec![30, 10])
        ]).unwrap();
        let revision = Patch::new(
            vec![("item", vec![30, 10])],
            nd::arr1(&[100., 300.]).into_dyn(),
        )
        .unwrap();
        base.apply(&revision).unwrap();

        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert_eq!(modified[[1]], 300.);

        // Unsorted, mismatched labels
        let mut base = Patch::try_from_axes(vec![("item", vec![30, 10])]).unwrap();
        let revision = Patch::new(
            vec![("item", vec![10, 30])],
            nd::arr1(&[300., 100.]).into_dyn(),
        )
        .unwrap();
        base.apply(&revision).unwrap();

        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert_eq!(modified[[1]], 300.);
    }

    #[test]
    fn patch_2d_apply() {
        let item_axis = Axis::new("item", vec![1, 3]).unwrap();
        let store_axis = Axis::new("store", vec![-1, -3]).unwrap();

        // Perfect patch, same order and same overlap
        let mut base = Patch::try_from_axes(vec![&item_axis, &store_axis]).unwrap();
        let revision = Patch::new(
            vec![&item_axis, &store_axis],
            nd::arr2(&[[100., 200.], [300., 400.]]).into_dyn(),
        )
        .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert_eq!(modified[[0, 0]], 100.);
        assert_eq!(modified[[0, 1]], 200.);
        assert_eq!(modified[[1, 0]], 300.);
        assert_eq!(modified[[1, 1]], 400.);
    }
}
