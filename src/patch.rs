use crate::{Axis, Fallible, Label, StoiError};
use itertools::Itertools;
use ndarray::ArrayD;
use num_traits::Zero;
use std::collections::HashMap;

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
///   - A regular array of some datatype in the same order as the list of axes
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct Patch<Elem: Copy + Default + Zero> {
    /// Names and labels of the axes
    axes: Vec<Axis>,
    /// Tensor containing all the elements of this patch
    dense: ArrayD<Elem>,
}

impl<Elem: Copy + Default + Zero> Patch<Elem> {
    /// Create a new patch from an array and some labels
    ///
    /// The labels must be in sorted order (within the patch), if not they will be sorted.
    /// The axes dimensions must match the dense array's dimensions, otherwise it will error.
    pub fn new(axes: Vec<Axis>, dense: ArrayD<Elem>) -> Fallible<Self> {
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

        Ok(Self { axes, dense })
    }

    /// Create a new empty patch given some axes
    pub fn from_axes(axes: Vec<Axis>) -> Fallible<Self> {
        let dims = axes.iter().map(|a| a.len()).collect_vec();
        if dims.iter().product::<usize>() > 256 << 20 {
            return Err(StoiError::TooLarge(
                "Patches must be 256 million elements or less (1GB of 32bit floats)",
            ));
        }
        Self::new(axes, ArrayD::default(dims))
    }

    /// How many dimensions in this patch
    pub fn ndim(&self) -> usize {
        self.dense.ndim()
    }

    /// Apply another patch to this one, changing `self` where it overlaps with `pat`.
    ///
    /// This is not the same as merging the patches, because this only changes `self` where it
    /// overlaps with `pat`, and won't allocate or expand either one.
    pub fn apply(&mut self, pat: &Patch<Elem>) -> Fallible<()> {
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
        // TODO: Avoid cloning the axes
        let shard_axes = axis_shuffle
            .iter()
            .map(|&ax_ix| pat.axes[ax_ix].clone())
            .collect_vec();
        // Get rid of the reference to pat because we will just confuse ourselves otherwise.
        // Because it's axes don't match self.
        std::mem::drop(pat);

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
        for ax_ix in 0..shard.ndim() {
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
                    .map(|l| pat_label_to_idx.get(l).copied())
                    .collect::<Vec<Option<usize>>>(),
            );
        }

        //
        // TODO: Slice off any excess and try to copy a patch as a single dense array
        //

        //
        // Assign every element. This may need to change for some optimizations later.
        //
        let mut pat_point = vec![0usize; shard.ndim()];
        'cell: for (self_point, self_value) in self.dense.indexed_iter_mut() {
            // Fill in the location to read from in patch
            for ax_ix in 0..shard.ndim() {
                if let Some(pat_idx) = label_shuffles[ax_ix][self_point[ax_ix]] {
                    // This cell in self has a buddy in pat
                    pat_point[ax_ix] = pat_idx;
                } else {
                    // This cell has no buddy
                    continue 'cell;
                }
                // TODO: maybe uget when this looks stable
                *self_value = shard[&pat_point[..]];
            }
        }
        Ok(())
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
    pub fn to_dense(&self) -> nd::ArrayD<Elem> {
        self.dense.clone()
    }

    /// Get a reference to the content
    pub fn content(&self) -> nd::ArrayViewD<Elem> {
        self.dense.view()
    }

    /// Get a mutable reference to the content
    pub fn content_mut(&mut self) -> nd::ArrayViewMutD<Elem> {
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
        let item_axis = Axis::new("item", vec![Label(1), Label(3)]).unwrap();

        // Set both elements
        let mut base = Patch::from_axes(vec![item_axis.clone()]).unwrap();
        let revision =
            Patch::new(vec![item_axis.clone()], nd::arr1(&[100., 300.]).into_dyn()).unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert_eq!(modified[[1]], 300.);

        // Set one but miss the other
        let mut base = Patch::from_axes(vec![item_axis.clone()]).unwrap();
        let revision = Patch::new(
            vec![Axis::new("item", vec![Label(1), Label(2)]).unwrap()],
            nd::arr1(&[100., 300.]).into_dyn(),
        )
        .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert_eq!(modified[[1]], 0.);

        // Miss both
        let mut base = Patch::from_axes(vec![item_axis.clone()]).unwrap();
        let revision = Patch::new(
            vec![Axis::new("item", vec![Label(10), Label(30)]).unwrap()],
            nd::arr1(&[100., 300.]).into_dyn(),
        )
        .unwrap();
        base.apply(&revision).unwrap();

        let modified = base.to_dense();
        assert_eq!(modified[[0]], 0.);
        assert_eq!(modified[[1]], 0.);

        // Unsorted labels
        let mut base =
            Patch::from_axes(vec![Axis::new("item", vec![Label(30), Label(10)]).unwrap()]).unwrap();
        let revision = Patch::new(
            vec![Axis::new("item", vec![Label(30), Label(10)]).unwrap()],
            nd::arr1(&[100., 300.]).into_dyn(),
        )
        .unwrap();
        base.apply(&revision).unwrap();

        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert_eq!(modified[[1]], 300.);

        // Unsorted, mismatched labels
        let mut base =
            Patch::from_axes(vec![Axis::new("item", vec![Label(30), Label(10)]).unwrap()]).unwrap();
        let revision = Patch::new(
            vec![Axis::new("item", vec![Label(10), Label(30)]).unwrap()],
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
        let item_axis = Axis::new("item", vec![Label(1), Label(3)]).unwrap();
        let store_axis = Axis::new("store", vec![Label(-1), Label(-3)]).unwrap();

        // Set along both axes
        let mut base = Patch::from_axes(vec![item_axis.clone(), store_axis.clone()]).unwrap();
        let revision = Patch::new(
            vec![item_axis.clone(), store_axis.clone()],
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
