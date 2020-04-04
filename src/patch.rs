use crate::{Axis, Fallible, Label, StoiError};
use arrayvec::ArrayVec;
use itertools::Itertools;
use ndarray as nd;
use ndarray::{
    Array4, ArrayD, ArrayView, ArrayView4, ArrayViewMut, ArrayViewMut4,
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Read, Write};

type A4D = ArrayVec<[usize; 4]>;

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
    dense: Array4<f32>,
    // TODO: Bounding box for this patch on the global axes.
    // - Includes an ID for the catalog, to prevent using them with the wrong catalog
    // - If present, then the axes also match the order of the global axes
}

impl Patch {
    /// Create a new patch from an array and some labels
    ///
    /// The axes dimensions must match the dense array's dimensions, otherwise it will error.
    fn new_4d(axes: Vec<Axis>, content: Option<Array4<f32>>) -> Fallible<Self> {
        if axes.len() > 4 {
            return Err(StoiError::MisalignedAxes(
                "Patch must have up to 4 axes".into(),
            ));
        }

        match content {
            None => {
                // They have not provided content; we must allocate
                let mut dims = axes.iter().map(|a| a.len()).collect_vec();
                let dims_size: usize = dims.iter().product::<usize>();
                if dims_size > 256 << 20 {
                    return Err(StoiError::TooLarge(
                        "Patches must be 256 million elements or less (1GB of 32bit floats)",
                    ));
                }
                // Add empty dimensions where necessary
                while dims.len() < 4 {
                    dims.push(1);
                }

                Ok(Self {
                    axes,
                    dense: Array4::from_elem((dims[0], dims[1], dims[2], dims[3]), std::f32::NAN),
                })
            }
            Some(dense) => {
                // They provided some content
                if axes.len() > dense.ndim() {
                    return Err(StoiError::InvalidValue(
                        "Too many labeled axes for the axes in the dense tensor.",
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
                let mut dims = dense.shape().to_vec();
                // Add empty dimensions where necessary
                while dims.len() < 4 {
                    dims.push(1);
                }

                Ok(Self {
                    axes,
                    dense: dense
                        .into_shape((dims[0], dims[1], dims[2], dims[3]))
                        .unwrap(), // shape error is impossible here
                })
            }
        }
    }

    /// Create a new patch from an array and some labels
    ///
    /// The axes dimensions must match the dense array's dimensions, otherwise it will error.
    pub fn new(axes: Vec<Axis>, content: Option<ArrayD<f32>>) -> Fallible<Self> {
        if axes.is_empty() {
            return Err(StoiError::MisalignedAxes(
                "Patches must have at least one axis".into(),
            ));
        }

        match content {
            None => {
                // They have not provided content; we must allocate
                let mut dims = axes.iter().map(|a| a.len()).collect_vec();
                let dims_size: usize = dims.iter().product::<usize>();
                if dims_size > 256 << 20 {
                    return Err(StoiError::TooLarge(
                        "Patches must be 256 million elements or less (1GB of 32bit floats)",
                    ));
                }
                // Add empty dimensions where necessary
                while dims.len() < 4 {
                    dims.push(1);
                }

                Ok(Self {
                    axes,
                    dense: Array4::from_elem((dims[0], dims[1], dims[2], dims[3]), std::f32::NAN),
                })
            }
            Some(dense) => {
                // They provided some content
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
                let mut dims = dense.shape().to_vec();
                // Add empty dimensions where necessary
                while dims.len() < 4 {
                    dims.push(1);
                }

                Ok(Self {
                    axes,
                    dense: dense
                        .into_shape((dims[0], dims[1], dims[2], dims[3]))
                        .unwrap(), // shape error is impossible here
                })
            }
        }
    }

    /// Convenience method to create a builder
    pub fn build() -> PatchBuilder {
        PatchBuilder::new()
    }

    /// How many dimensions in this patch
    ///
    /// It's always stored as 4D but this is how many it logically has
    pub fn ndim(&self) -> usize {
        self.axes.len()
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

        // For each of the four axes, give the corresponding other axis
        // Any missing axes are just 1's and don't have labels
        let mut axis_shuffle = [0usize; 4];
        for self_ax_ix in 0..4 {
            axis_shuffle[self_ax_ix] = match self.axes.get(self_ax_ix) {
                Some(self_axis) => pat
                    .axes
                    .iter()
                    .position(|pat_axis| self_axis.name == pat_axis.name)
                    .unwrap(),
                None => self_ax_ix
            };
        }
        // Now roll the tensor if necessary
        let shard = pat.dense.view().permuted_axes(axis_shuffle);
        let shard_axes = axis_shuffle
            .iter()
            .filter_map(|&ax_ix| pat.axes.get(ax_ix))
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
            .collect::<A4D>()
            .into_inner()
            .unwrap();
        let mut union = Array4::from_elem(max_shape, std::f32::NAN);

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
        for ax_ix in 0..4 {
            if ax_ix < self.ndim() {
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
            } else {
                label_shuffles.push(vec![0]);
            }
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

            union = self.shuffle_pull_ndim(union, &label_shuffles[..]);

            for (ax_ix, label_shuffle) in label_shuffles.iter().enumerate() {
                for (self_idx, pat_idx) in label_shuffle.iter().enumerate() {
                    if *pat_idx == std::usize::MAX {
                        union
                            .index_axis_mut(nd::Axis(ax_ix), self_idx)
                            .fill(std::f32::NAN);
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
        read: ArrayView4<f32>,
        write: ArrayViewMut4<f32>,
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

    /// Shuffle each plane of all tensor axes, with possible replication
    ///
    /// "pull" here means there is one index for each write plane.
    /// As a result, you can make multiple copies of the each plane if you want.
    ///
    /// Use std::usize::MAX to skip a plane
    fn shuffle_pull_ndim(&self, original: Array4<f32>, shuffles: &[Vec<usize>]) -> Array4<f32> {
        assert!(shuffles.len() == original.ndim());
        let mut scratch = original.clone();
        match self.ndim() {
            4 => Self::shuffle_pull_4d(&original.view(), &mut scratch.view_mut(), &shuffles[..]),
            3 => Self::shuffle_pull_3d(
                &original.index_axis(nd::Axis(3), 0),
                &mut scratch.index_axis_mut(nd::Axis(3), 0), 
                &shuffles[..shuffles.len()-1]),
            2 => Self::shuffle_pull_2d(
                &original
                    .index_axis(nd::Axis(3), 0)
                    .index_axis(nd::Axis(2), 0),
                &mut scratch
                    .index_axis_mut(nd::Axis(3), 0)
                    .index_axis_mut(nd::Axis(2), 0), 
                &shuffles[..shuffles.len()-2]),
            1 => Self::shuffle_pull_1d(
                    &original
                        .index_axis(nd::Axis(3), 0)
                        .index_axis(nd::Axis(2), 0)
                        .index_axis(nd::Axis(1), 0),
                    &mut scratch
                        .index_axis_mut(nd::Axis(3), 0)
                        .index_axis_mut(nd::Axis(2), 0)
                        .index_axis_mut(nd::Axis(1), 0), 
                    &shuffles[0]),
            _ => panic!("Invalid Patch! Can't have more than 4 dimensions!")
        }
        //original
        scratch
    }

    /// Shuffle each plane of all tensor axes, with possible replication
    ///
    /// "pull" here means there is one index for each write plane.
    /// As a result, you can make multiple copies of the each plane if you want.
    ///
    /// Use std::usize::MAX to skip a plane
    ///
    /// The results will be in "original" after this function completes
    fn shuffle_pull_4d(
        read: &ArrayView<f32, nd::Ix4>,
        write: &mut ArrayViewMut<f32, nd::Ix4>,
        shuffles: &[Vec<usize>],
    ) {
        assert_eq!(shuffles.len(), 4);

        // Not last dimension: recursive copy
        let ax0 = nd::Axis(0);
        let shuf0 = &shuffles[0];
        assert!(shuf0.len() <= write.len_of(ax0));
        for write_index in 0..shuf0.len() {
            if shuf0[write_index] != std::usize::MAX {
                Self::shuffle_pull_3d(
                    &read.index_axis(ax0, shuf0[write_index]),
                    &mut write.index_axis_mut(ax0, write_index),
                    &shuffles[1..],
                )
            }
        }
    }

    /// Shuffle each plane of all tensor axes, with possible replication
    /// see shuffle_pull_4d.
    fn shuffle_pull_3d(
        read: &ArrayView<f32, nd::Ix3>,
        write: &mut ArrayViewMut<f32, nd::Ix3>,
        shuffles: &[Vec<usize>],
    ) {
        assert_eq!(shuffles.len(), 3);

        // Not last dimension: recursive copy
        let ax0 = nd::Axis(0);
        let shuf0 = &shuffles[0];
        assert!(shuf0.len() <= write.len_of(ax0));
        for write_index in 0..shuf0.len() {
            if shuf0[write_index] != std::usize::MAX {
                Self::shuffle_pull_2d(
                    &read.index_axis(ax0, shuf0[write_index]),
                    &mut write.index_axis_mut(ax0, write_index),
                    &shuffles[1..],
                )
            }
        }
    }

    /// Shuffle each plane of all tensor axes, with possible replication
    /// see shuffle_pull_4d.
    fn shuffle_pull_2d(
        read: &ArrayView<f32, nd::Ix2>,
        write: &mut ArrayViewMut<f32, nd::Ix2>,
        shuffles: &[Vec<usize>],
    ) {
        assert_eq!(shuffles.len(), 2);

        // Not last dimension: recursive copy
        let ax0 = nd::Axis(0);
        let shuf0 = &shuffles[0];
        assert!(shuf0.len() <= write.len_of(ax0));
        for write_index in 0..shuf0.len() {
            if shuf0[write_index] != std::usize::MAX {
                Self::shuffle_pull_1d(
                    &read.index_axis(ax0, shuf0[write_index]),
                    &mut write.index_axis_mut(ax0, write_index),
                    &shuffles[1],
                )
            }
        }
    }

    /// Copy (N-1)D planes, with possible replication
    ///
    /// "pull" here means there is one index for each write plane.
    /// As a result, you can make multiple copies of the each plane if you want.
    ///
    /// Use std::usize::MAX to skip a plane
    fn shuffle_pull_1d(
        read: &ArrayView<f32, nd::Ix1>,
        write: &mut ArrayViewMut<f32, nd::Ix1>,
        shuffle: &[usize],
    ) {
        // Note: it's totally OK if some of the patch is not written
        // I'm not sure what the preference would be if shuffle is larger
        // so let's just jump out then
        assert!(shuffle.len() <= write.len());
        for write_index in 0..shuffle.len() {
            if shuffle[write_index] != std::usize::MAX {
                write[write_index] = read[shuffle[write_index]];
            }
        }
    }

    /// Merge two patches together into a larger patch
    ///
    /// This is actually pretty simple, it works by creating a new Patch and applying
    /// all of the patches to it.
    pub fn merge(operands: &[&Patch]) -> Fallible<Patch> {
        if operands.is_empty() {
            return Err(StoiError::InvalidValue(
                "Empty merge. There is no identity for patches because it's not clear what the axes would be.",
            ));
        }

        // There must have been a first one and it must have had axes
        let mut axes = operands[0].axes().iter().cloned().collect_vec();
        for operand in &operands[1..] {
            if !operand
                .axes()
                .iter()
                .map(|ax| &ax.name)
                .eq(axes.iter().map(|ax| &ax.name))
            {
                return Err(StoiError::InvalidValue(
                    "Unmatched axes. All operands of Patch::merge() must have the same axis names in the same order.",
                ));
            }
            for (ax_ix, axis) in operand.axes().into_iter().enumerate() {
                axes[ax_ix].union(&axis); // In-place
            }
        }
        let mut target = Patch::new(axes, None)?;
        for operand in operands {
            target.apply(operand)?;
        }
        Ok(target)
    }

    /// Split a patch in half if it's larger than it probably should be.
    ///
    /// This
    ///
    /// Accepts:
    ///     long_axis: the global axis to split in half
    ///
    /// Returns:
    ///     Either: A vec with one element, which is a Cow::Borrowed(&self)
    ///     Or: A vec with 2+ elements, which are all Cow::Owned(Patch)
    pub fn maybe_split(&self, global_long_axis: &Axis) -> Fallible<Vec<Cow<Patch>>> {
        if let Some((long_ax_ix, long_axis)) = self
            .axes()
            .iter()
            .enumerate()
            .find(|a| a.1.name == global_long_axis.name)
        {
            if self.dense.len() < 4 << 20 {
                // Looks good to me
                Ok(vec![Cow::Borrowed(&self)])
            } else {
                // This is a heuristic and it could use more serious study
                let long_axis_labelset: HashMap<Label, usize> = long_axis
                    .labels()
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(a, b)| (b, a))
                    .collect();

                let global_locations = global_long_axis
                    .labels()
                    .iter()
                    .filter_map(|global_label| long_axis_labelset.get(global_label))
                    .copied()
                    .collect_vec();

                if global_locations.len() < long_axis_labelset.len() {
                    return Err(StoiError::MisalignedAxes(
                        "Patch contains labels not present in the global axis. 
                        Always union global axes against patch axes before splitting a patch,
                        because otherwise it's not clear what the Patch's bounding box would be."
                            .into(),
                    ));
                }

                // The important part - split the long axis in half according to the global axis order
                let (left_patch_indices, right_patch_indices) =
                    global_locations.split_at(global_locations.len() / 2);

                let patches = [left_patch_indices, right_patch_indices][..]
                    .into_iter()
                    .map(|indices| {
                        let mut axes = self.axes.clone();
                        // Replace the long axis
                        axes[long_ax_ix] = Axis::new_unchecked(
                            &long_axis.name,
                            indices
                                .iter()
                                .map(|ix| long_axis.labels()[*ix])
                                .collect_vec(),
                        );
                        // Slice the patch
                        Cow::Owned(
                            Patch::new(
                                axes,
                                Some(self.content().select(nd::Axis(long_ax_ix), indices)),
                            )
                            .unwrap(),
                        )
                    })
                    .collect_vec();
                Ok(patches)
            }
        } else {
            return Err(StoiError::MisalignedAxes(
                "Patch doesn't contain the global axis provided".into(),
            ));
        }
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
    ///     assert_eq!(
    ///         p.compact().to_dense(),
    ///         arr2(&[
    ///             [3, 5]
    ///         ]).into_dyn());
    pub fn compact(&self) -> Cow<Self> {
        // This is a ragged matrix, not a tensor
        // It's (ndim, len-of-that-dim), and represents if we are going to keep that slice
        let keep_indices = (0..self.ndim())
            .map(|ax_ix| {
                self.dense
                    .axis_iter(nd::Axis(ax_ix))
                    .map(|plane| plane.fold(false, |acc, x| acc || !x.is_nan()))
                    .enumerate()
                    .filter_map(|(i, a)| if a { Some(i) } else { None })
                    .collect_vec()
            })
            .collect_vec();

        // The total number of elements in the new patch
        let mut keep_lens: Vec<(usize, usize)> = keep_indices
            .iter()
            .map(|indices| indices.len())
            .enumerate()
            .collect();
        let total_new_elements: f32 = keep_indices
            .iter()
            .map(|indices| indices.len() as f32)
            .product();

        // If the juice is worth the squeeze
        if total_new_elements < self.dense.len() as f32 * 0.75 {
            // Remove the most selective axes first
            keep_lens.sort_unstable_by_key(|&(ax_ix, ct)| ct / self.dense.len_of(nd::Axis(ax_ix)));
            let mut dense = Cow::Borrowed(&self.dense);
            let new_axes = self
                .axes
                .iter()
                .enumerate()
                .map(|(ax_ix, axis)| {
                    // Delete elements
                    dense = Cow::Owned(dense.select(nd::Axis(ax_ix), &keep_indices[ax_ix]));

                    // Delete labels
                    Axis::new_unchecked(
                        &axis.name,
                        keep_indices[ax_ix]
                            .iter()
                            .map(|&i| axis.labels()[i])
                            .collect(),
                    )
                })
                .collect_vec();
            Cow::Owned(Patch::new_4d(new_axes, Some(dense.into_owned())).unwrap())
        } else {
            Cow::Borrowed(self)
        }
    }

    /// Render the patch as a dense array. This always copies the data.
    pub fn to_dense(&self) -> nd::ArrayD<f32> {
        self.dense
            .clone()
            .into_dyn()
            .into_shape(&self.dense.shape()[..self.ndim()])
            .unwrap()
    }

    /// Get a reference to the content
    pub fn content(&self) -> nd::ArrayViewD<f32> {
        self.dense
            .view()
            .into_dyn()
            .into_shape(&self.dense.shape()[..self.ndim()])
            .unwrap()
    }

    /// Get a mutable reference to the content
    pub fn content_mut(&mut self) -> nd::ArrayViewMutD<f32> {
        let logical_shape = &self.dense.shape()[..self.ndim()].to_vec();
        self.dense
            .view_mut()
            .into_dyn()
            .into_shape(&logical_shape[..])
            .unwrap()
    }

    /// Get a shared reference to the axes within
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Serialize a patch the default way
    ///
    /// It's still possible to serialize a patch with serde, but this is the
    /// recommended method if you don't have reason to do otherwise, to avoid
    /// needless incompatibilities
    pub fn serialize_into<W: Write>(
        &self,
        compression: Option<PatchCompressionType>,
        mut buffer: &mut W,
    ) -> Fallible<()> {
        let compression = compression.unwrap_or(PatchCompressionType::Off);
        let options = PatchTag {
            magic: 0x494f5453, // "STOI"
            version: 1,
            compression,
            filters: vec![],
        };
        bincode::serialize_into(&mut buffer, &options)?;

        match options.compression {
            PatchCompressionType::Off => {
                bincode::serialize_into(&mut buffer, &self)?;
                Ok(())
            }
            PatchCompressionType::Brotli { quality } => {
                let mut brotli_writer = brotli::CompressorWriter::new(
                    &mut buffer,
                    4096,    /* Buffer size */
                    quality, /* Quality: 0-9 */
                    20,      /* Log2 buffer size */
                );

                bincode::serialize_into(&mut brotli_writer, &self)?;
                brotli_writer.flush()?;
                Ok(())
            }
            PatchCompressionType::LZ4 { quality } => {
                let mut lz4_writer = lz4::EncoderBuilder::new()
                    .level(quality)
                    .build(&mut buffer)?;

                bincode::serialize_into(&mut lz4_writer, &self)?;
                lz4_writer.finish().1?;

                Ok(())
            }
        }
    }

    /// Serialize the default way, into a fresh new Vec
    ///
    /// While this method is convenient, patches are usually pretty large, so
    /// try to use serialize_into and reuse buffers where possible.
    pub fn serialize(&self, compression: Option<PatchCompressionType>) -> Fallible<Vec<u8>> {
        let mut buffer = vec![0u8; 0];
        buffer.reserve(self.dense.len() * 5); // Save time allocating
        self.serialize_into(compression, &mut buffer)?;
        Ok(buffer)
    }

    /// Deserialize a patch the default way
    ///
    /// It's still possible to deserialize a patch with serde, but this is the
    /// recommended method if you don't have reason to do otherwise, to avoid
    /// needless incompatibilities.
    pub fn deserialize_from<R: Read>(mut buffer: R) -> Fallible<Self> {
        let options: PatchTag = bincode::deserialize_from(buffer.by_ref())?;

        match options.compression {
            PatchCompressionType::Off => Ok(bincode::deserialize_from(buffer)?),
            PatchCompressionType::Brotli { quality: _ } => {
                let brotli_reader = brotli::Decompressor::new(buffer, 4096);
                Ok(bincode::deserialize_from(brotli_reader)?)
            }
            PatchCompressionType::LZ4 { quality: _ } => {
                let lz4_reader = lz4::Decoder::new(buffer)?;
                Ok(bincode::deserialize_from(lz4_reader)?)
            }
        }
    }
}

/// An uncompressed prelude to Patch, to allow versions and serialization options
#[derive(Serialize, Deserialize, Debug, Clone)]
struct PatchTag {
    magic: u32,
    version: u8,
    compression: PatchCompressionType,
    filters: Vec<PatchFilter>,
}
/// Part of PatchTag, used for deserializing patches
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum PatchCompressionType {
    Off,
    Brotli { quality: u32 },
    LZ4 { quality: u32 },
}
/// Things you might have done to the patch to try to save space
/// There aren't any yet but it could happen and this lets us be compatible
#[derive(Serialize, Deserialize, Debug, Clone)]
enum PatchFilter {}

/// Convenience class to build patches with less typing
pub struct PatchBuilder {
    axes: Vec<Result<Axis, StoiError>>,
}
impl PatchBuilder {
    /// Create an empty Patch builder
    pub fn new() -> Self {
        Self { axes: vec![] }
    }
    /// Add an axis to the upcoming Patch
    pub fn axis(mut self, name: &str, labels: &[Label]) -> Self {
        self.axes.push(Axis::new(name, labels.to_vec()));
        self
    }
    /// Add a range-based axis to the upcoming Patch
    pub fn axis_range<R: IntoIterator<Item = Label>>(mut self, name: &str, range: R) -> Self {
        self.axes.push(Axis::new(name, range.into_iter().collect()));
        self
    }

    /// Give the content and finish
    pub fn content<J: Into<Option<ArrayD<f32>>>>(self, content: J) -> Fallible<Patch> {
        Patch::new(
            self.axes.into_iter().collect::<Fallible<Vec<Axis>>>()?,
            content.into().map(|c| c.into_dyn()),
        )
    }

    /// Create a 1d array on the spot, set the content, and return the new patch
    pub fn content_1d(self, content: &[f32]) -> Fallible<Patch> {
        Patch::new(
            self.axes.into_iter().collect::<Fallible<Vec<Axis>>>()?,
            Some(nd::arr1(content).into_dyn()),
        )
    }

    /// Create a 2d array on the spot, set the content, and return the new patch
    pub fn content_2d<V: nd::FixedInitializer<Elem = f32> + Clone>(
        self,
        content: &[V],
    ) -> Fallible<Patch> {
        Patch::new(
            self.axes.into_iter().collect::<Fallible<Vec<Axis>>>()?,
            Some(nd::arr2(content).into_dyn()),
        )
    }
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn patch_1d_apply_total_overlap_same_order() {
        // Set both elements
        let mut base = Patch::build().axis("item", &[1, 3]).content(None).unwrap();
        let revision = Patch::build()
            .axis("item", &[1, 3])
            .content_1d(&[100., 300.])
            .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert_eq!(modified[[1]], 300.);
    }

    #[test]
    fn patch_1d_apply_semi_overlap_same_order() {
        // Set one but miss the other
        let mut base = Patch::build().axis("item", &[1, 3]).content(None).unwrap();
        let revision = Patch::build()
            .axis("item", &[1, 2])
            .content_1d(&[100., 300.])
            .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert!(modified[[1]].is_nan());
    }

    #[test]
    fn patch_1d_apply_no_overlap_different_order() {
        // Miss both
        let mut base = Patch::build().axis("item", &[1, 3]).content(None).unwrap();
        let revision = Patch::build()
            .axis("item", &[30, 10])
            .content_1d(&[100., 300.])
            .unwrap();
        base.apply(&revision).unwrap();

        let modified = base.to_dense();
        assert!(modified[[0]].is_nan());
        assert!(modified[[1]].is_nan());
    }

    #[test]
    fn patch_1d_apply_total_overlap_same_order_unsorted() {
        // Unsorted labels
        let mut base = Patch::build()
            .axis("item", &[30, 10])
            .content(None)
            .unwrap();
        let revision = Patch::build()
            .axis("item", &[30, 10])
            .content_1d(&[100., 300.])
            .unwrap();
        base.apply(&revision).unwrap();

        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert_eq!(modified[[1]], 300.);
    }

    #[test]
    fn patch_1d_apply_total_overlap_different_order() {
        // Unsorted, mismatched labels
        let mut base = Patch::build()
            .axis("item", &[30, 10])
            .content(None)
            .unwrap();
        let revision = Patch::build()
            .axis("item", &[10, 30])
            .content_1d(&[300., 100.])
            .unwrap();
        base.apply(&revision).unwrap();

        let modified = base.to_dense();
        assert_eq!(modified[[0]], 100.);
        assert_eq!(modified[[1]], 300.);
    }

    #[test]
    fn patch_2d_apply_same_size_total_overlap_same_order() {
        // Perfect patch: (same size) (total overlap) (same order)
        let mut base = Patch::build()
            .axis("item", &[1, 3])
            .axis("store", &[1, 3])
            .content(None)
            .unwrap();
        let revision = Patch::build()
            .axis("item", &[1, 3])
            .axis("store", &[1, 3])
            .content_2d(&[[100., 200.], [300., 400.]])
            .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert_eq!(modified[[0, 0]], 100.);
        assert_eq!(modified[[0, 1]], 200.);
        assert_eq!(modified[[1, 0]], 300.);
        assert_eq!(modified[[1, 1]], 400.);
    }

    #[test]
    fn patch_2d_apply_same_size_total_overlap_different_order() {
        let mut base = Patch::build()
            .axis("item", &[1, 3])
            .axis("store", &[1, 3])
            .content(None)
            .unwrap();
        let revision = Patch::build()
            .axis("item", &[1, 3])
            .axis("store", &[3, 1])
            .content_2d(&[[200., 100.], [400., 300.]])
            .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert_eq!(modified[[0, 0]], 100.);
        assert_eq!(modified[[0, 1]], 200.);
        assert_eq!(modified[[1, 0]], 300.);
        assert_eq!(modified[[1, 1]], 400.);
    }

    #[test]
    fn patch_2d_apply_same_size_semi_overlap_same_order() {
        let mut base = Patch::build()
            .axis("item", &[1, 3])
            .axis("store", &[1, 3])
            .content(None)
            .unwrap();
        let revision = Patch::build()
            .axis("item", &[2, 3])
            .axis("store", &[1, 3])
            .content_2d(&[[100., 200.], [300., 400.]])
            .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert!(modified[[0, 0]].is_nan());
        assert!(modified[[0, 1]].is_nan());
        assert_eq!(modified[[1, 0]], 300.);
        assert_eq!(modified[[1, 1]], 400.);
    }

    #[test]
    fn patch_2d_apply_same_size_semi_overlap_different_order() {
        let mut base = Patch::build()
            .axis("item", &[1, 3])
            .axis("store", &[1, 3])
            .content(None)
            .unwrap();
        let revision = Patch::build()
            .axis("item", &[2, 3])
            .axis("store", &[3, 1])
            .content_2d(&[[200., 100.], [400., 300.]])
            .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert!(modified[[0, 0]].is_nan());
        assert!(modified[[0, 1]].is_nan());
        assert_eq!(modified[[1, 0]], 300.);
        assert_eq!(modified[[1, 1]], 400.);
    }

    #[test]
    fn patch_2d_apply_same_size_no_overlap() {
        let mut base = Patch::build()
            .axis("item", &[1, 3])
            .axis("store", &[1, 3])
            .content(None)
            .unwrap();
        let revision = Patch::build()
            .axis("item", &[2, 4])
            .axis("store", &[3, 1])
            .content_2d(&[[200., 100.], [400., 300.]])
            .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert!(modified[[0, 0]].is_nan());
        assert!(modified[[0, 1]].is_nan());
        assert!(modified[[1, 0]].is_nan());
        assert!(modified[[1, 1]].is_nan());
    }

    #[test]
    fn patch_2d_apply_larger_patch_semi_overlap_different_order() {
        let mut base = Patch::build()
            .axis("item", &[1, 3])
            .axis("store", &[1, 3])
            .content(None)
            .unwrap();
        let revision = Patch::build()
            .axis("item", &[0, 2, 3])
            .axis("store", &[3, 1])
            .content_2d(&[[200., 100.], [-1., -1.], [400., 300.]])
            .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert!(modified[[0, 0]].is_nan());
        assert!(modified[[0, 1]].is_nan());
        assert_eq!(modified[[1, 0]], 300.);
        assert_eq!(modified[[1, 1]], 400.);
    }

    #[test]
    fn patch_2d_apply_smaller_patch_semi_overlap_different_order() {
        let mut base = Patch::build()
            .axis("item", &[1, 2, 3])
            .axis("store", &[1, 3])
            .content(None)
            .unwrap();
        let revision = Patch::build()
            .axis("item", &[0, 3])
            .axis("store", &[3, 1])
            .content_2d(&[[200., 100.], [400., 300.]])
            .unwrap();
        base.apply(&revision).unwrap();
        let modified = base.to_dense();
        assert!(modified[[0, 0]].is_nan());
        assert!(modified[[0, 1]].is_nan());
        assert!(modified[[1, 0]].is_nan());
        assert!(modified[[1, 1]].is_nan());
        assert_eq!(modified[[2, 0]], 300.);
        assert_eq!(modified[[2, 1]], 400.);
    }

    #[test]
    fn patch_2d_merge() {
        let pat1 = Patch::build()
            .axis_range("x", 0..2)
            .axis_range("y", 0..2)
            .content_2d(&[[std::f32::NAN, 2.], [3., std::f32::NAN]])
            .unwrap();
        let pat2 = Patch::build()
            .axis_range("x", 0..2)
            .axis_range("y", 0..2)
            .content_2d(&[[1., std::f32::NAN], [std::f32::NAN, 4.]])
            .unwrap();
        let m = Patch::merge(&[&pat1, &pat2]).unwrap().to_dense();
        assert_eq!(m[[0, 0]], 1.);
        assert_eq!(m[[0, 1]], 2.);
        assert_eq!(m[[1, 0]], 3.);
        assert_eq!(m[[1, 1]], 4.);
    }

    #[test]
    fn patch_serialize_round_trip() {
        let pat1 = Patch::build()
            .axis("item", &[0, 3])
            .axis("store", &[3, 1])
            .content_2d(&[[200., 100.], [400., 300.]])
            .unwrap();

        let mut buffer = vec![0u8; 0];
        pat1.serialize_into(None, &mut buffer).unwrap();
        // serialize() and serialize_into() should be the same
        assert_eq!(buffer, pat1.serialize(None).unwrap());
        let pat2 = Patch::deserialize_from(&buffer[..]).unwrap();
        assert_eq!(pat1, pat2);
    }
}
