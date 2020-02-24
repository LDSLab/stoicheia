use crate::sqlite::SQLiteConnection;
use crate::Fallible;
use itertools::Itertools;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    Axis, AxisSegment, AxisSelection, BoundingBox, Patch, PatchID, PatchRequest, Quilt,
    QuiltDetails, StoiError,
};

pub struct Catalog {
    storage: Arc<SQLiteConnection>,
}
impl Catalog {
    /// Connect to a Stoicheia catalog.
    ///
    /// If url is "", then an in-memory catalog will be created.
    /// If url is a file path, then a new SQLite-based catalog will be created.
    /// Other URLs may be added to support other storage schemes.
    pub fn connect(url: &str) -> Fallible<Self> {
        Ok(if url == "" {
            Catalog {
                storage: SQLiteConnection::connect_in_memory()?,
            }
        } else {
            Catalog {
                storage: SQLiteConnection::connect(url.into())?,
            }
        })
    }

    /// List the quilts available in this Catalog
    pub fn list_quilts(&self) -> Fallible<HashMap<String, QuiltDetails>> {
        self.storage.txn()?.list_quilts()
    }

    /// Create a quilt, and create axes as necessary to make it.
    ///
    /// With `ignore_if_exists=true`, you can make this idempotent rather than fail. In either case
    /// it will not replace or modify an existing quilt.
    pub fn create_quilt(
        &self,
        quilt_name: &str,
        axes_names: &[&str],
        ignore_if_exists: bool,
    ) -> Fallible<()> {
        let txn = self.storage.txn()?;
        for axis_name in axes_names {
            txn.create_axis(axis_name, true)?;
        }
        txn.create_quilt(quilt_name, axes_names, ignore_if_exists)?;
        txn.finish()?;
        Ok(())
    }

    /// Get a quilt by name, as a convenient way to access patches
    ///
    /// This doesn't load any of the content into memory, it just provides a convenient
    /// way to access patches. As a result, this is pretty cheap, but it does incur IO
    /// to get the quilt's metadata so it is still fallible.
    pub fn get_quilt(&self, quilt_name: &str, tag: &str) -> Fallible<Quilt> {
        // This will fail if the quilt doesn't already exist
        self.storage.txn()?.get_quilt_details(quilt_name)?;
        Ok(Quilt::new(quilt_name.into(), tag.into(), self))
    }

    /// Get details about a quilt by name
    ///
    /// What details are available may depend on the quilt, and fields are likely to
    /// be added with time (so be careful with serializing QuiltDetails)
    pub fn get_quilt_details(&self, quilt_name: &str) -> Fallible<QuiltDetails> {
        self.storage.txn()?.get_quilt_details(quilt_name)
    }

    /// Create an empty axis
    pub fn create_axis(&self, axis_name: &str, ignore_if_exists: bool) -> Fallible<()> {
        let txn = self.storage.txn()?;
        txn.create_axis(axis_name, ignore_if_exists)?;
        txn.finish()?;
        Ok(())
    }

    /// Get all the labels of an axis, in the order you would expect them to be stored
    pub fn get_axis(&self, name: &str) -> Fallible<Axis> {
        self.storage.txn()?.get_axis(name)
    }

    /// Replace the labels of an axis, in the order you would expect them to be stored.
    ///
    /// Returns true iff the axis was mutated in the process
    pub fn union_axis(&self, new_axis: &Axis) -> Fallible<bool> {
        let txn = self.storage.txn()?;
        let mutated = txn.union_axis(new_axis)?;
        txn.finish()?;
        Ok(mutated)
    }

    /// Fetch a patch from a quilt.
    ///
    /// - You can request any slice, and it will be assembled from the underlying commits.
    ///   - How many patches it's assembled from depends on the storage order
    ///     (which is the order the labels are specified in the axis, not in your request)
    /// - You can request elements you haven't initialized yet, and you'll get zeros.
    /// - You may only request patches up to 1 GB, as a safety valve
    pub fn fetch(&self, quilt_name: &str, tag: &str, mut request: PatchRequest) -> Fallible<Patch> {
        if request.is_empty() {
            return Err(StoiError::InvalidValue(
                "Patches must have at least one axis",
            ));
        }

        // All of this needs to be done in one transaction
        let txn = self.storage.txn()?;

        //
        // Find all the labels of the axes they are planning to use
        //

        // Names and all labels of all of the axes involved
        let mut axes = vec![];
        // Segments of each axis, which will be the edges of bounding boxes
        let mut segments_by_axis = vec![];

        // They can't possibly use more than ten axes - just a safety measure.
        request.truncate(10);

        // Small note: we limit the axes here just as a safety measure
        for sel in request {
            let (axis, segments) = txn.get_axis_from_selection(sel)?;
            axes.push(axis);
            segments_by_axis.push(segments);
        }

        // Tack on any axes they forgot
        for axis_name in txn.get_quilt_details(quilt_name)?.axes {
            if axes.iter().find(|a| a.name == axis_name).is_none() {
                let (axis, segments) =
                    txn.get_axis_from_selection(AxisSelection::All { name: axis_name })?;
                axes.push(axis);
                segments_by_axis.push(segments);
            }
        }

        // At this point we know how big the output will be.
        // The error here is early to avoid the IO
        // and we don't construct the patch (which would have noticed and raised the same error)
        // in order to avoid holding memory longer
        if axes.iter().map(|a| a.len()).product::<usize>() > 256 << 20 {
            return Err(StoiError::TooLarge(
                "Patches must be 256 million elements or less (1GB of 32bit floats)",
            ));
        }

        //
        // Find all bounding boxes we need to get the cartesian product of all the axis segments
        //

        // If there are more than 1000 bounding boxes, collapse them, to protect the R*tree (or whatever index) from DOS
        let total_bounding_boxes: usize = segments_by_axis.iter().map(|s| s.len()).product();
        let bounding_boxes = if total_bounding_boxes <= 1000 {
            vec![segments_by_axis
                .iter()
                .map(|_| (0, std::usize::MAX))
                .collect()]
        } else {
            segments_by_axis
                .iter()
                .multi_cartesian_product()
                .map(|segments_group| segments_group.into_iter().cloned().collect())
                .collect()
        };

        //
        // Find the patches we need to fill all the bounding boxes
        //

        let patch_ids = txn.get_patches_by_bounding_boxes(&quilt_name, &tag, &bounding_boxes)?;

        //
        // Download and apply all the patches
        //

        // TODO: This should definitely be async or at least concurrent
        let mut target_patch = Patch::new(axes, None)?;
        for patch_id in patch_ids {
            let source_patch = txn.get_patch(patch_id)?;
            target_patch.apply(&source_patch)?;
        }

        Ok(target_patch)
    }

    /// Commit a patch to a quilt.
    ///
    /// Commits are a pretty expensive operation - the system is designed for more reads than writes.
    /// In specific, it will do at least all the following:
    ///
    /// - Get quilt details, including full copies of all the axes
    /// - Check, compact, and compress all the patches, splitting and balancing search indices
    /// - Extend all the axes (if necessary) to include the area the patches cover
    /// - Upload all the patches and their data
    /// - Log the commit and change the tags to point to it
    ///
    pub fn commit(
        &self,
        quilt_name: &str,
        parent_tag: Option<&str>,
        new_tag: Option<&str>,
        message: &str,
        patches: Vec<Patch>,
    ) -> Fallible<()> {
        // TODO: Implement split/balance...
        // TODO: Think about axis versioning - maybe not a good idea anyway?
        let txn = self.storage.txn()?;

        // Check that the axes are consistent
        let quilt_details = txn.get_quilt_details(quilt_name)?;
        for patch in &patches {
            if patch
                .axes()
                .iter()
                .map(|a| &a.name)
                .sorted()
                .ne(quilt_details.axes.iter().sorted())
            {
                return Err(StoiError::MisalignedAxes(format!(
                    "the quilt \"{}\" has axes [{}] but the patch has [{}], which doesn't match. Please note broadcasting is not (yet) supported. The patch axes should match exactly.",
                    quilt_name, quilt_details.axes.iter().join(", "), patch.axes().iter().map(|a|&a.name).join(", ")
                )));
            }
        }

        // Extend all axes as necessary to complete the patching
        for axis_name in &quilt_details.axes {
            let mut axis = txn.get_axis(axis_name)?;
            let mut mutated = false;
            for patch in &patches {
                // Linear search over max 4 elements so don't sweat it
                mutated |= axis.union(&patch.axes().iter().find(|a| &a.name == axis_name).unwrap());
            }
            if mutated {
                // This is actually quite expensive so it's worth avoiding it where possible
                txn.union_axis(&axis)?;
            }
        }

        txn.put_commit(
            quilt_name,
            parent_tag.unwrap_or("latest"),
            new_tag.unwrap_or("latest"),
            message,
            patches,
        )?;

        txn.finish()?;
        Ok(())
    }

    /// Untag a commit, to "delete" it
    ///
    /// Untagging a commit doesn't remove its effects, it only makes it inaccessible
    /// and allows (now or any time in the future) for the library to:
    ///
    /// - Merge it into its successots, if it has any
    /// - Garbage collect it otherwise
    pub fn untag(&self, _quilt_name: &str, _tag: &str) -> Fallible<()> {
        // TODO: Implement untag
        Ok(())
    }
}

pub(crate) trait StorageConnection: Send + Sync {
    type Transaction: StorageTransaction;
    fn txn(self) -> Fallible<Self::Transaction>;
}

/// A connection to tensor storage
pub(crate) trait StorageTransaction {
    /// Get only the metadata associated with a quilt by name
    fn get_quilt_details(&self, quilt_name: &str) -> Fallible<QuiltDetails>;

    /// Create a new quilt (doesn't create associated axes)
    fn create_quilt(
        &self,
        quilt_name: &str,
        axes_names: &[&str],
        ignore_if_exists: bool,
    ) -> Fallible<()>;

    /// List all the quilts in the catalog
    fn list_quilts(&self) -> Fallible<HashMap<String, QuiltDetails>>;

    /// List all the patches that intersect a bounding box
    ///
    /// There may be false positives; some patches may not actually overlap
    /// There are no false negatives; all patches that overlap will be returned
    ///
    /// This method exists in case the database supports efficient multidimensional range queries
    /// such as SQLite or Postgres/PostGIS
    fn get_patches_by_bounding_boxes(
        &self,
        quilt_name: &str,
        tag: &str,
        bounds: &[BoundingBox],
    ) -> Fallible<Box<dyn Iterator<Item = PatchID>>>;

    /// Get a single patch by ID
    fn get_patch(&self, id: PatchID) -> Fallible<Patch>;

    /// Create an axis, possibly ignoring it if it exists
    fn create_axis(&self, name: &str, ignore_if_exists: bool) -> Fallible<()>;

    /// Get all the labels of an axis, in the order you would expect them to be stored
    fn get_axis(&self, name: &str) -> Fallible<Axis>;

    /// Overwrite a whole axis. Only do this through `Catalog.union_axis()`.
    fn put_axis(&self, axis: &Axis) -> Fallible<()>;

    /// Make changes to a tensor via a commit
    ///
    /// This is only available together, so that the underlying storage media can do this
    /// atomically without a complicated API
    fn put_commit(
        &self,
        quilt_name: &str,
        parent_tag: &str,
        new_tag: &str,
        message: &str,
        patches: Vec<Patch>,
    ) -> Fallible<()>;

    /// Finish and commit the transaction successfully
    fn finish(self) -> Fallible<()>;

    /// Use the actual axis values to resolve a request into specific labels
    ///
    /// This is necessary because we need to turn the axis labels into storage indices for range queries
    fn get_axis_from_selection(&self, sel: AxisSelection) -> Fallible<(Axis, Vec<AxisSegment>)> {
        Ok(match sel {
            AxisSelection::All { name } => {
                let axis = self.get_axis(&name)?;
                let full_range = (0, axis.len());
                (axis, vec![full_range])
            }
            AxisSelection::Labels { name, labels } => {
                // TODO: Profile this - it could be a performance issue
                let axis: Axis = self.get_axis(&name)?;
                let labelset = axis.labelset();
                let start_ix = axis
                    .labels()
                    .iter()
                    .position(|x| labelset.contains(&x))
                    .unwrap_or(axis.len());
                let end_ix = axis.labels()[start_ix..]
                    .iter()
                    .rposition(|x| labelset.contains(&x))
                    .unwrap_or(0);
                (Axis::new(name, labels)?, vec![(start_ix, end_ix)])
            }
            AxisSelection::RangeInclusive { name, start, end } => {
                // Axis labels are not guaranteed to be sorted because it may be optimized for storage, not lookup
                let axis: Axis = self.get_axis(&name)?;
                let lab = axis.labels();
                let start_ix = lab
                    .iter()
                    .position(|&x| x == start)
                    // If we can't find that label we don't search anything
                    .unwrap_or(axis.len());
                let end_ix = start_ix
                    + lab[start_ix..]
                        .iter()
                        .position(|&x| x == end)
                        .unwrap_or(axis.len() - start_ix);
                (
                    Axis::new(&axis.name, Vec::from(&lab[start_ix..=end_ix]))?,
                    vec![(start_ix, end_ix)],
                )
            }
        })
    }

    /// Replace the labels of an axis, in the order you would expect them to be stored.
    ///
    /// Returns true iff the axis was mutated in the process
    fn union_axis(&self, new_axis: &Axis) -> Fallible<bool> {
        let mut existing_axis = self
            .get_axis(&new_axis.name)
            .unwrap_or(Axis::empty(&new_axis.name));
        let mutated = existing_axis.union(new_axis);
        if mutated {
            self.put_axis(&existing_axis)?;
        }
        Ok(mutated)
    }
}

pub struct CommitDetails {
    comm_id: i64,
    quilt_name: String,
    tag: String,
    message: String,
    patches: Vec<PatchID>,
}

#[cfg(test)]
mod tests {
    use crate::{Axis, AxisSelection, Catalog, Label, Patch};

    #[test]
    fn test_create_axis() {
        let cat = Catalog::connect("").unwrap();
        cat.create_axis("xjhdsa", false)
            .expect("Should be fine to create one that doesn't exist yet");
        cat.create_axis("xjhdsa", true)
            .expect("Should be fine to try to create an axis that exists");
        cat.create_axis("xjhdsa", false)
            .expect_err("Should fail to create duplicate axis");

        cat.get_axis("uyiuyoiuy")
            .expect_err("Should throw an error for an axis that doesn't exist.");
        let mut ax = cat
            .get_axis("xjhdsa")
            .expect("Should be able to get an axis I just made");
        assert!(ax.labels() == &[] as &[Label]);

        ax = Axis::new("uyiuyoiuy", vec![1, 5]).expect("Should be able to create an axis");

        // Union an axis
        cat.union_axis(&ax)
            .expect("Should be able to union an axis");
        ax = cat
            .get_axis("uyiuyoiuy")
            .expect("Axis should exist after union");
        assert_eq!(ax.labels(), &[1, 5]);

        cat.union_axis(&ax).expect("Union twice is a no-op");
        ax = cat
            .get_axis("uyiuyoiuy")
            .expect("Axis should still exist after second union");
        assert_eq!(ax.labels(), &[1, 5]);

        cat.union_axis(&Axis::new("uyiuyoiuy", vec![0, 5]).unwrap())
            .expect("Union should append");
        ax = cat.get_axis("uyiuyoiuy").unwrap();
        assert_eq!(ax.labels(), &[1, 5, 0]);
    }

    #[test]
    fn test_create_quilt() {
        let cat = Catalog::connect("").unwrap();
        // This should automatically create the axes as well, so it doesn't complain
        cat.create_quilt("sales", &["itm", "lct", "day"], true)
            .unwrap();
    }

    #[test]
    fn test_basic_fetch() {
        let cat = Catalog::connect("").unwrap();
        cat.create_quilt("sales", &["itm", "lct", "day"], true)
            .unwrap();

        // This should assume the axes' labels exist if you specify them, but not if you don't
        let mut pat = cat
            .fetch(
                "sales",
                "latest",
                vec![
                    AxisSelection::All { name: "itm".into() },
                    AxisSelection::Labels {
                        name: "lct".into(),
                        labels: vec![1],
                    },
                ],
            )
            .unwrap();
        // We asked for two dimensions but any dimensions you missed will be tacked on
        assert_eq!(pat.ndim(), 3);
        assert_eq!(pat.content().shape(), &[0, 1, 0]);

        pat = Patch::build()
            .axis_range("itm", 9..12)
            .axis_range("xyz", 2..4)
            .content(None)
            .unwrap();

        pat.content_mut().fill(1.0);
    }
}
