use crate::sqlite::{SQLiteConnection, SQLiteTransaction};
use crate::Fallible;
use crate::Label;
use itertools::Itertools;
use std::collections::{HashMap, HashSet, BTreeMap};
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use crate::{Axis, AxisSegment, AxisSelection, BoundingBox, Patch, PatchID, PatchRef, StoiError};

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

    /// Start a new transaction on the quilt
    pub fn begin(&mut self) -> Fallible<SQLiteTransaction> {
        self.storage.txn()
    }
}

pub trait StorageConnection: Send + Sync {
    type Transaction: StorageTransaction;
    fn txn(self) -> Fallible<Self::Transaction>;
}

/// A connection to tensor storage
pub trait StorageTransaction {
    /// Increment a counter by name, used for performance statistics
    fn trace(&mut self, name: &'static str, increment: usize);
    /// Get only the metadata associated with a quilt by name
    fn get_quilt_details(&mut self, quilt_name: &str) -> Fallible<QuiltDetails>;

    /// Create a new quilt (doesn't create associated axes)
    fn create_quilt(
        &mut self,
        quilt_name: &str,
        axes_names: &[&str],
        ignore_if_exists: bool,
    ) -> Fallible<()>;

    /// List all the quilts in the catalog
    fn list_quilts(&mut self) -> Fallible<HashMap<String, QuiltDetails>>;

    /// List all the patches that intersect a bounding box
    ///
    /// There may be false positives; some patches may not actually overlap
    /// There are no false negatives; all patches that overlap will be returned
    ///
    /// This method exists in case the database supports efficient multidimensional range queries
    /// such as SQLite or Postgres/PostGIS
    fn get_patches_by_bounding_boxes(
        &mut self,
        quilt_name: &str,
        tag: &str,
        deep: bool,
        bounds: &[BoundingBox],
    ) -> Fallible<Vec<PatchRef>>;

    /// Get a single patch by ID
    fn get_patch(&mut self, id: PatchID) -> Fallible<Patch>;

    /// Get all the labels of an axis, in the order you would expect them to be stored.
    /// 
    /// Returns an empty axis if this axis is missing.
    fn get_axis(&mut self, name: &str) -> Fallible<&Axis>;

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
    fn create_commit(
        &mut self,
        quilt_name: &str,
        parent_tag: &str,
        new_tag: &str,
        message: &str,
        patches: &[&Patch],
    ) -> Fallible<()> {
        self.trace("create_commit", 1);
        // Check that the axes are consistent
        let quilt_details = self.get_quilt_details(quilt_name)?;
        for patch in patches {
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
            let mut axis = self.get_axis(axis_name)?.clone();
            let mut mutated = false;
            for patch in patches {
                // Linear search over max 4 elements so don't sweat it
                mutated |= axis.union(&patch.axes().iter().find(|a| &a.name == axis_name).unwrap());
            }
            if mutated {
                // This is actually quite expensive so it's worth avoiding it where possible
                self.union_axis(&axis)?;
            }
        }

        // Split the patches into reasonable sizes
        let mut split_patches = vec![];
        for &patch in patches {
            // TODO: Extra clone here?
            split_patches.extend(self.maybe_split(patch.to_owned())?);
        }

        self.put_commit(
            quilt_name,
            parent_tag,
            new_tag,
            message,
            &split_patches.iter().collect_vec(),
        )?;
        Ok(())
    }

    /// Make changes to a tensor via a commit
    ///
    /// This is only available together, so that the underlying storage media can do this
    /// atomically without a complicated API
    fn put_commit(
        &mut self,
        quilt_name: &str,
        parent_tag: &str,
        new_tag: &str,
        message: &str,
        patches: &[&Patch],
    ) -> Fallible<()>;

    /// Rollback the transaction
    fn rollback(self) -> Fallible<()>;

    /// Finish and commit the transaction successfully
    fn finish(self) -> Fallible<()>;

    /// Use the actual axis values to resolve a request into specific labels
    ///
    /// This is necessary because we need to turn the axis labels into storage indices for range queries
    fn get_axis_from_selection(
        &mut self,
        name: &str,
        sel: AxisSelection,
    ) -> Fallible<(Axis, Vec<AxisSegment>)> {
        self.trace("get_axis_from_selection", 1);
        Ok(match sel {
            AxisSelection::All => {
                let axis = self.get_axis(&name)?;
                let full_range = (0, axis.len());
                (axis.clone(), vec![full_range])
            }
            AxisSelection::Labels(labels) => {
                // TODO: Profile this - it could be a performance issue
                let axis = self.get_axis(&name)?;
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
            AxisSelection::LabelSlice(start, end) => {
                // Axis labels are not guaranteed to be sorted because it may be optimized for storage, not lookup
                let axis = self.get_axis(&name)?;
                let lab = axis.labels();
                let start_ix = lab
                    .iter()
                    .position(|&x| x == start)
                    // If we can't find that label we don't search anything
                    .unwrap_or(axis.len());
                let end_ix = lab[start_ix..].iter().position(|&x| x == end).unwrap_or(0);
                let end_ix = (1 + start_ix + end_ix).min(axis.len());
                (
                    Axis::new(&axis.name, Vec::from(&lab[start_ix..end_ix]))?,
                    vec![(start_ix, end_ix)],
                )
            }
            AxisSelection::StorageSlice(start_ix, end_ix) => {
                let axis = self.get_axis(&name)?;
                let lab = axis.labels();
                (
                    Axis::new(&axis.name, Vec::from(&lab[start_ix..end_ix]))?,
                    vec![(start_ix, end_ix)],
                )
            }
        })
    }

    /// Replace the labels of an axis, in the order you would expect them to be stored.
    ///
    /// Returns true iff the axis was mutated in the process
    fn union_axis(&mut self, new_axis: &Axis) -> Fallible<bool>;

    /// Fetch a patch from a quilt.
    ///
    /// - You can request any slice, and it will be assembled from the underlying commits.
    ///   - How many patches it's assembled from depends on the storage order
    ///     (which is the order the labels are specified in the axis, not in your request)
    /// - You can request elements you haven't initialized yet, and you'll get NANs.
    /// - You can only request patches up to 1 GB, as a safety valve
    fn fetch(
        &mut self,
        quilt_name: &str,
        tag: &str,
        mut request: Vec<AxisSelection>,
    ) -> Fallible<Patch> {
        self.trace("fetch", 1);

        //
        // Find all the labels of the axes they are planning to use
        //
        let quilt_details = self.get_quilt_details(quilt_name)?;

        // Names and all labels of all of the axes involved
        let mut axes = vec![];
        // Segments of each axis, which will be the edges of bounding boxes
        let mut segments_by_axis = vec![];

        // They can't possibly use more than ten axes - just a safety measure.
        request.reverse(); // So we can iterate it and take ownership
        for axis_name in &quilt_details.axes {
            let (axis, segments) = match request.pop() {
                Some(sel) => self.get_axis_from_selection(axis_name, sel)?,
                None => self.get_axis_from_selection(axis_name, AxisSelection::All)?,
            };
            axes.push(axis);
            segments_by_axis.push(segments);
        }
        assert!(axes.len() >= 1, StoiError::MisalignedAxes("No axes for quilt in fetch()".into()));

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
        let bounding_boxes = if total_bounding_boxes > 1000 {
            vec![[
                (0usize, 1usize << 60),
                (0usize, 1usize << 60),
                (0usize, 1usize << 60),
                (0usize, 1usize << 60),
            ]]
        } else {
            segments_by_axis
                .iter()
                .multi_cartesian_product()
                .map(|segments_group| {
                    [
                        **segments_group.get(0).unwrap_or(&&(0usize, 1usize << 60)),
                        **segments_group.get(1).unwrap_or(&&(0usize, 1usize << 60)),
                        **segments_group.get(2).unwrap_or(&&(0usize, 1usize << 60)),
                        **segments_group.get(3).unwrap_or(&&(0usize, 1usize << 60)),
                    ]
                })
                .collect::<Vec<BoundingBox>>()
        };

        //
        // Find the patches we need to fill all the bounding boxes
        //

        let patch_refs =
            self.get_patches_by_bounding_boxes(&quilt_name, &tag, true, &bounding_boxes)?;

        //
        // Download and apply all the patches
        //

        // TODO: This should definitely be async or at least concurrent
        let mut target_patch = Patch::new(axes, None)?;
        for patch_ref in patch_refs {
            let source_patch = self.get_patch(patch_ref.id)?;
            target_patch.apply(&source_patch)?;
        }

        Ok(target_patch)
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
    fn maybe_split(&mut self, original: Patch) -> Fallible<Vec<Patch>> {
        self.trace("maybe_split", 1);
        match original.content().len() {
            0 => Ok(vec![]),                   // Take out the trash
            1..=1048576 => Ok(vec![original]), // Cap at 4 MB
            _ => {
                // Split everything else
                self.trace("maybe_split split", 1);

                // Split it along it's longest axis
                let (long_ax_ix, long_axis) = original
                    .axes()
                    .iter()
                    .enumerate()
                    .max_by_key(|(_ax_ix, ax)| ax.labels().len())
                    .unwrap(); // <- Patch::new() checks for at least one axis
                               // Replace the patch axis for the global axis by that name

                let global_long_axis = self.get_axis(&long_axis.name)?;
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

                let mut patches = vec![];
                for indices in &[left_patch_indices, right_patch_indices] {
                    let mut axes = original.axes().to_vec();
                    // Replace the long axis
                    axes[long_ax_ix] = Axis::new_unchecked(
                        &long_axis.name,
                        indices
                            .iter()
                            .map(|ix| long_axis.labels()[*ix])
                            .collect_vec(),
                    );
                    // Slice the patch
                    let sliced_patch = Patch::new(
                        axes.to_vec(),
                        Some(original.content().select(nd::Axis(long_ax_ix), indices)),
                    )
                    .unwrap()
                    .compact()
                    .into_owned();
                    patches.extend(self.maybe_split(sliced_patch)?)
                }
                Ok(patches)
            }
        }
    }

    /// Get the bounding box of a patch
    ///
    /// These bounding boxes depend on the storage order of the catalog, so they aren't something
    /// the Patch could know on its own, instead you find this through the catalog
    fn get_bounding_box(&mut self, patch: &Patch) -> Fallible<BoundingBox> {
        self.trace("get_bounding_box", 1);
        let bbvec = (0..4)
            .map(|ax_ix| match patch.axes().get(ax_ix) {
                Some(patch_axis) => {
                    let patch_axis_labelset: HashSet<Label> =
                        patch_axis.labels().iter().copied().collect();
                    let global_axis = self.get_axis(&patch_axis.name)?;
                    let first = global_axis
                        .labels()
                        .iter()
                        .position(|x| patch_axis_labelset.contains(x));
                    let last = global_axis
                        .labels()
                        .iter()
                        .rposition(|x| patch_axis_labelset.contains(x));

                    Ok((first.unwrap_or(0), last.unwrap_or(1 << 60)))
                }
                None => Ok((0, 1 << 60)),
            })
            .collect::<Fallible<Vec<AxisSegment>>>()?;
        Ok(bbvec[..].try_into()?)
    }


    /// Untag a commit, to "delete" it
    ///
    /// Untagging a commit doesn't remove its effects, it only makes it inaccessible
    /// and allows (now or any time in the future) for the library to:
    ///
    /// - Merge it into its successors, if it has any
    /// - Garbage collect it otherwise
    fn untag(&self, _quilt_name: &str, _tag: &str) -> Fallible<()> {
        // TODO: Implement untag
        Ok(())
    }

    /// Retrieve performance counters, useful for debugging performance problems
    /// 
    /// Returns: a Map containing the counters by name
    fn get_performance_counters(&self) -> BTreeMap<&'static str, usize>;
}

/// Metadata about a quilt
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct QuiltDetails {
    pub(crate) name: String,
    pub(crate) axes: Vec<String>,
}
/// Read a QuiltDetails from SQLite
impl TryFrom<&rusqlite::Row<'_>> for QuiltDetails {
    type Error = rusqlite::Error;

    fn try_from(row: &rusqlite::Row) -> Result<Self, Self::Error> {
        Ok(QuiltDetails {
            name: row.get("quilt_name")?,
            axes: serde_json::from_str(&row.get::<_, String>("axes")?)
                // Fudging the error types here a little bit - but it's close
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Axis, AxisSelection, Catalog, ContentPattern, Patch, StorageTransaction};
    use itertools::Itertools;

    #[test]
    fn test_create_quilt() {
        let mut cat = Catalog::connect("").unwrap();
        let mut txn = cat.begin().unwrap();
        // This should automatically create the axes as well, so it doesn't complain
        txn.create_quilt("sales", &["itm", "lct", "day"], true)
            .unwrap();
    }
    /// Fetching from an empty quilt should create an empty patch
    #[test]
    fn test_fetch_empty_quilt() {
        let mut cat = Catalog::connect("").unwrap();
        let mut txn = cat.begin().unwrap();
        txn.create_quilt("sales", &["itm", "lct", "day"], true)
            .unwrap();

        // This should assume the axes' labels exist if you specify them, but not if you don't
        let pat = txn
            .fetch(
                "sales",
                "latest",
                vec![AxisSelection::All, AxisSelection::Labels(vec![1])],
            )
            .unwrap();
        // We asked for two dimensions but any dimensions you missed will be tacked on
        assert_eq!(pat.ndim(), 3);
        assert_eq!(pat.content().shape(), &[0, 1, 0]);
    }
    /// Commit one patch to the quilt and check that it survives a round trip
    #[test]
    fn test_commit_first_patches() {
        let mut cat = Catalog::connect("").unwrap();
        let mut txn = cat.begin().unwrap();
        txn.create_quilt("sales", &["dim0", "dim1"], true)
            .unwrap();

        //
        // 1: Does commit + fetch work?
        //
        
        // Check that the quilt details match what we just posted
        assert_eq!(
            txn.get_quilt_details("sales").unwrap().axes,
            vec!["dim0".to_string(), "dim1".to_string()]
        );
        
        // Commit a patch
        let mut reference_patch = Patch::autogenerate(ContentPattern::Random, 5);
        txn.create_commit("sales", "latest", "latest", "message", &[&reference_patch])
            .unwrap();
        
        // Check that the first patch was saved correctly
        let output_patch = txn
            .fetch("sales", "latest", vec![])
            .unwrap();
        assert_eq!(reference_patch.content(), output_patch.content());

        //
        // 2. Does overwriting work?
        //

        // Commit another patch over the same spot
        // If we just used another random patch it would be somewhere else
        // So instead we set the content of our first patch.
        let temp_reference_patch = Patch::autogenerate(ContentPattern::Random, 5);
        // This direct assignment ignores label alignment.
        reference_patch.content_mut().assign(&temp_reference_patch.content());
        txn.create_commit("sales", "latest", "latest", "message", &[&reference_patch])
            .unwrap();

        // We should see only the new patch because it occludes the first one totally
        let output_patch = txn
            .fetch("sales", "latest", vec![])
            .unwrap();
        assert_eq!(reference_patch.content(), output_patch.content());


        //
        // 3. Do transactions work?
        //

        // Commit the transaction and nothing should have changed
        txn.finish().unwrap();
        let mut txn = cat.begin().unwrap();
        let output_patch = txn
            .fetch("sales", "latest", vec![])
            .unwrap();
        assert_eq!(reference_patch.content(), output_patch.content());

        // Duplicate of the first round of overwriting
        // Commit another patch that occludes it totally.
        // This time we're testing that multiple transactions work too.
        let temp_reference_patch = Patch::autogenerate(ContentPattern::Random, 5);
        reference_patch.content_mut().assign(&temp_reference_patch.content());
        txn.create_commit("sales", "latest", "latest", "message", &[&reference_patch])
            .unwrap();
        let output_patch = txn
            .fetch("sales", "latest", vec![])
            .unwrap();
        assert_eq!(reference_patch.content(), output_patch.content());
        txn.finish().unwrap();

        // Another duplicate of overwriting, this time we rollback
        let mut txn = cat.begin().unwrap();
        let temp_reference_patch = Patch::autogenerate(ContentPattern::Random, 5);
        let mut rollback_reference_patch = reference_patch.clone();
        rollback_reference_patch.content_mut().assign(&temp_reference_patch.content());
        txn.create_commit("sales", "latest", "latest", "message", &[&rollback_reference_patch])
            .unwrap();
        txn.rollback().unwrap(); // Oopsie, undo!
        let mut txn = cat.begin().unwrap();
        let output_patch = txn
            .fetch("sales", "latest", vec![])
            .unwrap();
        // Should be back to before
        assert_eq!(reference_patch.content(), output_patch.content());

        //
        // 4. Do transactions rollback
        //



    }

    /// Test that fetches incur the right number of reads (low read amplification)
    #[test]
    #[ignore]
    fn test_read_amplification() {
        let mut cat = populate_quilt();
        let mut txn = cat.begin().unwrap();
        txn.fetch("quilt", "latest",
            vec![
                // Only label slice is inclusive
                AxisSelection::LabelSlice(0, 100),
                AxisSelection::StorageSlice(0, 100),
            ],
        ).unwrap();
        let ctr = txn.get_performance_counters();
        // It could be done in 1, so cap it at 4.
        println!("{:?}", ctr);
        assert!(ctr["get_patch"] <= 4);
    }


    /// Check that the state of the quilt is consistent as we keep adding patches and commits
    #[test]
    #[ignore]
    fn test_populate_quilt() {
        populate_quilt();
    }
    fn populate_quilt() -> Catalog {
        use rand::prelude::*;

        // Needs to be large enough to require multiple patches,
        // and patches can reasonably contain 1000x1000 elements
        let (w, h) = (10000, 10000);
        let master = Patch::autogenerate(ContentPattern::Sparse, 10000);
        let master_content = master.content();

        let mut cat = Catalog::connect("").unwrap();
        let mut txn = cat.begin().unwrap();
        txn.create_quilt("quilt", &["x", "y"], true).unwrap();
        txn.union_axis(&Axis::range("x", 0..w as i64)).unwrap();
        txn.union_axis(&Axis::range("y", 0..h as i64)).unwrap();
        //txn.finish().unwrap();

        //let mut txn = cat.begin().unwrap();
        // eight transactions
        for _ in 0..8 {
            // Of eight commits
            for _ in 0..8 {
                // Get slices out of the generated matrix and put them into the empty matrix,
                // then check them against the catalog.
                // To make this more fun, the first axis uses inclusive label based indexing,
                // but the second axis uses open-closed storage-index based indexing.
                // As long as we provide the axes first, this should be fine.

                // All four of px, pxe, py, pye use [x,y) inclusiveness
                let mut t = thread_rng();
                let px: usize = t.gen_range(0, w-100);
                let py: usize = t.gen_range(0, h-100);
                // This +1 is because inclusive label slicing makes empty slices impossible
                let pxe = px + t.gen_range(1, (w-px).min(1000));
                let pye = py + t.gen_range(1, (h-py).min(1000));

                // with 8 patches
                let mut patches = vec![];
                for _ in 0..8 {
                    let content = master_content.slice(s![px..pxe, py..pye]);
                    // These two ranges are [x,y)
                    let patch = Patch::build()
                        .axis_range("x", px as i64..pxe as i64)
                        .axis_range("y", py as i64..pye as i64)
                        .content(content.to_owned().into_dyn())
                        .unwrap();
                    patches.push(patch);
                }

                txn.create_commit("quilt", "latest", "latest", "hi", &patches.iter().collect_vec()[..])
                    .unwrap();

                // let fetched_patch = txn
                //     .fetch(
                //         "quilt",
                //         "latest",
                //         vec![
                //             // Only label slice is inclusive
                //             AxisSelection::LabelSlice(px as i64, -1 + pxe as i64),
                //             AxisSelection::StorageSlice(py, pye),
                //         ],
                //     )
                //     .unwrap();
                // let mean_nonnan_err = (&patch.content() - &fetched_patch.content())
                //     .fold_skipnan((0., 0.), |(c, a), x| (c+1., a+x.raw().abs()));
                // assert!((mean_nonnan_err.1 / mean_nonnan_err.0.max(1.)) < 0.001);
            }
        }
        txn.finish().unwrap();
        cat
    }
}
