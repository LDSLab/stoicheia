use failure::{err_msg, Fallible};
use std::convert::TryFrom;
use std::collections::HashSet;
use itertools::Itertools;

use crate::{Axis, AxisSegment, AxisSelection, Catalog, PatchRequest, Patch, PatchID, Label, BoundingBox};

/// Metadata about a quilt
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct QuiltMeta {
    pub(crate) name: String,
    pub(crate) axes: Vec<String>,
}
/// Read a QuiltMeta from SQLite
impl TryFrom<&rusqlite::Row<'_>> for QuiltMeta {
    type Error = rusqlite::Error;

    fn try_from(row: &rusqlite::Row) -> Result<Self, Self::Error> {
        Ok(QuiltMeta {
            name: row.get("quilt_name")?,
            axes: serde_json::from_str(&row.get::<_, String>("axes")?)
                // Fudging the error types here a little bit - but it's close
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
        })
    }
}

/// A mutable set of patches
///
/// When you read or write to a quilt, the IO runs on a subset of patches
/// and es the persistence layer to fetch and store them
pub struct Quilt<'t> {
    name: String,
    catalog: &'t dyn Catalog,
}
impl<'t> Quilt<'t> {
    /// Initialize a new quilt
    pub fn new(name: String, catalog: &'t dyn Catalog) -> Self {
        Self { name, catalog }
    }

    /// Apply a patch to a quilt
    ///
    /// The tensor name is part of the patch so it doesn't need to be specified
    pub fn apply(&mut self, pat: Patch<f32>) -> Fallible<()> {
        self.catalog.put_patch(PatchID(0), pat)?;
        Ok(())
    }

    /// Resolve the labels that a patch request selects
    fn get_axis_from_selection(&self, sel: AxisSelection) -> Fallible<(Axis, Vec<AxisSegment>)> {
        Ok(match sel {
            AxisSelection::All{name} => {
                let axis = self.catalog.get_axis(&name)?;
                let full_range = 0..=axis.labels.len();
                (axis, vec![full_range])
            },
            AxisSelection::Labels{name, labels} => {
                // TODO: Profile this - it could be a performance issue
                let axis: Axis = self.catalog.get_axis(&name)?;
                let label_hash : HashSet<Label> = labels.iter().copied().collect();
                let start_ix = axis.labels.iter().position(|x| label_hash.contains(x)).unwrap_or(axis.labels.len());
                let end_ix = axis.labels[start_ix..].iter().rposition(|x| label_hash.contains(x)).unwrap_or(0);
                (Axis::new(name.into(), labels), vec![start_ix..=end_ix])
            },
            AxisSelection::RangeInclusive { name, start, end } => {
                // Axis labels are not guaranteed to be sorted because it may be optimized for storage, not lookup
                let axis: Axis = self.catalog.get_axis(&name)?;
                let start_ix = axis
                    .labels
                    .iter()
                    .position(|&x| x == start)
                    // If we can't find that label we don't search anything
                    .unwrap_or(axis.labels.len());
                let end_ix = start_ix
                    + axis.labels[start_ix..]
                        .iter()
                        .position(|&x| x == end)
                        .unwrap_or(axis.labels.len() - start_ix);
                (Axis {
                    name: axis.name,
                    labels: Vec::from(&axis.labels[start_ix..=end_ix]),
                },
                vec![start_ix..=end_ix]
                )
            }
        })
    }

    /// Assemble a patch from a quilt
    ///
    /// The tensor name is part of the patch so it doesn't need to be specified
    pub fn assemble(&mut self, req: PatchRequest) -> Fallible<Patch<f32>> {
        ensure!(
            !req.axes.is_empty(),
            err_msg("Patches must have at least one axis")
        );

        //
        // Find all the labels of the axes they are planning to use
        //

        // Names and all labels of all of the axes involved
        let mut axes = vec![];
        // Segments of each axis, which will be the edges of bounding boxes
        let mut segments_by_axis = vec![];
        for sel in req.axes {
            let (axis, segments) = self.get_axis_from_selection(sel)?;
            axes.push(axis);
            segments_by_axis.push(segments);
        }
        // At this point we know how big the output will be
        ensure!(
            !axes.iter().map(|a| a.labels.len()).product::<usize>() <= 256 << 20,
            err_msg("Patches must be 256 million elements or less (1GB of 32bit floats)")
        );

        //
        // Find all bounding boxes we need to get the cartesian product of all the axis segments
        //

        // If there are more than 1000 bounding boxes, collapse them, to protect the R*tree (or whatever index) from DOS
        let total_bounding_boxes : usize = segments_by_axis.iter().map(|s| s.len()).product();
        let bounding_boxes = if total_bounding_boxes <= 1000 {
            vec![ BoundingBox(segments_by_axis.iter().map(|_| 0..=std::usize::MAX).collect()) ]
        } else {
            segments_by_axis
                .iter()
                .multi_cartesian_product()
                .map(|segments_group|{
                    BoundingBox(segments_group.into_iter().cloned().collect())
                })
                .collect()
        };

        //
        // Find the patches we need to fill all the bounding boxes
        //

        // TODO: This could be async
        let mut patch_ids = HashSet::new();
        for bbox in &bounding_boxes {
            for patch_id in self.catalog.get_patches_by_bounding_box(&self.name,bbox)? {
                patch_ids.insert(patch_id);
            }
        }

        //
        // Download and apply all the patches
        // 

        // TODO: This should definitely be async or at least concurrent
        let mut target_patch = Patch::from_axes(axes)?;
        for patch_id in patch_ids {
            let source_patch = self.catalog.get_patch(patch_id)?;
            target_patch.apply(&source_patch)?;
        }
        
        Ok(target_patch)
    }
}
