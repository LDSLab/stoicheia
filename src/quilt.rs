use failure::Fallible;
use std::convert::TryFrom;

use crate::{Axis, Catalog, Label, Patch};

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct PatchRequest {
    axes: Vec<(String, AxisSelection)>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum AxisSelection {
    All,
    Range { start: Label, end: Label },
    Labels(Vec<Label>),
}

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
        self.catalog.put_patch(0, pat)?;
        Ok(())
    }

    /// Resolve the labels that a patch request selects
    fn get_axis_from_selection(&self, axis_name: &str, req: AxisSelection) -> Fallible<Axis> {
        let axis: Axis = self.catalog.get_axis(axis_name)?;
        Ok(match req {
            AxisSelection::All => axis,
            AxisSelection::Labels(labels) => Axis::new(axis_name.into(), labels),
            AxisSelection::Range { start, end } => {
                // Axis labels are not guaranteed to be sorted because it may be optimized for storage, not lookup
                let start_ix = axis
                    .labels
                    .iter()
                    .position(|&x| x == start)
                    .unwrap_or(axis.labels.len());
                let end_ix = start_ix
                    + axis.labels[start_ix..]
                        .iter()
                        .position(|&x| x == end)
                        .unwrap_or(axis.labels.len() - start_ix);
                Axis {
                    name: axis.name,
                    labels: Vec::from(&axis.labels[start_ix..end_ix]),
                }
            }
        })
    }

    /// Assemble a patch from a quilt
    ///
    /// The tensor name is part of the patch so it doesn't need to be specified
    pub fn assemble(&mut self, req: PatchRequest) -> Fallible<Patch<f32>> {
        for patch_id in self.catalog.get_patches_by_bounding_box(
            &self.name,
            &[], // no box - means everywhere
        ) {}
        let pat = self.catalog.get_patch(0)?;
        ensure!(pat.is_some(), "Failed to retrieve patch!");
        Ok(pat.unwrap())
    }
}
