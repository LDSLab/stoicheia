use crate::Fallible;
use std::convert::TryFrom;

use crate::{Catalog, Patch, PatchID};

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
        // TODO: balance the patches and handle overlaps correctly
        self.catalog.put_patch(PatchID(rand::random()), pat)?;
        Ok(())
    }

}
