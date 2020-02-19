use crate::Fallible;
use std::convert::TryFrom;

use crate::{Catalog, Patch, PatchID};

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

/// A convenience class for accessing many patches from the same tensor
///
/// Creating this class is quite cheap as it merely references functions also available in Catalog
pub struct Quilt<'t> {
    name: String,
    tag: String,
    catalog: &'t Catalog,
}
impl<'t> Quilt<'t> {
    /// Create a new quilt connect to a Catalog
    pub(crate) fn new(name: String, tag: String, catalog: &'t Catalog) -> Self {
        Self { name, tag, catalog }
    }

    pub fn details(&self) -> Fallible<QuiltDetails> {
        self.catalog.get_quilt_details(&self.name)
    }

    pub fn commit(
        &self,
        new_tag: Option<&str>,
        message: &str,
        patches: Vec<Patch<f32>>,
    ) -> Fallible<i64> {
        self.catalog.commit(
            &self.name,
            &self.tag,
            new_tag.unwrap_or(&self.tag),
            message,
            patches,
        )
    }
}
