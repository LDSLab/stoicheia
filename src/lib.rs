#![feature(is_sorted, result_cloned)]
#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;
extern crate rusqlite as sql;

mod patch;
pub use patch::{Patch};

mod quilt;
pub use quilt::{Quilt, QuiltMeta};

mod catalog;
pub use catalog::{Catalog, MemoryCatalog, SQLiteCatalog};


/// The user-readable label for an axis. It may be huge, negative, and on-consecutive
#[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub struct Label(i64);
impl rusqlite::ToSql for Label {
    fn to_sql(&self) -> Result<sql::types::ToSqlOutput<'_>, sql::Error> {
        Ok(sql::types::ToSqlOutput::Owned(
            sql::types::Value::Integer(self.0)
        ))
    }
}
impl rusqlite::types::FromSql for Label {
    fn column_result(x: sql::types::ValueRef<'_>) -> Result<Self, sql::types::FromSqlError> {
        Ok(Label(i64::column_result(x)?))
    }
}

/// The database ID of a patch. Essentially meaningless, and may be random.
#[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub struct PatchID(i64);
impl rusqlite::ToSql for PatchID {
    fn to_sql(&self) -> Result<sql::types::ToSqlOutput<'_>, sql::Error> {
        Ok(sql::types::ToSqlOutput::Owned(
            sql::types::Value::Integer(self.0)
        ))
    }
}
impl rusqlite::types::FromSql for PatchID {
    fn column_result(x: sql::types::ValueRef<'_>) -> Result<Self, sql::types::FromSqlError> {
        Ok(PatchID(i64::column_result(x)?))
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct PatchRequest {
    axes: Vec<AxisSelection>,
}

/// Selection by axis labels, similar to .loc[] in Pandas
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub enum AxisSelection {
    All { name: String },
    RangeInclusive { name: String, start: Label, end: Label },
    Labels{ name: String, labels: Vec<Label> },
}

/// Selection by axis indicess, similar to .iloc[] in Pandas
type AxisSegment = std::ops::RangeInclusive<usize>;

/// Selection by bounding box, which is always by index
pub struct BoundingBox(Vec<AxisSegment>);


/// A dimension that can be used as labels for patches
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Debug)]
pub struct Axis {
    pub name: String,
    pub labels: Vec<Label>,
}
impl Axis {
    pub fn new(name: String, labels: Vec<Label>) -> Axis {
        Axis { name, labels }
    }
    pub fn len(&self) -> usize {
        self.labels.len()
    }
}
