#![feature(is_sorted, result_cloned)]
//! Sharded tensor storage and retrieval
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate ndarray as nd;
extern crate rusqlite as sql;

mod patch;
pub use patch::Patch;

mod quilt;
pub use quilt::{Quilt, QuiltDetails};

mod catalog;
pub use catalog::Catalog;

mod sqlite;

mod axis;
pub use axis::Axis;

mod error;
pub use error::{Fallible, StoiError};

#[cfg(feature = "python")]
pub mod python;

/// A user-defined signed integer label for a particular component of an axis
///
/// Labels of an axis may not be consecutive, and they define both the storage and retrieval order.
/// This is important because we trust the user knows what items will be used together, and without
/// this, patches may cluster meaningless groups of points.
pub type Label = i64;

/// The database ID of a patch.
#[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub struct PatchID(i64);
impl rusqlite::ToSql for PatchID {
    fn to_sql(&self) -> Result<sql::types::ToSqlOutput<'_>, sql::Error> {
        Ok(sql::types::ToSqlOutput::Owned(sql::types::Value::Integer(
            self.0,
        )))
    }
}
impl rusqlite::types::FromSql for PatchID {
    fn column_result(x: sql::types::ValueRef<'_>) -> Result<Self, sql::types::FromSqlError> {
        Ok(PatchID(i64::column_result(x)?))
    }
}

pub type PatchRequest = Vec<AxisSelection>;

/// Selection by axis labels, similar to .loc[] in Pandas
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
#[serde(tag = "type")]
pub enum AxisSelection {
    All {
        name: String,
    },
    RangeInclusive {
        name: String,
        start: Label,
        end: Label,
    },
    Labels {
        name: String,
        labels: Vec<Label>,
    },
}

/// Selection by axis indicess, similar to .iloc[] in Pandas
pub(crate) type AxisSegment = (usize, usize);

/// An N-dimensional box referencing a contiguous region of multiple axes.
///
/// Remember that in these boxes, storage indices (usize) are always consecutive, but labels (i64) may not be.
pub(crate) type BoundingBox = Vec<AxisSegment>;
