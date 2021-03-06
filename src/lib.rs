#![feature(is_sorted, result_cloned)]
//! Sharded tensor storage and retrieval
#[macro_use]
extern crate serde_derive;
#[allow(unused_imports)]
#[macro_use] // Macro used in tests
extern crate ndarray as nd;
extern crate rusqlite as sql;
#[allow(unused_imports)]
#[macro_use] // Macro used in tests
extern crate approx; // for approximately eq for f32/f64

mod patch;
pub use patch::{ContentPattern, Patch, PatchCompressionType};

mod catalog;
pub use catalog::{Catalog, QuiltDetails, StorageTransaction};

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

/// Patch details, everything you can get without loading the content
///
/// These are used for compaction and balancing
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct PatchRef {
    id: PatchID,
    bounding_box: BoundingBox,
    decompressed_size: u64,
}

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

/// Selection by axis labels, similar to .loc[] in Pandas
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub enum AxisSelection {
    All,
    LabelSlice(Label, Label),
    Labels(Vec<Label>),
    StorageSlice(usize, usize),
}

/// Selection by axis indices, similar to .iloc[] in Pandas
pub(crate) type AxisSegment = (usize, usize);

/// A 4-dimensional box referencing a contiguous region of multiple axes.
///
/// Remember that in these boxes, storage indices (usize) are always consecutive,
/// but labels (i64) may not be.
pub(crate) type BoundingBox = [AxisSegment; 4];

/// Performance metrics
///
/// In most cases you should treat these as implementation details
/// but for debugging performance problems it can be quite valuable.
/// Use these with transaction method get_performance_counters()
#[derive(PartialEq, Eq, Debug, Clone, Copy, enum_map::Enum)]
pub enum Counter {
    /// An axis was deserialized.
    /// This typically runs ~100MB/s so it's modestly expensive.
    ReadAxis,
    /// An axis label was serialized.
    /// Axes store labels individually so the whole axis isn't serialized at once.
    /// Instead, we note the number of times an axis label was added
    WriteAxisLabel,
    /// The global axis tables were tested for presence of an axis label
    /// If trials >> writes, then the axis cache is not performing well; likely a bug
    TrialAxisLabel,

    /// A patch was deserialized.
    /// This is typically the largest IO. It should be 100+ MB/s but patches are large
    ReadPatch,
    /// A patch was serialized.
    /// This can be very slow, even less than 10 MB/s. These should be carefully minimized.
    WritePatch,
    /// A patch was deleted.
    /// This is pretty fast and usually indicates rebalancing is going on
    DeletePatch,

    /// The patch index was searched.
    /// This is typically fast, but can spike with certain label-based queries
    /// because they make create very many bounding boxes
    SearchPatches,

    /// Estimated total bytes of IO. (serialized)
    ReadBytes,
    /// Estimated total bytes of IO. (serialized)
    WriteBytes,

    /// Created a commit
    CreateCommit,
    /// Fetched a patch
    Fetch,
    /// Resolved an axis selection into a labelset
    ResolveSelection,

    MaybeSplit,
    Split,
    GetBoundingBox,
    PutCommit,
    PutCommitGetPatch,
    PutCommitFetch,
}
