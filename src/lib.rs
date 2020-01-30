#![feature(is_sorted, result_cloned)]
#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;

/// The user-readable label for an axis (maybe not consecutive)
pub type Label = i64;
pub type PatchID = i64;

mod patch;
pub use patch::{Axis, Patch};

mod quilt;
pub use quilt::{AxisSelection, PatchRequest, Quilt, QuiltMeta};

mod catalog;
pub use catalog::{Catalog, MemoryCatalog, SQLiteCatalog};
