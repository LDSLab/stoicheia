#![feature(is_sorted)]
#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;

mod patch;
pub use patch::Patch;

mod quilt;
pub use quilt::{PatchRequest, PatchSelection, Quilt, QuiltMeta};

mod catalog;
pub use catalog::Catalog;
