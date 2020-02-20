use numpy::{IntoPyArray, PyArray1, PyArrayDyn};
use pyo3::prelude::*;

/// A sequence of distinct signed integer labels uniquely mapping to indices of an axis
///
///  - In a dense patch, it represents the storage order along one dimension
///  - In a catalog, it determines storage order for all the quilts
///  - If fully sparse patches are supported in the future, axes may then be permitted to repeat
#[pyclass]
pub struct Axis {
    pub inner: crate::Axis,
}
#[pymethods]
impl Axis {
    /// Create a new named axis from a copy of a sequence of integer labels
    ///
    /// Keep in mind that order matters:
    ///  - with a Patch, it specifies the order of elements in that patch
    ///  - in a Catalog, it specifies the order of storage (and the most efficient retrieval order)
    #[new]
    pub fn new(obj: &PyRawObject, name: String, labels: &PyArrayDyn<i64>) -> PyResult<()> {
        obj.init(Self {
            inner: crate::Axis::new(name, labels.as_array().iter().copied().collect())?,
        });
        Ok(())
    }

    /// Get the name of this axis
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Get a copy of the integer labels associated with each element of this axis
    pub fn labels<'py>(&self, py: Python<'py>) -> &'py PyArray1<i64> {
        Vec::from(self.inner.labels()).into_pyarray(py)
    }

    /// Merge the labels of two axes, removing duplicates and appending new elements
    ///
    /// This will not change labels in self, because downstream that means patches would need to
    /// be rebuilt, in the case of a catalog-level Axis.
    pub fn union(&mut self, other: &Self) {
        self.inner.union(&other.inner);
    }
}
