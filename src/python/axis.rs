use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn};
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
    /// Create a new named axis with a set of labels
    #[new]
    pub fn new(obj: &PyRawObject, name: String, labels: &PyArrayDyn<i64>) -> PyResult<()> {
        obj.init(Self {
            inner: crate::Axis::new(name, labels.as_array().iter().map(|&i| crate::Label(i)).collect())?,
        });
        Ok(())
    }

    /// Get a copy of the integer labels associated with each element of this axis
    pub fn labels(&self, py: Python) -> Py<PyArrayDyn<i64>> {
        self.inner
            .labels()
            .iter()
            .map(|label| label.0)
            .collect::<Array1<i64>>()
            .into_dyn()
            .into_pyarray(py)
            .to_owned()
    }

    /// Merge the labels of two axes, removing duplicates and appending new elements
    ///
    /// This will not change labels in self, because downstream that means patches would need to
    /// be rebuilt, in the case of a catalog-level Axis.
    pub fn union(&mut self, other: &Self) {
        self.inner.union(&other.inner);
    }
}
