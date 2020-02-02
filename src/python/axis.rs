use crate::{Axis, Label};
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::prelude::*;

/// A sequence of signed integer labels uniquely mapping to indices of an axis
///
/// The meaning and restrictions of the labels depend a lot on the context
///
///  - In a dense patch, it represents the storage order along one dimension,
///    and it needs to be unique.
///  - In a sparse patch, it is the coordinates of each populated cell,
///    and it does not need to be unique.
///  - In a catalog, it determines storage order for all the quilts,
///    and it needs to be unique.
#[pyclass]
pub struct PyAxis {
    pub inner: Axis,
}
#[pymethods]
impl PyAxis {
    /// Create a new named axis with a set of labels
    #[new]
    pub fn new(obj: &PyRawObject, name: String, labels: &PyArrayDyn<i64>) {
        obj.init(PyAxis {
            inner: Axis::new(name, labels.as_array().iter().map(|&i| Label(i)).collect()),
        });
    }

    /// Get a copy of the integer labels associated with each element of this axis
    pub fn labels(&self, py: Python) -> Py<PyArrayDyn<i64>> {
        self.inner
            .labels
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
    pub fn union(&mut self, other: &PyAxis) {
        self.inner.union(&other.inner);
    }
}
