use pyo3::prelude::*;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn};
use super::PyAxis;
use crate::Patch;
use failure::Fallible;

#[pyclass]
pub struct PyPatch {
    pub inner: Patch<f32>,
}

#[pymethods]
impl PyPatch {
    /// Create a new dense patch from a set of labels
    #[new]
    pub fn new(obj: &PyRawObject, axes: &PyAxis) -> PyResult<()> {
        obj.init(PyPatch {
            inner: Patch::from_axes(
                vec![ axes.inner.clone() ]
            ).unwrap()
        });
        Ok(())
    }

    pub fn to_dense(&self, py: Python) -> Py<PyArrayDyn<f32>> {
        self.inner.to_dense().into_pyarray(py).to_owned()
    }
}