use super::PyAxis;
use crate::{Fallible, Patch};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
pub struct PyPatch {
    pub inner: Patch<f32>,
}

#[pymethods]
impl PyPatch {
    /// Create a new dense patch from a set of labels
    #[new]
    pub fn new(obj: &PyRawObject, axes: &PyList) -> PyResult<()> {
        let axes: Vec<&PyAxis> = axes.extract()?;
        obj.init(PyPatch {
            inner: Patch::from_axes(axes.into_iter().map(|ax| ax.inner.clone()).collect())?,
        });
        Ok(())
    }

    /// Create a new patch from axes
    #[staticmethod]
    pub fn from_axes(axes: &PyList) -> PyResult<PyPatch> {
        let axes: Vec<&PyAxis> = axes.extract()?;
        Ok(PyPatch {
            inner: Patch::from_axes(axes.into_iter().map(|ax| ax.inner.clone()).collect())?,
        })
    }

    /// Make a copy of the dense array stored in this patch
    pub fn export<'py>(&self, py: Python<'py>) -> (Vec<&'py PyArray1<i64>>, Py<PyArrayDyn<f32>>) {
        (
            self.inner
                .axes()
                .iter()
                .map(|a| PyArray1::from_slice(py,
                    &a.labels()
                    .iter()
                    .map(|l| l.0)
                    .collect::<Vec<_>>()
                ))
                .collect(),
            self.inner.to_dense().into_pyarray(py).to_owned(),
        )
    }
}
