use numpy::{IntoPyArray, PyArray1, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
pub struct Patch {
    pub inner: crate::Patch<f32>,
}

#[pymethods]
impl Patch {
    /// Create a new dense patch from a set of labels
    #[new]
    pub fn new(obj: &PyRawObject, axes: &PyList) -> PyResult<()> {
        let axes: Vec<&super::Axis> = axes.extract()?;
        obj.init(Self {
            inner: crate::Patch::from_axes(axes.into_iter().map(|ax| ax.inner.clone()).collect())?,
        });
        Ok(())
    }

    /// Create a new patch from axes
    #[staticmethod]
    pub fn from_axes(axes: &PyList) -> PyResult<Self> {
        let axes: Vec<&super::Axis> = axes.extract()?;
        Ok(Self {
            inner: crate::Patch::from_axes(axes.into_iter().map(|ax| ax.inner.clone()).collect())?,
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
