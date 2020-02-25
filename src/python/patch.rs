use numpy::{IntoPyArray, PyArray1, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
pub struct Patch {
    pub inner: crate::Patch,
}

#[pymethods]
impl Patch {
    /// Create a new patch from label axes and content
    ///
    /// This copies the content and labels to prevent mutation,
    /// so it's not very efficient.
    #[new]
    pub fn new(obj: &PyRawObject, axes: &PyList, content: &PyArrayDyn<f32>) -> PyResult<()> {
        let axes: Vec<&super::Axis> = axes.extract()?;
        obj.init(Self {
            inner: crate::Patch::new(
                axes.into_iter().map(|ax| ax.inner.clone()).collect(),
                Some(content.as_array().to_owned()),
            )?,
        });
        Ok(())
    }

    /// Create a new patch from label axes and content
    ///
    /// The shape of the new patch will be the lengths of the axes, so be careful,
    /// it's easy to create huge tensors by accident this way.
    ///
    /// It makes a copy of the axes, which is usually fine.
    #[staticmethod]
    pub fn try_from_axes(axes: &PyList) -> PyResult<Self> {
        let axes: Vec<&super::Axis> = axes.extract()?;
        Ok(Self {
            inner: crate::Patch::new(
                axes.into_iter().map(|ax| ax.inner.clone()).collect(),
                None
            )?,
        })
    }

    /// Export this patch to a list of axes and a content array
    ///
    /// This copies the content to prevent mutation, so it's not very efficient.
    pub fn export<'py>(&self, py: Python<'py>) -> (Vec<&'py PyArray1<i64>>, Py<PyArrayDyn<f32>>) {
        (
            self.inner
                .axes()
                .iter()
                .map(|a| PyArray1::from_slice(py, a.labels()))
                .collect(),
            self.inner.to_dense().into_pyarray(py).to_owned(),
        )
    }
}
