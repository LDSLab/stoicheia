//! Stoicheia Python API Bindings
//!
//! These bindings allow accessing stoicheia from Python, like this:
//!
//! ```py
//! from stoicheia import Catalog, Patch, Quilt, Axis
//! cat = Catalog("example.db")
//! ```
use crate::error::StoiError;
use crate::Catalog as CatalogTrait;
use itertools::Itertools;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

mod axis;
mod patch;

pub use axis::Axis;
pub use patch::Patch;

#[pymodule]
fn stoicheia(_py: Python, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn axpy(a: f64, x: ArrayViewD<f64>, y: ArrayViewD<f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m, "axpy")]
    fn axpy_py(
        py: Python,
        a: f64,
        x: &PyArrayDyn<f64>,
        y: &PyArrayDyn<f64>,
    ) -> Py<PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        axpy(a, x, y).into_pyarray(py).to_owned()
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = x.as_array_mut();
        mult(a, x);
        Ok(())
    }
    m.add_class::<crate::python::axis::Axis>()?;
    m.add_class::<crate::python::patch::Patch>()?;
    m.add_class::<Catalog>()?;
    Ok(())
}

// TODO: More detailed exception handling for Python bindings
impl From<crate::StoiError> for PyErr {
    fn from(s: StoiError) -> PyErr {
        PyErr::new::<pyo3::exceptions::ValueError, _>(format!("{:?}", s))
    }
}

#[pyclass]
pub struct Catalog {
    inner: Arc<crate::SQLiteCatalog>,
}

#[pymethods]
impl Catalog {
    /// Create a new catalog by connecting to a URL
    ///
    /// Or if you provide a file path, you'll get an SQLite based catalog.
    /// If you provide nothing, you'll get an in-memory SQLite based catalog.
    #[new]
    pub fn new(obj: &PyRawObject, url: Option<String>) -> PyResult<()> {
        let cat = match url {
            Some(path) => crate::SQLiteCatalog::connect(path.into())?,
            None => crate::SQLiteCatalog::connect_in_memory()?,
        };
        obj.init(Self { inner: cat });
        Ok(())
    }

    /// Create a new quilt in the catalog, given a name and the axes it uses
    pub fn create_quilt(&self, quilt_name: String, axes: Vec<String>) -> PyResult<()> {
        self.inner.create_quilt(
            &quilt_name,
            &axes.iter().map(|s| s.as_ref()).collect_vec()[..],
        )?;
        Ok(())
    }

    /// Fetch a patch from a quilt, assembling it from parts as necessary
    ///
    /// ```py
    /// # You can get a slice of any tensor
    /// patch = cat.fetch(
    ///     "tot_sal_amt",  # <- Quilt name
    ///     "latest",       # <- Tag name ("latest" is the default)
    ///     # The rest of the arguments specify the slice you want.
    ///
    ///     # You can select by axis labels (not by storage order!)
    ///     itm = [1,2,3],
    ///
    ///     # slice() or None both will get the whole axis.
    ///     lct = slice(),
    ///
    ///     # Giving just one label will not remove that axis
    ///     # (because that makes merging patches easier)
    ///     day = 721,
    /// )
    /// ```
    #[args(axes = "**")]
    pub fn fetch(
        &self,
        quilt_name: String,
        axes: Option<&PyDict>,
    ) -> PyResult<crate::python::Patch> {
        let mut axes_selections = vec![];

        // These type-check gymnastics is to handle the different ways to specify patch labels
        if let Some(x) = axes {
            // We need to iterate because the order matters and HashSet would have missed that
            for (k, v) in x.iter() {
                let axis_name = k.extract()?;

                if let Ok(selection) = v.extract::<Vec<i64>>() {
                    axes_selections.push(crate::AxisSelection::Labels {
                        name: axis_name,
                        labels: selection.into_iter().map(|l| crate::Label(l)).collect_vec(),
                    });
                } else if let Ok(selection) = v.extract::<(i64, i64)>() {
                    axes_selections.push(crate::AxisSelection::RangeInclusive {
                        name: axis_name,
                        start: crate::Label(selection.0),
                        end: crate::Label(selection.1),
                    });
                } else if let Ok(selection) = v.extract::<i64>() {
                    axes_selections.push(crate::AxisSelection::Labels {
                        name: axis_name,
                        labels: vec![crate::Label(selection)],
                    });
                } else {
                    // Play it safe and don't just ignore errors
                    Err(StoiError::InvalidValue(
                        "Didn't recognize one of the axis selections",
                    ))?;
                }
            }
        }

        Ok(crate::python::Patch {
            inner: self.inner.fetch(&quilt_name, axes_selections)?,
        })
    }
}
