use crate::*;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::prelude::*;

mod axis;
mod patch;

pub use axis::PyAxis;
pub use patch::PyPatch;

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
    m.add_class::<axis::PyAxis>()?;
    m.add_class::<patch::PyPatch>()?;
    Ok(())
}

// TODO: More detailed exception handling for Python bindings
impl From<crate::StoiError> for PyErr {
    fn from(s: StoiError) -> PyErr {
        PyErr::new::<pyo3::exceptions::ValueError, _>(s.to_string())
    }
}
