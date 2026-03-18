mod explainer;

use crate::explainer::StrideExplainer;
use faer::{Col, Mat};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[pyclass]
pub struct Striders {
    inner: StrideExplainer,
}

#[pymethods]
impl Striders {
    #[new]
    #[pyo3(signature = (num_bases=50, lambda_reg=0.01, sigma=0.5))]
    pub fn pynew(num_bases: usize, lambda_reg: f32, sigma: f32) -> Self {
        Self {
            inner: StrideExplainer::new(num_bases, lambda_reg, sigma),
        }
    }

    pub fn fit(&mut self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray1<f32>) {
        let x_view = x.as_array();
        let y_view = y.as_array();

        let n = x_view.shape()[0];
        let p = x_view.shape()[1];
        let x_mat = Mat::<f32>::from_fn(n, p, |i, j| x_view[[i, j]]);
        let y_col = Col::<f32>::from_fn(n, |i| y_view[i]);
        self.inner.fit(&x_mat, &y_col);
    }

    pub fn explain<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>)> {
        let x_view = x.as_array();
        let (n, p) = (x_view.shape()[0], x_view.shape()[1]);
        let x_mat = Mat::<f32>::from_fn(n, p, |i, j| x_view[[i, j]]);
        let (pred, strides) = self.inner.explain(&x_mat);
        let pred_ndarray = Array1::<f32>::from_iter(pred.iter().cloned());
        let py_pred = pred_ndarray.to_pyarray(py);
        let strides_ndarray = Array2::from_shape_fn((n, p), |(i, j)| strides[(i, j)]);
        let py_strides = strides_ndarray.to_pyarray(py);
        Ok((py_pred, py_strides))
    }
}

#[pymodule]
fn striders(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Striders>()?;
    Ok(())
}
