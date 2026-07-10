pub mod explainer;

use faer::{ColRef, MatRef};
use numpy::ndarray::{ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::explainer::StrideExplainer;

/// Converts `numpy` views into `faer` reference types.
pub trait IntoFaer {
    type Faer;
    fn into_faer(self) -> Self::Faer;
}

/// Converts `faer` types into `ndarray` views.
pub trait IntoNdarray {
    type Ndarray;
    fn into_ndarray(self) -> Self::Ndarray;
}

impl<'a, T: numpy::Element + 'a> IntoFaer for PyReadonlyArray2<'a, T> {
    type Faer = MatRef<'a, T>;

    /// Converts a `PyReadonlyArray2` into a `faer::MatRef`.
    fn into_faer(self) -> Self::Faer {
        let raw_arr = self.as_raw_array();
        let nrows = raw_arr.nrows();
        let ncols = raw_arr.ncols();
        let strides: [isize; 2] = raw_arr.strides().try_into().unwrap();
        unsafe { MatRef::from_raw_parts(raw_arr.as_ptr(), nrows, ncols, strides[0], strides[1]) }
    }
}

impl<'a, T: numpy::Element + 'a> IntoFaer for PyReadonlyArray1<'a, T> {
    type Faer = ColRef<'a, T>;

    /// Converts a `PyReadonlyArray1` into a `faer::ColRef`.
    fn into_faer(self) -> Self::Faer {
        let raw_arr = self.as_raw_array();
        let nrows = raw_arr.len();
        let strides: [isize; 1] = raw_arr.strides().try_into().unwrap();
        unsafe { ColRef::from_raw_parts(raw_arr.as_ptr(), nrows, strides[0]) }
    }
}

impl<'a, T> IntoNdarray for MatRef<'a, T> {
    type Ndarray = ArrayView2<'a, T>;

    /// Converts a `faer::MatRef` into an `ndarray::ArrayView2`.
    fn into_ndarray(self) -> Self::Ndarray {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride() as usize;
        let col_stride = self.col_stride() as usize;
        unsafe {
            ArrayView2::from_shape_ptr(
                (nrows, ncols).strides((row_stride, col_stride)),
                self.as_ptr(),
            )
        }
    }
}

impl<'a, T> IntoNdarray for ColRef<'a, T> {
    type Ndarray = ArrayView1<'a, T>;

    /// Converts a `faer::ColRef` into an `ndarray::ArrayView1`.
    fn into_ndarray(self) -> Self::Ndarray {
        let nrows = self.nrows();
        let row_stride = self.row_stride() as usize;
        unsafe { ArrayView1::from_shape_ptr(nrows.strides(row_stride), self.as_ptr()) }
    }
}

#[pyclass]
pub struct Striders {
    inner: StrideExplainer,
}

#[pymethods]
impl Striders {
    #[new]
    #[pyo3(signature = (num_bases=50, lambda_reg=0.01))]
    pub fn pynew(num_bases: usize, lambda_reg: f32) -> Self {
        Self {
            inner: StrideExplainer::new(num_bases, lambda_reg),
        }
    }

    pub fn fit(&mut self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray1<f32>) {
        let x_mat = x.into_faer();
        let y_col = y.into_faer();
        self.inner.fit(x_mat, y_col);
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_mat = x.into_faer();
        let pred = self.inner.predict(x_mat);
        let py_pred = pred.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_pred)
    }

    pub fn explain<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = x.into_faer();
        let strides = self.inner.explain(x_mat);
        let py_strides = strides.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_strides)
    }
}

#[pymodule]
fn striders(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Striders>()?;
    Ok(())
}
