use derive_more::{Debug, Display};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

mod mcmc_data;
mod mcmc_options;
mod mcmc_result;
mod mcmc_state;
mod prior_hyper_params;
pub use self::{
	mcmc_data::MCMCData,
	mcmc_options::MCMCOptions,
	mcmc_result::MCMCResult,
	mcmc_state::MCMCState,
	prior_hyper_params::PriorHyperParams,
};

pub type ClusterLabel = u32;

#[derive(Default, Debug, Display, Clone, PartialEq)]
pub(crate) struct Array2Wrapper<T>(pub(crate) Array2<T>);

impl<T> From<Array2Wrapper<T>> for Array2<T> {
	fn from(matrix: Array2Wrapper<T>) -> Self { matrix.0 }
}

impl<'a, T> From<&'a Array2Wrapper<T>> for &'a Array2<T> {
	fn from(matrix: &'a Array2Wrapper<T>) -> Self { &matrix.0 }
}

impl<'a, T> From<&'a mut Array2Wrapper<T>> for &'a mut Array2<T> {
	fn from(matrix: &'a mut Array2Wrapper<T>) -> Self { &mut matrix.0 }
}

impl<'py, T: numpy::Element> IntoPyObject<'py> for &Array2Wrapper<T> {
	type Error = PyErr;
	type Output = Bound<'py, PyArray2<T>>;
	type Target = PyArray2<T>;

	fn into_pyobject(self, py: Python<'py>) -> std::result::Result<Self::Output, Self::Error> {
		Ok(PyArray2::from_array(py, &self.0))
	}
}

#[derive(Default, Debug, Display, Clone, PartialEq)]
pub(crate) struct Array1Wrapper<T>(Array1<T>);

impl<T> From<Array1Wrapper<T>> for Array1<T> {
	fn from(matrix: Array1Wrapper<T>) -> Self { matrix.0 }
}

impl<'a, T> From<&'a Array1Wrapper<T>> for &'a Array1<T> {
	fn from(matrix: &'a Array1Wrapper<T>) -> Self { &matrix.0 }
}

impl<'a, T> From<&'a mut Array1Wrapper<T>> for &'a mut Array1<T> {
	fn from(matrix: &'a mut Array1Wrapper<T>) -> Self { &mut matrix.0 }
}

impl<'py, T: numpy::Element> IntoPyObject<'py> for &Array1Wrapper<T> {
	type Error = PyErr;
	type Output = Bound<'py, PyArray1<T>>;
	type Target = PyArray1<T>;

	fn into_pyobject(self, py: Python<'py>) -> std::result::Result<Self::Output, Self::Error> {
		Ok(PyArray1::from_array(py, &self.0))
	}
}
