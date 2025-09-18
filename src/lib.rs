//! Bayesian distance clustering using cohesion and repulsion.
use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub mod mcmc;
mod mcmc_data;
mod mcmc_model;
mod mcmc_options;
mod mcmc_result;
mod utils;

pub use mcmc::state::State;
pub use mcmc_data::MCMCData;
pub use mcmc_model::{LikelihoodOptions, PriorHyperParams};
pub use mcmc_options::MCMCOptions;
pub use mcmc_result::MCMCResult;
pub use utils::ClusterLabel;

/// Bayesian distance clustering using cohesion and repulsion.
#[pymodule]
fn redclust(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<MCMCOptions>()?;
	m.add_class::<PriorHyperParams>()?;
	m.add_class::<MCMCData>()?;
	m.add_class::<MCMCResult>()?;
	m.add_class::<State>()?;
	m.add_function(wrap_pyfunction!(mcmc::py_run_sampler, m)?)?;
	m.add_wrapped(wrap_pymodule!(mcmc_model::mcmc_model_pymodule))?;
	Ok(())
}
