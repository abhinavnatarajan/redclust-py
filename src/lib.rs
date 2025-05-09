//! Bayesian distance clustering using cohesion and repulsion.
use pyo3::prelude::*;

pub mod mcmc;
mod sampling;
mod types;
mod utils;

pub use types::{MCMCData, MCMCOptions, MCMCResult, MCMCState, PriorHyperParams};

/// Bayesian distance clustering using cohesion and repulsion.
#[pymodule]
fn redclust(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<types::MCMCOptions>()?;
	m.add_class::<types::PriorHyperParams>()?;
	m.add_class::<types::MCMCData>()?;
	m.add_class::<types::MCMCResult>()?;
	m.add_class::<types::MCMCState>()?;
	m.add_function(wrap_pyfunction!(mcmc::py_run_sampler, m)?)?;
	Ok(())
}
