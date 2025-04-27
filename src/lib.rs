//! Bayesian distance clustering using cohesion and repulsion.
use pyo3::prelude::*;

mod fit_prior;
pub mod mcmc;
mod sampling;
mod types;

pub use types::{MCMCData, MCMCOptions, MCMCResult, MCMCState, PriorHyperParams};

/// Bayesian distance clustering using cohesion and repulsion.
#[pymodule]
fn redclust(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<types::MCMCOptions>()?;
	m.add_class::<types::PriorHyperParams>()?;
	m.add_class::<types::MCMCData>()?;
	m.add_class::<types::MCMCResult>()?;
	m.add_class::<types::MCMCState>()?;
	m.add_function(wrap_pyfunction!(mcmc::ln_likelihood, m)?)?;
	m.add_function(wrap_pyfunction!(mcmc::ln_prior, m)?)?;
	Ok(())
}
