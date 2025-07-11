//! Bayesian distance clustering using cohesion and repulsion.
use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub mod mcmc;
mod mcmc_data;
mod mcmc_options;
mod mcmc_result;
mod mcmc_state;
mod prior;
mod sampling;
mod util_types;
mod utils;

pub use mcmc_data::*;
pub use mcmc_options::*;
pub use mcmc_result::*;
pub use mcmc_state::*;
pub use prior::*;
use util_types::*;
use utils::*;

/// Bayesian distance clustering using cohesion and repulsion.
#[pymodule]
fn redclust(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<MCMCOptions>()?;
	m.add_class::<PriorHyperParams>()?;
	m.add_class::<MCMCData>()?;
	m.add_class::<MCMCResult>()?;
	m.add_class::<MCMCState>()?;
	m.add_function(wrap_pyfunction!(mcmc::py_run_sampler, m)?)?;
	m.add_wrapped(wrap_pymodule!(prior::prior_pymodule))?;
	Ok(())
}
