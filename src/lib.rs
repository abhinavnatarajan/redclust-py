//! Bayesian distance clustering using cohesion and repulsion.
use pyo3::{prelude::*, wrap_pymodule};

mod input_data;
mod mcmc;
mod mcmc_options;
mod model;
mod result;
mod utils;

pub use mcmc::ln_likelihood;
pub use mcmc::ln_prior;
pub use mcmc::run_sampler;
pub use input_data::InputData;
pub use mcmc::state::State;
pub use mcmc_options::MCMCOptions;
pub use model::{
	ClusterSizePrior,
	InterClusterDissimilarityPrior,
	LikelihoodOptions,
	NumClustersPrior,
	PriorHyperParams,
};
pub use result::MCMCResult;
#[doc(inline)]
pub use utils::ClusterLabel;

/// Bayesian distance clustering using cohesion and repulsion.
#[cfg(feature = "python-module")]
#[pymodule]
fn redclust(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<MCMCOptions>()?;
	m.add_class::<PriorHyperParams>()?;
	m.add_class::<InputData>()?;
	m.add_class::<MCMCResult>()?;
	m.add_class::<State>()?;
	m.add_function(wrap_pyfunction!(mcmc::py_run_sampler, m)?)?;
	m.add_wrapped(wrap_pymodule!(model::mcmc_model_pymodule))?;
	Ok(())
}
