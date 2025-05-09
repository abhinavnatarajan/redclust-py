use std::fmt::{Debug, Display, Formatter};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use pyo3::{prelude::*, types::PyDict};

use crate::{
	types::{Array2Wrapper, ClusterLabel, MCMCOptions, PriorHyperParams},
	utils::float_vec_max,
};

#[derive(Debug, Clone, Accessors, PartialEq)]
#[access(get, defaults(get(cp)))]
#[pyclass(get_all, str)]
/// Result from the MCMC sampler
pub struct MCMCResult {
	/// Cluster allocations - each column is one sample.
	#[access(get(cp = false))]
	pub(crate) clusts: Vec<Vec<ClusterLabel>>,

	/// Matrix of marginal posterior co-clustering probabilities.
	#[access(get(cp = false))]
	pub(crate) posterior_coclustering: Array2Wrapper<f64>,

	/// Number of clusters in each sample.
	#[access(get(cp = false))]
	pub(crate) n_clusts: Vec<usize>,

	/// Effective sample size for n_clusts.
	pub(crate) n_clusts_ess: f64,

	/// Autocorrelation function for n_clusts.
	#[access(get(cp = false))]
	pub(crate) n_clusts_acf: Vec<f64>,

	/// Integrated autocorrelation time for n_clusts.
	pub(crate) n_clusts_iac: f64,

	/// Mean of n_clusts.
	pub(crate) n_clusts_mean: f64,

	/// Variance of n_clusts.
	pub(crate) n_clusts_variance: f64,

	/// Samples of the parameter r.
	#[access(get(cp = false))]
	pub(crate) r: Vec<f64>,

	/// Effective sample size for parameter r.
	pub(crate) r_ess: f64,

	/// Autocorrelation function for parameter r.
	#[access(get(cp = false))]
	pub(crate) r_acf: Vec<f64>,

	/// Integrated autocorrelation time for parameter r.
	pub(crate) r_iac: f64,

	/// Mean of parameter r across samples.
	pub(crate) r_mean: f64,

	/// Variance of parameter r across samples.
	pub(crate) r_variance: f64,

	/// Samples of the parameter p.
	#[access(get(cp = false))]
	pub(crate) p: Vec<f64>,

	/// Effective sample size for parameter p.
	pub(crate) p_ess: f64,

	/// Autocorrelation function for parameter p.
	#[access(get(cp = false))]
	pub(crate) p_acf: Vec<f64>,

	/// Integrated autocorrelation time for parameter p.
	pub(crate) p_iac: f64,

	/// Mean of parameter p across samples.
	pub(crate) p_mean: f64,

	/// Variance of parameter p across samples.
	pub(crate) p_variance: f64,

	/// Record of split-merge proposal acceptances.
	#[access(get(cp = false))]
	pub(crate) splitmerge_acceptances: Vec<bool>,

	/// Record of which split-merge proposals were splits (true) vs merges
	/// (false).
	#[access(get(cp = false))]
	pub(crate) splitmerge_splits: Vec<bool>,

	/// Acceptance rate for split-merge proposals.
	pub(crate) splitmerge_acceptance_rate: f64,

	/// Record of acceptances in the Metropolis-Hastings step for r.
	#[access(get(cp = false))]
	pub(crate) r_acceptances: Vec<bool>,

	/// Acceptance rate for the parameter r.
	pub(crate) r_acceptance_rate: f64,

	/// Total runtime of the MCMC in milliseconds.
	pub(crate) runtime: f64,

	/// Average time per iteration in milliseconds.
	pub(crate) mean_iter_time: f64,

	/// Log-likelihood values across iterations.
	#[access(get(cp = false))]
	pub(crate) ln_lik: Vec<f64>,

	/// Log-posterior values across iterations.
	#[access(get(cp = false))]
	pub(crate) ln_posterior: Vec<f64>,

	/// MCMC options used.
	#[access(get(cp = false))]
	pub(crate) options: MCMCOptions,

	/// Prior hyperparameters used.
	#[access(get(cp = false))]
	pub(crate) params: PriorHyperParams,
}

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum PointEstimator {
	/// Maximum likelihood estimate (MLE).
	MaxLikelihood,
	/// Maximum a posteriori (MAP) estimate.
	MaxPosteriorProb,
}

#[pymethods]
impl MCMCResult {
	/// Point estimate clustering.
	pub fn point_estimate(&self, method: PointEstimator) -> Result<Vec<ClusterLabel>> {
		let objective = match method {
			PointEstimator::MaxPosteriorProb => &self.ln_posterior,
			PointEstimator::MaxLikelihood => &self.ln_lik,
		};
		let idx = float_vec_max(objective)
			.map_err(|e| anyhow!("Error while optimising objective: {}", e))?
			.0;
		Ok(self.clusts[idx].clone())
	}

	/// Convert the MCMCResult to a dictionary.
	fn as_dict(this: Bound<'_, Self>) -> Result<Bound<'_, PyDict>> {
		let me = this.borrow();
		let dict: Bound<'_, PyDict> = PyDict::new(this.py());
		dict.set_item("clusts", &me.clusts)?;
		dict.set_item("posterior_coclustering", &me.posterior_coclustering)?;
		dict.set_item("n_clusts", &me.n_clusts)?;
		dict.set_item("n_clusts_ess", me.n_clusts_ess)?;
		dict.set_item("n_clusts_acf", &me.n_clusts_acf)?;
		dict.set_item("n_clusts_iac", me.n_clusts_iac)?;
		dict.set_item("n_clusts_mean", me.n_clusts_mean)?;
		dict.set_item("n_clusts_variance", me.n_clusts_variance)?;
		dict.set_item("r", &me.r)?;
		dict.set_item("r_ess", me.r_ess)?;
		dict.set_item("r_acf", &me.r_acf)?;
		dict.set_item("r_iac", me.r_iac)?;
		dict.set_item("r_mean", me.r_mean)?;
		dict.set_item("r_variance", me.r_variance)?;
		dict.set_item("p", &me.p)?;
		dict.set_item("p_ess", me.p_ess)?;
		dict.set_item("p_acf", &me.p_acf)?;
		dict.set_item("p_iac", me.p_iac)?;
		dict.set_item("p_mean", me.p_mean)?;
		dict.set_item("p_variance", me.p_variance)?;
		dict.set_item("splitmerge_acceptances", &me.splitmerge_acceptances)?;
		dict.set_item("splitmerge_splits", &me.splitmerge_splits)?;
		dict.set_item("splitmerge_acceptance_rate", me.splitmerge_acceptance_rate)?;
		dict.set_item("r_acceptances", &me.r_acceptances)?;
		dict.set_item("r_acceptance_rate", me.r_acceptance_rate)?;
		dict.set_item("runtime", me.runtime)?;
		dict.set_item("mean_iter_time", me.mean_iter_time)?;
		dict.set_item("loglik", &me.ln_lik)?;
		dict.set_item("logposterior", &me.ln_posterior)?;
		dict.set_item("options", me.options)?;
		dict.set_item("params", me.params.clone())?;
		Ok(dict)
	}
}

impl Display for MCMCResult {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> { Debug::fmt(self, f) }
}
