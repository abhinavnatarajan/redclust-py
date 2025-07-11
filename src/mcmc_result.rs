use std::fmt::{Debug, Display, Formatter};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use pyo3::{prelude::*, types::PyDict};

use crate::*;

#[derive(Debug, Clone, Accessors, PartialEq, Default)]
#[access(get, defaults(get(cp = false)))]
#[pyclass(get_all, str)]
/// Result from the MCMC sampler
pub struct MCMCResult {
	/// Cluster allocations - each column is one sample.
	pub(crate) clusts: Vec<Vec<ClusterLabel>>,

	/// Number of clusters in each sample.
	pub(crate) num_clusts: Vec<usize>,

	/// Samples of the parameter r.
	pub(crate) r: Vec<f64>,

	/// Samples of the parameter p.
	pub(crate) p: Vec<f64>,

	/// Log-likelihood of the samples.
	pub(crate) ln_likelihood: Vec<f64>,

	/// Log-posterior of the samples.
	pub(crate) ln_posterior: Vec<f64>,

	/// Acceptance rate for split-merge proposals.
	#[access(get(cp = true))]
	pub(crate) splitmerge_acceptance_rate: f64,

	/// Acceptance rate for the parameter r.
	#[access(get(cp = true))]
	pub(crate) r_acceptance_rate: f64,
}

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum PointEstimatorMethod {
	/// Maximum likelihood estimate (MLE).
	MaxLikelihood,
	/// Maximum a posteriori (MAP) estimate.
	MaxPosteriorProb,
}

impl MCMCResult {
	/// Create a new MCMCResult with enough allocation for a given number of
	/// points and samples.
	pub fn with_capacity(n_pts: usize, n_samples: usize) -> Self {
		Self {
			clusts: vec![Vec::with_capacity(n_pts); n_samples],
			num_clusts: vec![0; n_samples],
			r: Vec::with_capacity(n_samples),
			p: Vec::with_capacity(n_samples),
			splitmerge_acceptance_rate: 0.0, // todo
			r_acceptance_rate: 0.0,
			ln_likelihood: Vec::with_capacity(n_samples),
			ln_posterior: Vec::with_capacity(n_samples),
		}
	}

	/// Merge two MCMCResults. Useful when combining results from multiple runs.
	pub fn merge_with(mut self, other: Self) -> Self {
		let self_n_samples = self.clusts.len();
		let other_n_samples = other.clusts.len();
		self.r_acceptance_rate = (self.r_acceptance_rate * self_n_samples as f64
			+ other.r_acceptance_rate * other_n_samples as f64)
			/ (self_n_samples + other_n_samples) as f64;
		self.splitmerge_acceptance_rate = (self.splitmerge_acceptance_rate * self_n_samples as f64
			+ other.splitmerge_acceptance_rate * other_n_samples as f64)
			/ (self_n_samples + other_n_samples) as f64;
		self.clusts.extend(other.clusts);
		self.num_clusts.extend(other.num_clusts);
		self.r.extend(other.r);
		self.p.extend(other.p);
		self.ln_likelihood.extend(other.ln_likelihood);
		self.ln_posterior.extend(other.ln_posterior);
		self
	}
}

impl Display for MCMCResult {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> { Debug::fmt(self, f) }
}

#[pymethods]
impl MCMCResult {
	/// Point estimate clustering.
	pub fn point_estimate(&self, method: PointEstimatorMethod) -> Result<Vec<ClusterLabel>> {
		let objective = match method {
			PointEstimatorMethod::MaxPosteriorProb => &self.ln_posterior,

			PointEstimatorMethod::MaxLikelihood => &self.ln_likelihood,
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
		dict.set_item("num_clusts", &me.num_clusts)?;
		dict.set_item("r", &me.r)?;
		dict.set_item("p", &me.p)?;
		dict.set_item("splitmerge_acceptance_rate", me.splitmerge_acceptance_rate)?;
		dict.set_item("r_acceptance_rate", me.r_acceptance_rate)?;
		dict.set_item("ln_likelihood", &me.ln_likelihood)?;
		dict.set_item("ln_posterior", &me.ln_posterior)?;
		Ok(dict)
	}
}
