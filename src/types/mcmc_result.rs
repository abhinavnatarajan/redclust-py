use std::fmt::{Debug, Display, Formatter};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use pyo3::{prelude::*, types::PyDict};

use crate::{types::ClusterLabel, utils::float_vec_max};

#[derive(Debug, Clone, Accessors, PartialEq, Default)]
#[access(get, defaults(get(cp)))]
#[pyclass(get_all, str)]
/// Result from the MCMC sampler
pub struct MCMCResult {
	/// Cluster allocations - each column is one sample.
	#[access(get(cp = false))]
	pub(crate) clusts: Vec<Vec<ClusterLabel>>,

	/// Number of clusters in each sample.
	#[access(get(cp = false))]
	pub(crate) n_clusts: Vec<usize>,

	/// Samples of the parameter r.
	#[access(get(cp = false))]
	pub(crate) r: Vec<f64>,

	/// Samples of the parameter p.
	#[access(get(cp = false))]
	pub(crate) p: Vec<f64>,

	/// Log-likelihood of the samples.
	#[access(get(cp = false))]
	pub(crate) ln_lik: Vec<f64>,

	/// Log-posterior of the samples.
	#[access(get(cp = false))]
	pub(crate) ln_posterior: Vec<f64>,

	/// Acceptance rate for split-merge proposals.
	pub(crate) splitmerge_acceptance_rate: f64,

	/// Acceptance rate for the parameter r.
	pub(crate) r_acceptance_rate: f64,
}

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum PointEstimator {
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
			n_clusts: vec![0; n_samples],
			r: Vec::with_capacity(n_samples),
			p: Vec::with_capacity(n_samples),
			splitmerge_acceptance_rate: 0.0, // todo
			r_acceptance_rate: 0.0,
			ln_lik: Vec::with_capacity(n_samples),
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
		self.n_clusts.extend(other.n_clusts);
		self.r.extend(other.r);
		self.p.extend(other.p);
		self.ln_lik.extend(other.ln_lik);
		self.ln_posterior.extend(other.ln_posterior);
		self
	}
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
		dict.set_item("n_clusts", &me.n_clusts)?;
		dict.set_item("r", &me.r)?;
		dict.set_item("p", &me.p)?;
		dict.set_item("splitmerge_acceptance_rate", me.splitmerge_acceptance_rate)?;
		dict.set_item("r_acceptance_rate", me.r_acceptance_rate)?;
		Ok(dict)
	}
}

impl Display for MCMCResult {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> { Debug::fmt(self, f) }
}
