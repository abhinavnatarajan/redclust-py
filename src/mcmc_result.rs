use std::fmt::{Debug, Display, Formatter};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use itertools::Itertools;
use ndarray::{ArrayView1, s};
#[cfg(feature = "python-module")]
use pyo3::{prelude::*, types::PyDict};

use crate::*;

/// Auxiliary statistics for MCMC samples: mean, variance, autocorrelation
/// function, integrated autocorrelation time, and effective sample size.
#[derive(Debug, PartialEq, Clone, Default, Accessors)]
#[access(get, defaults(get(cp)))]
#[cfg_attr(feature = "python-module", pyclass(get_all, str, eq))]
pub struct AuxStats {
	/// Mean.
	pub(crate) mean: f64,
	/// Variance.
	pub(crate) var: f64,
	/// Autocorrelation function.
	#[access(get(cp = false))]
	pub(crate) acf: Vec<f64>,
	/// Integrated autocorrelation time.
	pub(crate) iac: f64,
	/// Effective sample size.
	pub(crate) ess: f64,
}

impl Display for AuxStats {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> { Debug::fmt(self, f) }
}

/// Mean, variance, autocorrelation function, integrated autocorrelation time,
/// and effective sample size of a vector.
fn aux_stats(x: &ArrayView1<f64>) -> AuxStats {
	if x.is_empty() {
		return AuxStats::default();
	}
	let lags = (0..(x.len() - 1).min(10 * (x.len() as f64).log10() as usize)).collect_vec();
	let n = x.len();
	let mean = x.mean().unwrap();
	let var = x.var(0.0);
	let y = x - mean;
	let mut acf = Vec::with_capacity(lags.len());
	for lag in lags {
		if lag >= n {
			acf.push(0.0);
		} else {
			acf.push(y.slice(s![..n - lag]).dot(&y.slice(s![lag..])) / var);
		}
	}
	let iac = acf.iter().sum::<f64>() * 2.0;
	let ess = x.len() as f64 / iac;
	AuxStats {
		mean,
		var,
		acf,
		iac,
		ess,
	}
}

#[cfg(feature = "python-module")]
#[pymethods]
impl AuxStats {
	fn as_dict(this: Bound<'_, Self>) -> Result<Bound<'_, PyDict>> {
		let this = this.borrow();
		let dict: Bound<'_, PyDict> = PyDict::new(this.py());
		dict.set_item("mean", this.mean)?;
		dict.set_item("var", this.var)?;
		dict.set_item("autocorrelation_function", &this.acf)?;
		dict.set_item("integrated_autocorrelation", this.iac)?;
		dict.set_item("effective_sample_size", this.ess)?;
		Ok(dict)
	}
}

#[derive(Debug, Clone, Accessors, PartialEq, Default)]
#[access(get, defaults(get(cp = false)))]
#[cfg_attr(feature = "python-module", pyclass(get_all, str, eq))]
/// Results from the MCMC sampler.
pub struct MCMCResult {
	/// Cluster allocations - each column is one sample.
	pub(crate) clusts: Vec<Vec<ClusterLabel>>,

	/// Number of clusters in each sample.
	pub(crate) num_clusts: Vec<usize>,

	/// Auxiliary statistics for samples of the number of clusters.
	pub(crate) num_clusts_aux: AuxStats,

	/// Samples of the parameter r.
	pub(crate) r: Vec<f64>,

	/// Auxiliary statistics for the samples of r.
	pub(crate) r_aux: AuxStats,

	/// Samples of the parameter p.
	pub(crate) p: Vec<f64>,

	/// Auxiliary statistics for the samples of p.
	pub(crate) p_aux: AuxStats,

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
			num_clusts_aux: AuxStats::default(),
			r: Vec::with_capacity(n_samples),
			r_aux: AuxStats::default(),
			p: Vec::with_capacity(n_samples),
			p_aux: AuxStats::default(),
			splitmerge_acceptance_rate: 0.0, // todo
			r_acceptance_rate: 0.0,
			ln_likelihood: Vec::with_capacity(n_samples),
			ln_posterior: Vec::with_capacity(n_samples),
		}
	}

	/// Compute the auxiliary statistics.
	fn compute_aux_stats(&mut self) -> &mut Self {
		self.num_clusts_aux = aux_stats(&ArrayView1::from(
			&self.num_clusts.iter().map(|&x| x as f64).collect_vec()[..],
		));
		self.r_aux = aux_stats(&ArrayView1::from(&self.r[..]));
		self.p_aux = aux_stats(&ArrayView1::from(&self.p[..]));
		self
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
		self.compute_aux_stats();
		self
	}
}

impl Display for MCMCResult {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> { Debug::fmt(self, f) }
}

#[cfg(feature = "python-module")]
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
		let this = this.borrow();
		let py = this.py();
		let dict: Bound<'_, PyDict> = PyDict::new(this.py());
		dict.set_item("clusts", &this.clusts)?;
		dict.set_item("num_clusts", &this.num_clusts)?;
		dict.set_item(
			"num_clusts_aux",
			AuxStats::as_dict(this.num_clusts_aux.clone().into_pyobject(py)?)?,
		)?;
		dict.set_item("r", &this.r)?;
		dict.set_item(
			"r_aux",
			AuxStats::as_dict(this.r_aux.clone().into_pyobject(py)?)?,
		)?;
		dict.set_item("p", &this.p)?;
		dict.set_item(
			"p_aux",
			AuxStats::as_dict(this.p_aux.clone().into_pyobject(py)?)?,
		)?;
		dict.set_item(
			"splitmerge_acceptance_rate",
			this.splitmerge_acceptance_rate,
		)?;
		dict.set_item("r_acceptance_rate", this.r_acceptance_rate)?;
		dict.set_item("ln_likelihood", &this.ln_likelihood)?;
		dict.set_item("ln_posterior", &this.ln_posterior)?;
		Ok(dict)
	}
}
