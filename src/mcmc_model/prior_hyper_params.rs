use std::{
	collections::HashMap,
	fmt::{Debug, Display, Formatter},
	num::NonZeroUsize,
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use ndarray::Array1;
#[cfg(feature = "python-module")]
use pyo3::prelude::*;
use rand::distributions::Distribution;
use statrs::distribution::{Beta, Gamma};

use super::{InterClusterDissimilarityPrior, IntraClusterDissimilarityPrior};
use crate::{
	mcmc_model::{ClusterSizePrior, NumClustersPrior},
	utils::Array1Wrapper,
	utils::get_rng,
};

/// Prior hyper-parameters for the Bayesian distance clustering algorithm.
#[derive(Debug, Accessors, PartialEq)]
#[access(get, defaults(get(cp)))]
#[cfg_attr(feature = "python-module", pyclass(get_all, str, eq))]
pub struct PriorHyperParams {
	/// Shape hyperparameter for the Gamma likelihood on intra-cluster
	/// dissimilarities. Smaller values lead to greater within-cluster
	/// cohesion.
	delta1: f64,

	/// Shape hyperparameter for the Gamma likelihood on inter-cluster
	/// dissimilarities. Larger values lead to greater inter-cluster repulsion.
	delta2: f64,

	/// Shape hyperparameter for the Gamma prior on
	/// the rate parameters of the Gamma likelihood for intra-cluster
	/// dissimilarities. on intra-cluster distances in the k-th cluster.
	/// Smaller values for alpha lead to more cohesion and less variability
	/// within clusters.
	alpha: f64,

	/// Rate hyperparameter for the Gamma prior on the
	/// rate parameters of the Gamma likelihood on intra-cluster distances.
	/// Larger values for beta lead to less cohesion and greater variability
	/// within clusters.
	beta: f64,

	/// Rate hyperparameter for the Gamma prior on the
	/// rate parameters of the Gamma likelihood on inter-cluster distances.
	/// Smaller values for zeta lead to less repulsion between clusters,
	/// but also sharper cluster boundaries.
	zeta: f64,

	/// Rate hyperparameter for the Gamma prior on the
	/// rate parameters of the Gamma likelihood on inter-cluster distances.
	/// Larger values for gamma lead to greater repulsion between clusters,
	/// but possibly fuzzier cluster boundaries.
	gamma: f64,

	/// Shape hyperparameter for the Gamma prior on the stopping count
	/// in the negative binomial likelihood for each of the cluster sizes.
	/// Greater values for eta leads to greater and more variable cluster sizes.
	eta: f64,

	/// Rate hyperparameter for the Gamma prior on the stopping count r
	/// in the negative binomial likelihood for each of the cluster sizes.
	/// Greater values for sigma lead to greater and more variable cluster
	/// sizes.
	sigma: f64,

	/// Standard deviation of the proposal distribution when sampling
	/// the stopping count r in the MCMC.
	proposalsd_r: f64,

	/// Hyperparameter in the Beta prior for the prior on the success
	/// probability p in the negative binomial likelihood for each of the
	/// cluster sizes. Greater values for u leads to larger and more variable
	/// cluster sizes.
	u: f64,

	/// Hyperparameter in the Beta prior for the prior on the success
	/// probability p in the negative binomial likelihood for each of the
	/// cluster sizes. Greater values for v leads to smaller and less variable
	/// cluster sizes.
	v: f64,
}

impl PriorHyperParams {
	/// Set the hyperparameter delta1.
	pub fn set_delta1(&mut self, delta1: f64) -> Result<&mut Self> {
		if delta1 <= 0.0 {
			return Err(anyhow!("delta1 must be positive"));
		}
		if delta1.is_nan() {
			return Err(anyhow!("delta1 must not be NaN"));
		}
		self.delta1 = delta1;
		Ok(self)
	}

	/// Set the hyperparameter delta1.
	pub fn with_delta1(mut self, delta1: f64) -> Result<Self> {
		self.set_delta1(delta1)?;
		Ok(self)
	}

	/// Set the hyperparameter delta2.
	pub fn set_delta2(&mut self, delta2: f64) -> Result<&mut Self> {
		if delta2 <= 0.0 {
			return Err(anyhow!("delta2 must be positive"));
		}
		if delta2.is_nan() {
			return Err(anyhow!("delta2 must not be NaN"));
		}
		self.delta2 = delta2;
		Ok(self)
	}

	/// Set the hyperparameter delta2.
	pub fn with_delta2(mut self, delta2: f64) -> Result<Self> {
		self.set_delta2(delta2)?;
		Ok(self)
	}

	/// Set the hyperparameter alpha.
	pub fn set_alpha(&mut self, alpha: f64) -> Result<&mut Self> {
		if alpha <= 0.0 {
			return Err(anyhow!("alpha must be positive"));
		}
		if alpha.is_nan() {
			return Err(anyhow!("alpha must not be NaN"));
		}
		self.alpha = alpha;
		Ok(self)
	}

	/// Set the hyperparameter alpha.
	pub fn with_alpha(mut self, alpha: f64) -> Result<Self> {
		self.set_alpha(alpha)?;
		Ok(self)
	}

	/// Set the hyperparameter beta.
	pub fn set_beta(&mut self, beta: f64) -> Result<&mut Self> {
		if beta <= 0.0 {
			return Err(anyhow!("beta must be positive"));
		}
		if beta.is_nan() {
			return Err(anyhow!("beta must not be NaN"));
		}
		self.beta = beta;
		Ok(self)
	}

	/// Set the hyperparameter beta.
	pub fn with_beta(mut self, beta: f64) -> Result<Self> {
		self.set_beta(beta)?;
		Ok(self)
	}

	/// Set the hyperparameter zeta.
	pub fn set_zeta(&mut self, zeta: f64) -> Result<&mut Self> {
		if zeta <= 0.0 {
			return Err(anyhow!("zeta must be positive"));
		}
		if zeta.is_nan() {
			return Err(anyhow!("zeta must not be NaN"));
		}
		self.zeta = zeta;
		Ok(self)
	}

	/// Set the hyperparameter zeta.
	pub fn with_zeta(mut self, zeta: f64) -> Result<Self> {
		self.set_zeta(zeta)?;
		Ok(self)
	}

	/// Set the hyperparameter gamma.
	pub fn set_gamma(&mut self, gamma: f64) -> Result<&mut Self> {
		if gamma <= 0.0 {
			return Err(anyhow!("gamma must be positive"));
		}
		if gamma.is_nan() {
			return Err(anyhow!("gamma must not be NaN"));
		}
		self.gamma = gamma;
		Ok(self)
	}

	/// Set the hyperparameter gamma.
	pub fn with_gamma(mut self, gamma: f64) -> Result<Self> {
		self.set_gamma(gamma)?;
		Ok(self)
	}

	/// Set the hyperparameter eta.
	pub fn set_eta(&mut self, eta: f64) -> Result<&mut Self> {
		if eta <= 0.0 {
			return Err(anyhow!("eta must be positive"));
		}
		if eta.is_nan() {
			return Err(anyhow!("eta must not be NaN"));
		}
		self.eta = eta;
		Ok(self)
	}

	/// Set the hyperparameter eta.
	pub fn with_eta(mut self, eta: f64) -> Result<Self> {
		self.set_eta(eta)?;
		Ok(self)
	}

	/// Set the hyperparameter sigma.
	pub fn set_sigma(&mut self, sigma: f64) -> Result<&mut Self> {
		if sigma <= 0.0 {
			return Err(anyhow!("sigma must be positive"));
		}
		if sigma.is_nan() {
			return Err(anyhow!("sigma must not be NaN"));
		}
		self.sigma = sigma;
		Ok(self)
	}

	/// Set the hyperparameter sigma.
	pub fn with_sigma(mut self, sigma: f64) -> Result<Self> {
		self.set_sigma(sigma)?;
		Ok(self)
	}

	/// Set the hyperparameter u.
	pub fn set_u(&mut self, u: f64) -> Result<&mut Self> {
		if u <= 0.0 {
			return Err(anyhow!("u must be positive"));
		}
		if u.is_nan() {
			return Err(anyhow!("u must not be NaN"));
		}
		self.u = u;
		Ok(self)
	}

	/// Set the hyperparameter u.
	pub fn with_u(mut self, u: f64) -> Result<Self> {
		self.set_u(u)?;
		Ok(self)
	}

	/// Set the hyperparameter v.
	pub fn set_v(&mut self, v: f64) -> Result<&mut Self> {
		if v <= 0.0 {
			return Err(anyhow!("v must be positive"));
		}
		if v.is_nan() {
			return Err(anyhow!("v must not be NaN"));
		}
		self.v = v;
		Ok(self)
	}

	/// Set the hyperparameter v.
	pub fn with_v(mut self, v: f64) -> Result<Self> {
		self.set_v(v)?;
		Ok(self)
	}

	/// Set the hyperparameter proposalsd_r.
	pub fn set_proposalsd_r(&mut self, proposalsd_r: f64) -> Result<&mut Self> {
		if proposalsd_r <= 0.0 {
			return Err(anyhow!("proposalsd_r must be positive"));
		}
		if proposalsd_r.is_nan() {
			return Err(anyhow!("proposalsd_r must not be NaN"));
		}
		self.proposalsd_r = proposalsd_r;
		Ok(self)
	}

	/// Set the hyperparameter proposalsd_r.
	pub fn with_proposalsd_r(mut self, proposalsd_r: f64) -> Result<Self> {
		self.set_proposalsd_r(proposalsd_r)?;
		Ok(self)
	}

	/// Prior distribution on r.
	pub fn r_prior(&self) -> Result<Gamma> {
		Ok(Gamma::new(self.eta, self.sigma)?)
	}

	/// Prior distribution on p.
	pub fn p_prior(&self) -> Result<Beta> {
		Ok(Beta::new(self.u, self.v)?)
	}

	/// Induced prior on cluster sizes.
	pub fn cluster_size_prior(&self) -> Result<ClusterSizePrior> {
		Ok(ClusterSizePrior {
			r_prior: self.r_prior()?,
			p_prior: self.p_prior()?,
		})
	}

	/// Sample from the induced prior on the number of clusters, conditioned on
	/// the number of points.
	pub fn num_clusters_prior(&self, n_pts: NonZeroUsize) -> Result<NumClustersPrior> {
		Ok(NumClustersPrior {
			r_prior: self.r_prior()?,
			p_prior: self.p_prior()?,
			n_pts: n_pts.get(),
		})
	}

	/// Induced prior on within-cluster distances,
	/// marginalising over cluster-specific parameters.
	pub fn intra_cluster_dissimilarity_prior(&self) -> Result<IntraClusterDissimilarityPrior> {
		Ok(IntraClusterDissimilarityPrior {
			delta1: self.delta1,
			lambda_prior: Gamma::new(self.alpha, self.beta)?,
		})
	}

	/// Induced prior on inter-cluster distances,
	/// marginalising over cluster-specific parameters.
	pub fn inter_cluster_dissimilarity_prior(&self) -> Result<InterClusterDissimilarityPrior> {
		Ok(InterClusterDissimilarityPrior {
			delta2: self.delta2,
			theta_prior: Gamma::new(self.zeta, self.gamma)?,
		})
	}
}

impl Default for PriorHyperParams {
	fn default() -> Self {
		Self {
			delta1: 1.0,
			delta2: 1.0,
			alpha: 1.0,
			beta: 1.0,
			zeta: 1.0,
			gamma: 1.0,
			eta: 1.0,
			sigma: 1.0,
			proposalsd_r: 1.0,
			u: 1.0,
			v: 1.0,
		}
	}
}

impl Display for PriorHyperParams {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
		write!(
			f,
			"PriorHyperParams {{\ndelta1: {},\ndelta2: {},\nalpha: {},\nbeta: {},\nzeta: \
			 {},\ngamma: {},\neta: {},\nsigma: {},\nproposalsd_r: {},\nu: {},\nv: {}\n}}",
			self.delta1,
			self.delta2,
			self.alpha,
			self.beta,
			self.zeta,
			self.gamma,
			self.eta,
			self.sigma,
			self.proposalsd_r,
			self.u,
			self.v,
		)
	}
}

impl From<&PriorHyperParams> for HashMap<String, f64> {
	fn from(params: &PriorHyperParams) -> Self {
		let mut map = HashMap::new();
		map.insert("delta1".to_string(), params.delta1());
		map.insert("delta2".to_string(), params.delta2());
		map.insert("alpha".to_string(), params.alpha());
		map.insert("beta".to_string(), params.beta());
		map.insert("zeta".to_string(), params.zeta());
		map.insert("gamma".to_string(), params.gamma());
		map.insert("eta".to_string(), params.eta());
		map.insert("sigma".to_string(), params.sigma());
		map.insert("proposalsd_r".to_string(), params.proposalsd_r());
		map.insert("u".to_string(), params.u());
		map.insert("v".to_string(), params.v());
		map
	}
}

#[cfg(feature = "python-module")]
#[pymethods]
impl PriorHyperParams {
	/// Set the hyperparameter delta1.
	#[setter(delta1)]
	fn py_set_delta1(&mut self, delta1: f64) -> Result<()> {
		self.set_delta1(delta1)?;
		Ok(())
	}

	/// Set the hyperparameter delta2.
	#[setter(delta2)]
	fn py_set_delta2(&mut self, delta2: f64) -> Result<()> {
		self.set_delta2(delta2)?;
		Ok(())
	}

	/// Set the hyperparameter alpha.
	#[setter(alpha)]
	fn py_set_alpha(&mut self, alpha: f64) -> Result<()> {
		self.set_alpha(alpha)?;
		Ok(())
	}

	/// Set the hyperparameter beta.
	#[setter(beta)]
	fn py_set_beta(&mut self, beta: f64) -> Result<()> {
		self.set_beta(beta)?;
		Ok(())
	}

	/// Set the hyperparameter zeta.
	#[setter(zeta)]
	fn py_set_zeta(&mut self, zeta: f64) -> Result<()> {
		self.set_zeta(zeta)?;
		Ok(())
	}

	/// Set the hyperparameter gamma.
	#[setter(gamma)]
	fn py_set_gamma(&mut self, gamma: f64) -> Result<()> {
		self.set_gamma(gamma)?;
		Ok(())
	}

	/// Set the hyperparameter eta.
	#[setter(eta)]
	fn py_set_eta(&mut self, eta: f64) -> Result<()> {
		self.set_eta(eta)?;
		Ok(())
	}

	/// Set the hyperparameter sigma.
	#[setter(sigma)]
	fn py_set_sigma(&mut self, sigma: f64) -> Result<()> {
		self.set_sigma(sigma)?;
		Ok(())
	}

	/// Set the hyperparameter u.
	#[setter(u)]
	fn py_set_u(&mut self, u: f64) -> Result<()> {
		self.set_u(u)?;
		Ok(())
	}

	/// Set the hyperparameter v.
	#[setter(v)]
	fn py_set_v(&mut self, v: f64) -> Result<()> {
		self.set_v(v)?;
		Ok(())
	}

	/// Set the hyperparameter proposalsd_r.
	#[setter(proposalsd_r)]
	fn py_set_proposalsd_r(&mut self, proposalsd_r: f64) -> Result<()> {
		self.set_proposalsd_r(proposalsd_r)?;
		Ok(())
	}

	/// Create a default instance of this struct.
	#[new]
	pub fn py_new() -> Self {
		PriorHyperParams::default()
	}

	/// Sample r from its prior.
	#[pyo3(name = "sample_r")]
	fn py_sample_r(&self, n_samples: usize, rng_seed: Option<u64>) -> Result<Array1Wrapper<f64>> {
		let mut rng = get_rng(rng_seed);
		let samples = self
			.r_prior()?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect::<Array1<_>>();
		Ok(Array1Wrapper(samples))
	}

	/// Sample p from its prior.
	#[pyo3(name = "sample_p")]
	fn py_sample_p(&self, n_samples: usize, rng_seed: Option<u64>) -> Result<Array1Wrapper<f64>> {
		let mut rng = get_rng(rng_seed);
		let samples = self
			.p_prior()?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect::<Array1<_>>();
		Ok(Array1Wrapper(samples))
	}

	/// Sample from the induced prior on cluster sizes.
	#[pyo3(name = "sample_cluster_sizes")]
	fn py_sample_cluster_sizes(
		&mut self,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Result<Array1Wrapper<usize>> {
		let mut rng = get_rng(rng_seed);
		let samples = self
			.cluster_size_prior()?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect::<Array1<_>>();
		Ok(Array1Wrapper(samples))
	}

	/// Sample from the induced prior on the number of clusters, conditioned on
	/// the number of points.
	#[pyo3(name = "sample_n_clusts")]
	fn py_sample_n_clusts(
		&mut self,
		n_pts: NonZeroUsize,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Result<Array1Wrapper<usize>> {
		let mut rng = get_rng(rng_seed);
		let samples = self
			.num_clusters_prior(n_pts)?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect::<Array1<_>>();
		Ok(Array1Wrapper(samples))
	}

	/// Sample from the induced prior on within-cluster distances,
	/// marginalising over cluster-specific parameters.
	#[pyo3(name = "sample_within_cluster_dists")]
	fn py_sample_within_cluster_dists(
		&mut self,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Result<Array1Wrapper<f64>> {
		let mut rng = get_rng(rng_seed);
		let samples = self
			.intra_cluster_dissimilarity_prior()?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect::<Array1<_>>();
		Ok(Array1Wrapper(samples))
	}

	/// Sample from the induced prior on inter-cluster distances, marginalising
	/// over cluster-specific parameters.
	#[pyo3(name = "sample_inter_cluster_dists")]
	fn py_sample_inter_cluster_dists(
		&mut self,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Result<Array1Wrapper<f64>> {
		let mut rng = get_rng(rng_seed);
		let samples = self
			.inter_cluster_dissimilarity_prior()?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect::<Array1<_>>();
		Ok(Array1Wrapper(samples))
	}

	fn __repr__(&self) -> String {
		format!(
			"PriorHyperParams(delta1={}, delta2={}, alpha={}, beta={}, zeta={}, gamma={}, eta={}, \
			 sigma={}, proposalsd_r={}, u={}, v={})",
			self.delta1(),
			self.delta2(),
			self.alpha(),
			self.beta(),
			self.zeta(),
			self.gamma(),
			self.eta(),
			self.sigma(),
			self.proposalsd_r(),
			self.u(),
			self.v(),
		)
	}

	/// Convert the PriorHyperParams object to a dictionary.
	fn as_dict(&self) -> HashMap<String, f64> {
		HashMap::from(self)
	}
}
