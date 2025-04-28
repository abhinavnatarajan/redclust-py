use std::{
	fmt::{Debug, Display, Formatter},
	num::NonZeroUsize,
	ops::RangeInclusive,
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use pyo3::{prelude::*, types::PyDict};

const DEFAULT_DELTA1: f64 = 1.0;
const DEFAULT_DELTA2: f64 = 1.0;
const DEFAULT_ALPHA: f64 = 1.0;
const DEFAULT_BETA: f64 = 1.0;
const DEFAULT_ZETA: f64 = 1.0;
const DEFAULT_GAMMA: f64 = 1.0;
const DEFAULT_ETA: f64 = 1.0;
const DEFAULT_SIGMA: f64 = 1.0;
const DEFAULT_PROPOSALSD_R: f64 = 1.0;
const DEFAULT_U: f64 = 1.0;
const DEFAULT_V: f64 = 1.0;
const DEFAULT_N_CLUSTS_INIT: NonZeroUsize = NonZeroUsize::new(1).unwrap();
const DEFAULT_REPULSION: bool = true;
const DEFAULT_N_CLUSTS_MIN: NonZeroUsize = NonZeroUsize::new(1).unwrap();
const DEFAULT_N_CLUSTS_MAX: NonZeroUsize = NonZeroUsize::new(usize::MAX).unwrap();

/// Prior hyper-parameters for the Bayesian distance clustering algorithm.
#[derive(Debug, Clone, Copy, Accessors, PartialEq)]
#[access(get, defaults(get(cp)))]
#[pyclass(str, eq)]
//TODO: validation for setters?
pub struct PriorHyperParams {
	/// Shape hyperparameter for the Gamma likelihood on intra-cluster
	/// dissimilarities. Smaller values lead to greater within-cluster
	/// cohesion.
	#[pyo3(get)]
	pub(crate) delta1: f64,

	/// Shape hyperparameter for the Gamma likelihood on inter-cluster
	/// dissimilarities. Larger values lead to greater inter-cluster repulsion.
	#[pyo3(get)]
	pub(crate) delta2: f64,

	/// Shape hyperparameter for the Gamma prior on
	/// the rate parameters of the Gamma likelihood for intra-cluster
	/// dissimilarities. on intra-cluster distances in the k-th cluster.
	/// Smaller values for alpha lead to more cohesion and less variability
	/// within clusters.
	#[pyo3(get)]
	pub(crate) alpha: f64,

	/// Rate hyperparameter for the Gamma prior on the
	/// rate parameters of the Gamma likelihood on intra-cluster distances.
	/// Larger values for beta lead to less cohesion and greater variability
	/// within clusters.
	#[pyo3(get)]
	pub(crate) beta: f64,

	/// Rate hyperparameter for the Gamma prior on the
	/// rate parameters of the Gamma likelihood on inter-cluster distances.
	/// Smaller values for zeta lead to less repulsion between clusters,
	/// but also sharper cluster boundaries.
	#[pyo3(get)]
	pub(crate) zeta: f64,

	/// Rate hyperparameter for the Gamma prior on the
	/// rate parameters of the Gamma likelihood on inter-cluster distances.
	/// Larger values for gamma lead to greater repulsion between clusters,
	/// but possibly fuzzier cluster boundaries.
	#[pyo3(get)]
	pub(crate) gamma: f64,

	/// Shape hyperparameter for the Gamma prior on the stopping count
	/// in the negative binomial likelihood for each of the cluster sizes.
	/// Greater values for eta leads to greater and more variable cluster sizes.
	#[pyo3(get)]
	pub(crate) eta: f64,

	/// Rate hyperparameter for the Gamma prior on the stopping count r
	/// in the negative binomial likelihood for each of the cluster sizes.
	/// Greater values for sigma lead to greater and more variable cluster
	/// sizes.
	#[pyo3(get)]
	pub(crate) sigma: f64,

	/// Standard deviation of the proposal distribution when sampling
	/// the stopping count r in the MCMC.
	#[pyo3(get)]
	pub(crate) proposalsd_r: f64,

	/// Hyperparameter in the Beta prior for the prior on the success
	/// probability p in the negative binomial likelihood for each of the
	/// cluster sizes. Greater values for u leads to larger and more variable
	/// cluster sizes.
	#[pyo3(get)]
	pub(crate) u: f64,

	/// Hyperparameter in the Beta prior for the prior on the success
	/// probability p in the negative binomial likelihood for each of the
	/// cluster sizes. Greater values for v leads to smaller and less variable
	/// cluster sizes.
	#[pyo3(get)]
	pub(crate) v: f64,

	/// Initial number of clusters to begin the MCMC.
	/// The actual clustering will be determined by a k-medoids algorithm.
	#[pyo3(get)]
	pub(crate) n_clusts_init: NonZeroUsize,

	/// Whether to use repulsive terms in the likelihood function when
	/// clustering.
	#[pyo3(get, set)]
	#[access(set)]
	pub(crate) repulsion: bool,

	/// Minimum number of clusters to allow in the clustering.
	#[pyo3(get)]
	pub(crate) n_clusts_min: NonZeroUsize,

	/// Maximum number of clusters to allow in the clustering.
	#[pyo3(get)]
	pub(crate) n_clusts_max: NonZeroUsize,
}

impl PriorHyperParams {
	/// Set the hyperparameter delta1.
	pub fn set_delta1(&mut self, delta1: f64) -> Result<&mut Self> {
		if delta1 <= 0.0 {
			return Err(anyhow!("delta1 must be positive"));
		}
		self.delta1 = delta1;
		Ok(self)
	}

	/// Set the hyperparameter delta2.
	pub fn set_delta2(&mut self, delta2: f64) -> Result<&mut Self> {
		if delta2 <= 0.0 {
			return Err(anyhow!("delta2 must be positive"));
		}
		self.delta2 = delta2;
		Ok(self)
	}

	/// Set the hyperparameter alpha.
	pub fn set_alpha(&mut self, alpha: f64) -> Result<&mut Self> {
		if alpha <= 0.0 {
			return Err(anyhow!("alpha must be positive"));
		}
		self.alpha = alpha;
		Ok(self)
	}

	/// Set the hyperparameter beta.
	pub fn set_beta(&mut self, beta: f64) -> Result<&mut Self> {
		if beta <= 0.0 {
			return Err(anyhow!("beta must be positive"));
		}
		self.beta = beta;
		Ok(self)
	}

	/// Set the hyperparameter zeta.
	pub fn set_zeta(&mut self, zeta: f64) -> Result<&mut Self> {
		if zeta <= 0.0 {
			return Err(anyhow!("zeta must be positive"));
		}
		self.zeta = zeta;
		Ok(self)
	}

	/// Set the hyperparameter gamma.
	pub fn set_gamma(&mut self, gamma: f64) -> Result<&mut Self> {
		if gamma <= 0.0 {
			return Err(anyhow!("gamma must be positive"));
		}
		self.gamma = gamma;
		Ok(self)
	}

	/// Set the hyperparameter eta.
	pub fn set_eta(&mut self, eta: f64) -> Result<&mut Self> {
		if eta <= 0.0 {
			return Err(anyhow!("eta must be positive"));
		}
		self.eta = eta;
		Ok(self)
	}

	/// Set the hyperparameter sigma.
	pub fn set_sigma(&mut self, sigma: f64) -> Result<&mut Self> {
		if sigma <= 0.0 {
			return Err(anyhow!("sigma must be positive"));
		}
		self.sigma = sigma;
		Ok(self)
	}

	/// Set the hyperparameter u.
	pub fn set_u(&mut self, u: f64) -> Result<&mut Self> {
		if u <= 0.0 {
			return Err(anyhow!("u must be positive"));
		}
		self.u = u;
		Ok(self)
	}

	/// Set the hyperparameter v.
	pub fn set_v(&mut self, v: f64) -> Result<&mut Self> {
		if v <= 0.0 {
			return Err(anyhow!("v must be positive"));
		}
		self.v = v;
		Ok(self)
	}

	/// Set the hyperparameter proposalsd_r.
	pub fn set_proposalsd_r(&mut self, proposalsd_r: f64) -> Result<&mut Self> {
		if proposalsd_r <= 0.0 {
			return Err(anyhow!("proposalsd_r must be positive"));
		}
		self.proposalsd_r = proposalsd_r;
		Ok(self)
	}

	/// Set the hyperparameter n_clusts_init.
	pub fn set_n_clusts_init(&mut self, n_clusts_init: NonZeroUsize) -> Result<&mut Self> {
		if !(self.n_clusts_min..=self.n_clusts_max).contains(&n_clusts_init) {
			return Err(anyhow!(
				"n_clusts_init must be in the range [{}, {}]",
				self.n_clusts_min,
				self.n_clusts_max
			));
		}
		self.n_clusts_init = n_clusts_init;
		Ok(self)
	}

	/// Set the range of allowed values for the number of clusters.
	pub fn set_n_clusts_range(
		&mut self,
		n_clusts_range: RangeInclusive<NonZeroUsize>,
	) -> Result<&mut Self> {
		if n_clusts_range.is_empty() {
			return Err(anyhow!("Range must be non-empty"));
		}
		self.n_clusts_min = *n_clusts_range.start();
		self.n_clusts_max = *n_clusts_range.end();
		Ok(self)
	}
}

#[pymethods]
impl PriorHyperParams {
	/// Set the hyperparameter delta1.
	#[setter(delta1)]
	fn py_set_delta1(this: Bound<'_, Self>, delta1: f64) -> Result<()> {
		this.borrow_mut().set_delta1(delta1)?;
		Ok(())
	}

	/// Set the hyperparameter delta2.
	#[setter(delta2)]
	fn py_set_delta2(this: Bound<'_, Self>, delta2: f64) -> Result<()> {
		this.borrow_mut().set_delta2(delta2)?;
		Ok(())
	}

	/// Set the hyperparameter alpha.
	#[setter(alpha)]
	fn py_set_alpha(this: Bound<'_, Self>, alpha: f64) -> Result<()> {
		this.borrow_mut().set_alpha(alpha)?;
		Ok(())
	}

	/// Set the hyperparameter beta.
	#[setter(beta)]
	fn py_set_beta(this: Bound<'_, Self>, beta: f64) -> Result<()> {
		this.borrow_mut().set_beta(beta)?;
		Ok(())
	}

	/// Set the hyperparameter zeta.
	#[setter(zeta)]
	fn py_set_zeta(this: Bound<'_, Self>, zeta: f64) -> Result<()> {
		this.borrow_mut().set_zeta(zeta)?;
		Ok(())
	}

	/// Set the hyperparameter gamma.
	#[setter(gamma)]
	fn py_set_gamma(this: Bound<'_, Self>, gamma: f64) -> Result<()> {
		this.borrow_mut().set_gamma(gamma)?;
		Ok(())
	}

	/// Set the hyperparameter eta.
	#[setter(eta)]
	fn py_set_eta(this: Bound<'_, Self>, eta: f64) -> Result<()> {
		this.borrow_mut().set_eta(eta)?;
		Ok(())
	}

	/// Set the hyperparameter sigma.
	#[setter(sigma)]
	fn py_set_sigma(this: Bound<'_, Self>, sigma: f64) -> Result<()> {
		this.borrow_mut().set_sigma(sigma)?;
		Ok(())
	}

	/// Set the hyperparameter u.
	#[setter(u)]
	fn py_set_u(this: Bound<'_, Self>, u: f64) -> Result<()> {
		this.borrow_mut().set_u(u)?;
		Ok(())
	}

	/// Set the hyperparameter v.
	#[setter(v)]
	fn py_set_v(this: Bound<'_, Self>, v: f64) -> Result<()> {
		this.borrow_mut().set_v(v)?;
		Ok(())
	}

	/// Set the hyperparameter proposalsd_r.
	#[setter(proposalsd_r)]
	fn py_set_proposalsd_r(this: Bound<'_, Self>, proposalsd_r: f64) -> Result<()> {
		this.borrow_mut().set_proposalsd_r(proposalsd_r)?;
		Ok(())
	}

	/// Set the hyperparameter n_clusts_init.
	#[setter(n_clusts_init)]
	fn py_set_n_clusts_init(this: Bound<'_, Self>, n_clusts_init: NonZeroUsize) -> Result<()> {
		this.borrow_mut().set_n_clusts_init(n_clusts_init)?;
		Ok(())
	}

	/// Get the range of allowed values for the number of clusters.
	#[getter(n_clusts_range)]
	fn py_get_n_clusts_range(&self) -> (NonZeroUsize, NonZeroUsize) {
		(self.n_clusts_min, self.n_clusts_max)
	}

	/// Set the range of allowed values for the number of clusters.
	#[setter(n_clusts_range)]
	fn py_set_n_clusts_range(
		this: Bound<'_, Self>,
		n_clusts_range: (NonZeroUsize, NonZeroUsize),
	) -> Result<()> {
		this.borrow_mut()
			.set_n_clusts_range(n_clusts_range.0..=n_clusts_range.1)?;
		Ok(())
	}

	/// Create a default instance of this struct.
	#[new]
	pub fn new() -> Self { PriorHyperParams::default() }

	fn __repr__(&self) -> String {
		format!(
			"PriorHyperParams(delta1={}, delta2={}, alpha={}, beta={}, zeta={}, gamma={}, eta={}, \
			 sigma={}, proposalsd_r={}, u={}, v={}, n_clusts_init={}, repulsion={}, \
			 n_clusts_min={}, n_clusts_max={})",
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
			self.n_clusts_init,
			self.repulsion,
			self.n_clusts_min,
			self.n_clusts_max,
		)
	}

	/// Convert the PriorHyperParams object to a dictionary.
	fn as_dict(this: Bound<'_, Self>) -> Result<Bound<'_, PyDict>> {
		let slf = this.borrow();
		let dict: Bound<'_, PyDict> = PyDict::new(this.py());
		dict.set_item("delta1", slf.delta1)?;
		dict.set_item("delta2", slf.delta2)?;
		dict.set_item("alpha", slf.alpha)?;
		dict.set_item("beta", slf.beta)?;
		dict.set_item("zeta", slf.zeta)?;
		dict.set_item("gamma", slf.gamma)?;
		dict.set_item("eta", slf.eta)?;
		dict.set_item("sigma", slf.sigma)?;
		dict.set_item("proposalsd_r", slf.proposalsd_r)?;
		dict.set_item("u", slf.u)?;
		dict.set_item("v", slf.v)?;
		dict.set_item("n_clusts_init", slf.n_clusts_init)?;
		dict.set_item("repulsion", slf.repulsion)?;
		dict.set_item("n_clusts_min", slf.n_clusts_min)?;
		dict.set_item("n_clusts_max", slf.n_clusts_max)?;
		Ok(dict)
	}
}

impl Default for PriorHyperParams {
	fn default() -> Self {
		PriorHyperParams {
			delta1: DEFAULT_DELTA1,
			delta2: DEFAULT_DELTA2,
			alpha: DEFAULT_ALPHA,
			beta: DEFAULT_BETA,
			zeta: DEFAULT_ZETA,
			gamma: DEFAULT_GAMMA,
			eta: DEFAULT_ETA,
			sigma: DEFAULT_SIGMA,
			proposalsd_r: DEFAULT_PROPOSALSD_R,
			u: DEFAULT_U,
			v: DEFAULT_V,
			n_clusts_init: DEFAULT_N_CLUSTS_INIT,
			repulsion: DEFAULT_REPULSION,
			n_clusts_min: DEFAULT_N_CLUSTS_MIN,
			n_clusts_max: DEFAULT_N_CLUSTS_MAX,
		}
	}
}

impl Display for PriorHyperParams {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
		write!(
			f,
			"PriorHyperParams {{\ndelta1: {},\ndelta2: {},\nalpha: {},\nbeta: {},\nzeta: \
			 {},\ngamma: {},\neta: {},\nsigma: {},\nproposalsd_r: {},\nu: {},\nv: \
			 {},\nn_clusts_init: {},\nrepulsion: {},\nn_clusts_min: {},\nn_clusts_max: {}\n}}",
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
			self.n_clusts_init,
			self.repulsion,
			self.n_clusts_min,
			self.n_clusts_max
		)
	}
}
