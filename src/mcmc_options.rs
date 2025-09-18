use std::{
	fmt::{Debug, Display, Formatter},
	num::NonZeroUsize,
};

use anyhow::Result;
#[cfg(feature = "python-module")]
use pyo3::{prelude::*, types::PyDict};

const DEFAULT_NUM_ITER: usize = 1000;
const DEFAULT_NUM_BURNIN: usize = 200;
const DEFAULT_THINNING: NonZeroUsize = NonZeroUsize::new(1).unwrap();
const DEFAULT_NUM_GIBBS_PASSES: usize = 5;
const DEFAULT_NUM_MH_STEPS: usize = 1;
const DEFAULT_NUM_CHAINS: NonZeroUsize = NonZeroUsize::new(1).unwrap();
const DEFAULT_RNG_SEED: Option<u64> = None;

/// Options for the MCMC algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "python-module", pyclass(get_all, set_all, str, eq))]
pub struct MCMCOptions {
	/// Number of iterations.
	pub num_iter: usize,

	/// Number of iterations to discard as burn-in.
	pub num_burnin: usize,

	/// Thinning factor.
	pub thinning: NonZeroUsize,

	/// Number of restricted Gibbs scans to perform during each
	/// Metropolis-Hastings split-merge step when sampling the cluster labels.
	pub num_gibbs_passes: usize,

	/// Number of Metropolis-Hastings steps to perform when sampling cluster
	/// labels.
	pub num_mh_steps: usize,

	/// Number of chains to run concurrently.
	pub num_chains: NonZeroUsize,

	/// Seed for the random number generator.
	pub rng_seed: Option<u64>,
}

impl Default for MCMCOptions {
	fn default() -> Self {
		MCMCOptions {
			num_iter: DEFAULT_NUM_ITER,
			num_burnin: DEFAULT_NUM_BURNIN,
			thinning: DEFAULT_THINNING,
			num_gibbs_passes: DEFAULT_NUM_GIBBS_PASSES,
			num_mh_steps: DEFAULT_NUM_MH_STEPS,
			num_chains: DEFAULT_NUM_CHAINS,
			rng_seed: DEFAULT_RNG_SEED,
		}
	}
}

impl MCMCOptions {
	/// Create a new MCMCOptions instance with the specified parameters.
	pub fn new() -> Self {
		Self::default()
	}

	/// Get the number of samples after burn-in and thinning.
	#[inline(always)]
	pub fn num_samples(&self) -> usize {
		self.num_iter
			.saturating_sub(self.num_burnin)
			.div_ceil(self.thinning.get())
	}
}

impl Display for MCMCOptions {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"MCMCOptions {{ n_iter: {}, n_burnin: {}, thinning: {}, n_gibbs: {}, n_mh: {}, \
			 n_chains: {} }}",
			self.num_iter,
			self.num_burnin,
			self.thinning,
			self.num_gibbs_passes,
			self.num_mh_steps,
			self.num_chains
		)
	}
}

#[cfg(feature = "python-module")]
#[pymethods]
impl MCMCOptions {
	/// Get the number of samples after burn-in and thinning.
	#[getter(num_samples)]
	fn py_num_samples(&self) -> usize {
		self.num_samples()
	}

	/// Create a new MCMCOptions instance with the specified parameters.
	#[new]
	#[pyo3(signature = (
			num_iter=DEFAULT_NUM_ITER,
			num_burnin=DEFAULT_NUM_BURNIN,
			thinning=DEFAULT_THINNING,
			num_gibbs_passes=DEFAULT_NUM_GIBBS_PASSES,
			num_mh_steps=DEFAULT_NUM_MH_STEPS,
			num_chains=DEFAULT_NUM_CHAINS,
			rng_seed=DEFAULT_RNG_SEED,
			))]
	pub fn py_new(
		num_iter: usize,
		num_burnin: usize,
		thinning: NonZeroUsize,
		num_gibbs_passes: usize,
		num_mh_steps: usize,
		num_chains: NonZeroUsize,
		rng_seed: Option<u64>,
	) -> Self {
		Self {
			num_iter,
			num_burnin,
			thinning,
			num_gibbs_passes,
			num_mh_steps,
			num_chains,
			rng_seed,
		}
	}

	fn __repr__(&self) -> String {
		format!(
			"MCMCOptions(n_iter={}, n_burnin={}, thinning={}, n_gibbs={}, n_mh={}, n_chains={})",
			self.num_iter,
			self.num_burnin,
			self.thinning,
			self.num_gibbs_passes,
			self.num_mh_steps,
			self.num_chains
		)
	}

	/// Dictionary representation of the MCMC options.
	fn as_dict(this: Bound<'_, Self>) -> Result<Bound<'_, PyDict>> {
		let slf = this.borrow();
		let dict: Bound<'_, PyDict> = PyDict::new(this.py());
		dict.set_item("num_iter", &slf.num_iter)?;
		dict.set_item("num_burnin", &slf.num_burnin)?;
		dict.set_item("thinning", &slf.thinning)?;
		dict.set_item("num_gibbs_passes", &slf.num_gibbs_passes)?;
		dict.set_item("num_mh_steps", slf.num_mh_steps)?;
		dict.set_item("num_chains", slf.num_chains)?;
		dict.set_item("rng_seed", slf.rng_seed)?;
		Ok(dict)
	}
}
