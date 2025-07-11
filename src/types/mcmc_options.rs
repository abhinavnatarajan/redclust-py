use std::{
	fmt::{Debug, Display, Formatter},
	num::NonZeroUsize,
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use pyo3::{prelude::*, types::PyDict};

const DEFAULT_N_ITER: usize = 1000;
const DEFAULT_N_BURNIN: usize = 200;
const DEFAULT_THINNING: NonZeroUsize = NonZeroUsize::new(1).unwrap();
const DEFAULT_N_GIBBS: usize = 5;
const DEFAULT_N_MH: usize = 1;
const DEFAULT_N_CHAINS: NonZeroUsize = NonZeroUsize::new(1).unwrap();
const DEFAULT_RNG_SEED: Option<u64> = None;

/// Options for the MCMC algorithm.
#[derive(Debug, Clone, Copy, Accessors, PartialEq, Eq)]
#[pyclass(get_all, str, eq)]
pub struct MCMCOptions {
	/// Number of iterations.
	#[pyo3(set)]
	pub n_iter: usize,

	/// Number of iterations to discard as burn-in.
	#[pyo3(set)]
	pub n_burnin: usize,

	/// Thinning factor.
	#[pyo3(set)]
	pub thinning: NonZeroUsize,

	/// Number of Gibbs steps per iteration.
	#[access(get)]
	pub(crate) n_gibbs: usize,

	/// Number of Metropolis-Hastings steps per iteration.
	#[access(get)]
	pub(crate) n_mh: usize,

	/// Number of chains to run concurrently.
	#[pyo3(set)]
	pub n_chains: NonZeroUsize,

	/// Seed for the random number generator.
	#[pyo3(set)]
	pub rng_seed: Option<u64>,
}

impl MCMCOptions {
	/// Set the number of Gibbs and Metropolis steps
	/// for each iteration of the cluster sampling.
	pub fn set_n_gibbs_mh(&mut self, n_gibbs: usize, n_mh: usize) -> Result<&mut Self> {
		if n_gibbs + n_mh == 0 {
			return Err(anyhow!(
				"At least one Gibbs or Metropolis step must be performed"
			));
		}
		self.n_gibbs = n_gibbs;
		self.n_mh = n_mh;
		Ok(self)
	}
}

impl Default for MCMCOptions {
	fn default() -> Self {
		MCMCOptions {
			n_iter: DEFAULT_N_ITER,
			n_burnin: DEFAULT_N_BURNIN,
			thinning: DEFAULT_THINNING,
			n_gibbs: DEFAULT_N_GIBBS,
			n_mh: DEFAULT_N_MH,
			n_chains: DEFAULT_N_CHAINS,
			rng_seed: DEFAULT_RNG_SEED,
		}
	}
}

#[pymethods]
impl MCMCOptions {
	/// Set the number of Gibbs steps
	/// for each iteration of the cluster sampling.
	#[pyo3(name = "set_n_gibbs_mh")]
	fn py_set_n_gibbs_mh(this: Bound<'_, Self>, n_gibbs: usize, n_mh: usize) -> Result<()> {
		this.borrow_mut().set_n_gibbs_mh(n_gibbs, n_mh)?;
		Ok(())
	}

	/// Get the number of samples after burn-in and thinning.
	#[getter]
	#[inline(always)]
	pub fn n_samples(&self) -> usize {
		self.n_iter
			.saturating_sub(self.n_burnin)
			.div_ceil(self.thinning.get())
	}

	/// Create a new MCMCOptions instance with the specified parameters.
	#[new]
	#[pyo3(signature = (
			n_iter=DEFAULT_N_ITER,
			n_burnin=DEFAULT_N_BURNIN,
			thinning=DEFAULT_THINNING,
			n_gibbs=DEFAULT_N_GIBBS,
			n_mh=DEFAULT_N_MH,
			n_chains=NonZeroUsize::new(1).unwrap(),
			rng_seed=None
			))]
	pub fn new(
		n_iter: usize,
		n_burnin: usize,
		thinning: NonZeroUsize,
		n_gibbs: usize,
		n_mh: usize,
		n_chains: NonZeroUsize,
		rng_seed: Option<u64>,
	) -> Result<Self> {
		let mut res = MCMCOptions {
			n_iter,
			n_burnin,
			thinning,
			n_gibbs: 1,
			n_mh: 1,
			n_chains,
			rng_seed,
		};
		res.set_n_gibbs_mh(n_gibbs, n_mh)?;
		Ok(res)
	}

	fn __repr__(&self) -> String {
		format!(
			"MCMCOptions(n_iter={}, n_burnin={}, thinning={}, n_gibbs={}, n_mh={}, n_chains={})",
			self.n_iter, self.n_burnin, self.thinning, self.n_gibbs, self.n_mh, self.n_chains
		)
	}

	/// Dictionary representation of the MCMC options.
	fn as_dict(this: Bound<'_, Self>) -> Result<Bound<'_, PyDict>> {
		let slf = this.borrow();
		let dict: Bound<'_, PyDict> = PyDict::new(this.py());
		dict.set_item("n_iter", &slf.n_iter)?;
		dict.set_item("n_burnin", &slf.n_burnin)?;
		dict.set_item("thinning", &slf.thinning)?;
		dict.set_item("n_gibbs", &slf.n_gibbs)?;
		dict.set_item("n_mh", slf.n_mh)?;
		dict.set_item("n_chains", slf.n_chains)?;
		dict.set_item("rng_seed", slf.rng_seed)?;
		Ok(dict)
	}
}

impl Display for MCMCOptions {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"MCMCOptions {{ n_iter: {}, n_burnin: {}, thinning: {}, n_gibbs: {}, n_mh: {}, \
			 n_chains: {} }}",
			self.n_iter, self.n_burnin, self.thinning, self.n_gibbs, self.n_mh, self.n_chains
		)
	}
}
