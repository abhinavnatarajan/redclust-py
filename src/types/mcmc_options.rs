use std::{
	collections::HashMap,
	fmt::{Debug, Display, Formatter},
	num::NonZeroUsize,
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use pyo3::prelude::*;

const DEFAULT_N_ITER: usize = 1000;
const DEFAULT_N_BURNIN: usize = 200;
const DEFAULT_THINNING: usize = 1;
const DEFAULT_N_GIBBS: usize = 5;
const DEFAULT_N_MH: usize = 1;

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
}

impl MCMCOptions {
	/// Set the number of Gibbs steps
	/// for each iteration of the cluster sampling.
	pub fn set_n_gibbs(&mut self, n_gibbs: usize) -> Result<&mut Self> {
		if n_gibbs + self.n_mh == 0 {
			return Err(anyhow!(
				"At least one Gibbs or Metropolis step must be performed"
			));
		}
		self.n_gibbs = n_gibbs;
		Ok(self)
	}

	/// Set the number of Metropolis-Hastings steps
	/// for each iteration of the cluster sampling.
	pub fn set_n_mh(&mut self, n_mh: usize) -> Result<&mut Self> {
		if self.n_gibbs + n_mh == 0 {
			return Err(anyhow!(
				"At least one Gibbs or Metropolis step must be performed"
			));
		}
		self.n_mh = n_mh;
		Ok(self)
	}
}

impl Default for MCMCOptions {
	fn default() -> Self {
		unsafe {
			MCMCOptions {
				n_iter: DEFAULT_N_ITER,
				n_burnin: DEFAULT_N_BURNIN,
				thinning: NonZeroUsize::new_unchecked(DEFAULT_THINNING),
				n_gibbs: DEFAULT_N_GIBBS,
				n_mh: DEFAULT_N_MH,
			}
		}
	}
}

#[pymethods]
impl MCMCOptions {
	/// Set the number of Gibbs steps
	/// for each iteration of the cluster sampling.
	#[setter(n_gibbs)]
	fn py_set_n_gibbs(this: Bound<'_, Self>, n_gibbs: usize) -> Result<()> {
		this.borrow_mut().set_n_gibbs(n_gibbs)?;
		Ok(())
	}

	/// Set the number of Metropolis-Hastings steps
	/// for each iteration of the cluster sampling.
	#[setter(n_mh)]
	fn py_set_n_mh(this: Bound<'_, Self>, n_mh: usize) -> Result<()> {
		this.borrow_mut().set_n_mh(n_mh)?;
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
			thinning=NonZeroUsize::new_unchecked(DEFAULT_THINNING),
			n_gibbs=DEFAULT_N_GIBBS,
			n_mh=DEFAULT_N_MH
			))]
	pub fn new(
		n_iter: usize,
		n_burnin: usize,
		thinning: NonZeroUsize,
		n_gibbs: usize,
		n_mh: usize,
	) -> Result<Self> {
		let mut res = MCMCOptions {
			n_iter,
			n_burnin,
			thinning,
			n_gibbs: 1,
			n_mh: 1,
		};
		res.set_n_gibbs(n_gibbs)?.set_n_mh(n_mh)?;
		Ok(res)
	}

	fn __repr__(&self) -> String {
		format!(
			"MCMCOptions(n_iter={}, n_burnin={}, thinning={}, n_gibbs={}, n_mh={})",
			self.n_iter, self.n_burnin, self.thinning, self.n_gibbs, self.n_mh
		)
	}

	/// Dictionary representation of the MCMC options.
	fn as_dict(&self) -> HashMap<String, usize> { HashMap::from(self) }
}

impl From<&MCMCOptions> for HashMap<String, usize> {
	fn from(options: &MCMCOptions) -> Self {
		HashMap::from([
			("n_iter".to_string(), options.n_iter),
			("n_burnin".to_string(), options.n_burnin),
			("thinning".to_string(), options.thinning.get()),
			("n_gibbs".to_string(), options.n_gibbs),
			("n_mh".to_string(), options.n_mh),
		])
	}
}

impl Display for MCMCOptions {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"MCMCOptions {{ n_iter: {}, n_burnin: {}, thinning: {}, n_gibbs: {}, n_mh: {} }}",
			self.n_iter, self.n_burnin, self.thinning, self.n_gibbs, self.n_mh
		)
	}
}
