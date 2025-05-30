use std::{
	collections::BTreeSet,
	fmt::{Debug, Display, Formatter},
	num::NonZeroUsize,
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use itertools::Itertools;
use pyo3::prelude::*;

use crate::types::ClusterLabel;
const DEFAULT_R: f64 = 1.0;
const DEFAULT_P: f64 = 0.5;

/// Current state of the MCMC sampler.
#[derive(Debug, Clone, Accessors, PartialEq)]
#[access(get)]
#[pyclass(get_all, str, eq)]
pub struct MCMCState {
	/// Current cluster allocation.
	#[access(get(cp = false))]
	pub(crate) clust_labels: Vec<ClusterLabel>,

	/// Parameter r.
	pub(crate) r: f64,

	/// Whether the last proposal for the parameter r was accepted.
	#[access(get(skip))]
	pub(crate) r_accepted: bool,

	/// Parameter p.
	pub(crate) p: f64,

	/// Cluster sizes.
	#[access(get(cp = false))]
	pub(crate) clust_sizes: Vec<usize>,

	/// List of labels of non-empty clusters.
	#[access(get(cp = false))]
	pub(crate) clust_list: BTreeSet<ClusterLabel>,
}

impl MCMCState {
	/// Set the parameter r. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	pub fn set_r(&mut self, r: f64) -> Result<&mut Self> {
		if !(0.0..f64::INFINITY).contains(&r) {
			return Err(anyhow!("r must be in the interval [0, ∞)]"));
		}
		self.r = r;
		Ok(self)
	}

	/// Set the parameter p. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	pub fn set_p(&mut self, p: f64) -> Result<&mut Self> {
		if !(0.0..=1.0).contains(&p) {
			return Err(anyhow!("p must be in the interval (0, 1)"));
		}
		self.p = p;
		Ok(self)
	}

	/// Set the cluster allocations. Useful to initialize a custom state when
	/// starting the MCMC sampler. The clusters must be in the range [0, n_pts),
	/// but are not required to be contiguous.
	pub fn set_clusts(&mut self, clusts: Vec<ClusterLabel>) -> Result<&mut Self> {
		let n_pts = clusts.len();
		if n_pts == 0 {
			return Err(anyhow!("Cluster allocation cannot be empty"));
		}
		if n_pts as u64 >= ClusterLabel::MAX as u64 {
			return Err(anyhow!(
				"Too many points; this library only supports up to {} points",
				ClusterLabel::MAX
			));
		}
		if *clusts.iter().max().unwrap() as usize >= n_pts {
			return Err(anyhow!("Cluster labels must be in the range [0, n) where n is the number of points"));
		}
		self.clust_labels = clusts;
		self.clust_sizes = vec![0; self.clust_labels.len()];
		self.clust_labels
			.iter()
			.counts()
			.into_iter()
			.for_each(|(k, v)| {
				self.clust_sizes[*k as usize] = v;
				self.clust_list.insert(*k);
			});
		Ok(self)
	}
}

#[pymethods]
impl MCMCState {
	#[new]
	#[pyo3(signature = (clusts, r = DEFAULT_R, p = DEFAULT_P))]
	pub fn new(clusts: Vec<ClusterLabel>, r: f64, p: f64) -> Result<Self> {
		let mut res = MCMCState {
			clust_labels: Vec::new(),
			r: DEFAULT_R,
			p: DEFAULT_P,
			r_accepted: true,
			clust_sizes: Vec::new(),
			clust_list: BTreeSet::new(),
		}; // we don't implement Default for MCMCState because an empty cluster list is invalid
		res.set_clusts(clusts)?.set_r(r)?.set_p(p)?;
		Ok(res)
	}

	/// Set the cluster allocations. Useful to initialize a custom state when
	/// starting the MCMC sampler.
	#[setter(clusts)]
	fn py_set_clusts(&mut self, clusts: Vec<ClusterLabel>) -> Result<()> {
		self.set_clusts(clusts)?;
		Ok(())
	}

	/// Set the parameter r. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	#[setter(r)]
	fn py_set_r(&mut self, r: f64) -> Result<()> {
		self.set_r(r)?;
		Ok(())
	}

	/// Set the parameter p. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	#[setter(p)]
	fn py_set_p(&mut self, p: f64) -> Result<()> {
		self.set_p(p)?;
		Ok(())
	}

	/// Number of clusters.
	#[inline(always)]
	pub fn n_clusts(&self) -> NonZeroUsize {
		unsafe { NonZeroUsize::new_unchecked(self.clust_list.len()) }
	}
}

impl Display for MCMCState {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> { Debug::fmt(self, f) }
}
