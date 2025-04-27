use std::{
	collections::HashMap,
	fmt::{Debug, Display, Formatter},
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use itertools::Itertools;
use pyo3::prelude::*;

use crate::types::ClusterLabel;

/// Current state of the MCMC sampler.
#[derive(Debug, Clone, Accessors, PartialEq, Default)]
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
	pub(crate) clust_sizes: HashMap<ClusterLabel, usize>,

	/// List of labels of non-empty clusters.
	#[access(get(cp = false))]
	pub(crate) clust_list: Vec<ClusterLabel>,
}

impl MCMCState {
	/// Set the parameter r. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	pub fn set_r(&mut self, r: f64) -> Result<&mut Self> {
		if r <= 0.0 || r.is_infinite() {
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
	/// starting the MCMC sampler.
	pub fn set_clusts(&mut self, clusts: Vec<ClusterLabel>) -> &mut Self {
		// We don't assume contiguity of cluster labels.
		self.clust_labels = clusts;
		self.clust_sizes = self.clust_labels.clone().into_iter().counts();
		self
	}
}

#[pymethods]
impl MCMCState {
	#[new]
	pub fn new(clusts: Vec<ClusterLabel>, r: f64, p: f64) -> Result<Self> {
		let mut res = MCMCState::default();
		res.set_clusts(clusts).set_r(r)?.set_p(p)?;
		Ok(res)
	}

	/// Set the cluster allocations. Useful to initialize a custom state when
	/// starting the MCMC sampler.
	#[setter(clusts)]
	fn py_set_clusts(&mut self, clusts: Vec<ClusterLabel>) -> Result<()> {
		self.set_clusts(clusts);
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
	pub fn n_clusts(&self) -> usize { self.clust_list.len() }
}

impl Display for MCMCState {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> { Debug::fmt(self, f) }
}
