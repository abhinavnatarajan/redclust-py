use std::{
	collections::BTreeSet,
	fmt::{Debug, Display, Formatter},
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use itertools::Itertools;
#[cfg(feature = "python-module")]
use pyo3::prelude::*;

use crate::ClusterLabel;
pub(super) mod helper;

const DEFAULT_R: f64 = 1.0;
const DEFAULT_P: f64 = 0.5;

/// Current state of the MCMC sampler.
/// Capable of handling cluster assignments with cluster labels
/// in the set $[0,$ `ClusterLabel::MAX`$)$.
/// `ClusterLabel::MAX` is used to indicate a data point without an assigned
/// cluster label.
#[derive(Debug, Clone, Accessors, PartialEq)]
#[access(get, defaults(get(cp)))]
#[cfg_attr(feature = "python-module", pyclass(get_all, str, eq))]
pub struct State {
	/// Current cluster allocation.
	#[access(get(cp = false))]
	clust_labels: Vec<ClusterLabel>,
	/// Current value of $r$.
	r: f64,
	/// Current value of $p$.
	p: f64,
}

impl State {
	/// Set the parameter $r$. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	fn set_r(&mut self, r: f64) -> Result<&mut Self> {
		if !(0.0..f64::INFINITY).contains(&r) {
			return Err(anyhow!("r must be in the interval [0, ∞)]"));
		}
		self.r = r;
		Ok(self)
	}

	/// Set the parameter $r$. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	fn with_r(mut self, r: f64) -> Result<Self> {
		self.set_r(r)?;
		Ok(self)
	}

	/// Set the parameter $p$. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	fn set_p(&mut self, p: f64) -> Result<&mut Self> {
		if !(0.0..=1.0).contains(&p) {
			return Err(anyhow!("p must be in the interval (0, 1)"));
		}
		self.p = p;
		Ok(self)
	}

	/// Set the parameter $p$. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	fn with_p(mut self, p: f64) -> Result<Self> {
		self.set_p(p)?;
		Ok(self)
	}

	/// Set the cluster allocations. Useful to initialize a custom state when
	/// starting the MCMC sampler. The clusters must be in the range $[0,$ `n_pts`$)$,
	/// but are not required to be contiguous.
	fn set_clusts(&mut self, clusts: Vec<ClusterLabel>) -> Result<&mut Self> {
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
			return Err(anyhow!(
				"Cluster labels must be in the range [0, n) where n is the number of points"
			));
		}
		self.clust_labels = clusts;
		Ok(self)
	}

	/// Set the cluster allocations. Useful to initialize a custom state when
	/// starting the MCMC sampler. The clusters must be in the range $[0,$ `n_pts`$)$,
	/// but are not required to be contiguous.
	fn with_clusts(mut self, clusts: Vec<ClusterLabel>) -> Result<Self> {
		self.set_clusts(clusts)?;
		Ok(self)
	}

	/// Get the list of elements with a given cluster label.
	pub fn items_with_label(&self, clust_label: ClusterLabel) -> Vec<usize> {
		self.clust_labels
			.iter()
			.positions(|&x| x == clust_label)
			.collect_vec()
	}

	/// Set of cluster labels in the clustering.
	pub fn clust_set(&self) -> BTreeSet<ClusterLabel> {
		BTreeSet::<ClusterLabel>::from_iter(self.clust_labels.iter().copied())
	}

	/// Initialize a new state.
	pub fn new(clusts: Vec<ClusterLabel>, r: Option<f64>, p: Option<f64>) -> Result<Self> {
		let res = State {
			clust_labels: Vec::new(),
			r: DEFAULT_R,
			p: DEFAULT_P,
		}; // we don't implement Default for MCMCState because an empty cluster list is invalid
		res.with_clusts(clusts)
			.and_then(|this| this.with_r(r.unwrap_or(DEFAULT_R)))
			.and_then(|this| this.with_p(p.unwrap_or(DEFAULT_P)))
	}

}

impl Display for State {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> { Debug::fmt(self, f) }
}

#[cfg(feature = "python-module")]
#[pymethods]
impl State {
	/// Initialize a new state.
	#[new]
	#[pyo3(signature = (clusts, r = DEFAULT_R, p = DEFAULT_P))]
	pub fn py_new(clusts: Vec<ClusterLabel>, r: Option<f64>, p: Option<f64>) -> Result<Self> {
		Self::new(clusts, r, p)
	}

	/// Set the cluster allocations. Useful to initialize a custom state when
	/// starting the MCMC sampler.
	#[setter(clusts)]
	fn py_set_clusts(&mut self, clusts: Vec<ClusterLabel>) -> Result<()> {
		self.set_clusts(clusts).map(|_| ())
	}

	/// Set the parameter $r$. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	#[setter(r)]
	fn py_set_r(&mut self, r: f64) -> Result<()> { self.set_r(r).map(|_| ()) }

	/// Set the parameter $p$. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	#[setter(p)]
	fn py_set_p(&mut self, p: f64) -> Result<()> { self.set_p(p).map(|_| ()) }
}
