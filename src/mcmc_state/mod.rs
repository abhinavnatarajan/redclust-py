use std::{
	collections::BTreeSet,
	fmt::{Debug, Display, Formatter},
	num::NonZeroUsize,
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use itertools::Itertools;
#[cfg(feature = "python-module")]
use pyo3::prelude::*;

use crate::ClusterLabel;
pub mod sampling;

const DEFAULT_R: f64 = 1.0;
const DEFAULT_P: f64 = 0.5;

/// Current state of the MCMC sampler.
/// Capable of handling cluster assignments with cluster labels
/// in the set [0, ClusterLabel::MAX).
/// ClusterLabel::MAX is used to indicate a data point without an assigned
/// cluster label.
#[derive(Debug, Clone, Accessors, PartialEq)]
#[access(get)]
#[cfg_attr(feature = "python-module", pyclass(get_all, str, eq))]
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

	/// Number of split-merge proposals that were accepted
	/// in the most recent label sampling step.
	#[access(get(skip))]
	pub(crate) splitmerge_accepted: usize,
}

impl MCMCState {
	/// Set the parameter r. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	fn set_r(&mut self, r: f64) -> Result<&mut Self> {
		if !(0.0..f64::INFINITY).contains(&r) {
			return Err(anyhow!("r must be in the interval [0, ∞)]"));
		}
		self.r = r;
		Ok(self)
	}

	/// Set the parameter r. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	fn with_r(mut self, r: f64) -> Result<Self> {
		self.set_r(r)?;
		Ok(self)
	}

	/// Set the parameter p. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	fn set_p(&mut self, p: f64) -> Result<&mut Self> {
		if !(0.0..=1.0).contains(&p) {
			return Err(anyhow!("p must be in the interval (0, 1)"));
		}
		self.p = p;
		Ok(self)
	}

	/// Set the parameter p. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	fn with_p(mut self, p: f64) -> Result<Self> {
		self.set_p(p)?;
		Ok(self)
	}

	/// Set the cluster allocations. Useful to initialize a custom state when
	/// starting the MCMC sampler. The clusters must be in the range [0, n_pts),
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

	/// Updates the cluster assignment for a data point.
	fn update_clust(&mut self, point_idx: usize, new_clust_label: ClusterLabel) {
		if new_clust_label != ClusterLabel::MAX {
			let old_clust_label = self.clust_labels[point_idx];
			let old_clust_size = &mut self.clust_sizes[old_clust_label as usize];
			*old_clust_size -= 1;
			if *old_clust_size == 0 {
				self.clust_list.remove(&old_clust_label);
			}
			self.clust_labels[point_idx] = new_clust_label;
			self.clust_sizes[new_clust_label as usize] += 1;
			self.clust_list.insert(new_clust_label);
		} else {
			self.delete_point(point_idx);
		}
	}

	/// Remove the cluster assignment for a data point.
	/// Does nothing if the point is already unassigned.
	fn delete_point(&mut self, point_idx: usize) {
		let item_clust = &mut self.clust_labels[point_idx];
		if *item_clust != ClusterLabel::MAX {
			let old_clust_label = *item_clust;
			let old_clust_size = &mut self.clust_sizes[old_clust_label as usize];
			*old_clust_size -= 1;
			if *old_clust_size == 0 {
				self.clust_list.remove(&old_clust_label);
			}
			*item_clust = ClusterLabel::MAX;
		}
	}

	/// Get the list of elements with a given cluster label.
	fn items_with_label(&self, clust_label: ClusterLabel) -> Vec<usize> {
		self.clust_labels
			.iter()
			.positions(|&x| x == clust_label)
			.collect_vec()
	}

	/// Find the first empty cluster label.
	fn first_empty_cluster(&self) -> Option<ClusterLabel> {
		self.clust_sizes
			.iter()
			.position(|&x| x == 0)
			.map(|x| x as ClusterLabel)
			.filter(|&x| x != ClusterLabel::MAX)
	}

	/// Set the cluster allocations. Useful to initialize a custom state when
	/// starting the MCMC sampler. The clusters must be in the range [0, n_pts),
	/// but are not required to be contiguous.
	fn with_clusts(mut self, clusts: Vec<ClusterLabel>) -> Result<Self> {
		self.set_clusts(clusts)?;
		Ok(self)
	}

	/// Number of clusters.
	#[inline(always)]
	pub fn num_clusts(&self) -> NonZeroUsize { NonZeroUsize::new(self.clust_list.len()).unwrap() }

	/// Size of a cluster.
	#[inline(always)]
	pub fn clust_size(&self, clust_label: ClusterLabel) -> usize {
		self.clust_sizes[clust_label as usize]
	}

	pub fn new(clusts: Vec<ClusterLabel>, r: Option<f64>, p: Option<f64>) -> Result<Self> {
		let res = MCMCState {
			clust_labels: Vec::new(),
			r: DEFAULT_R,
			p: DEFAULT_P,
			r_accepted: true,
			clust_sizes: Vec::new(),
			clust_list: BTreeSet::new(),
			splitmerge_accepted: 0,
		}; // we don't implement Default for MCMCState because an empty cluster list is invalid
		res.with_clusts(clusts)
			.and_then(|this| this.with_r(r.unwrap_or(DEFAULT_R)))
			.and_then(|this| this.with_p(p.unwrap_or(DEFAULT_P)))
	}
}

impl Display for MCMCState {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> { Debug::fmt(self, f) }
}

#[cfg(feature = "python-module")]
#[pymethods]
impl MCMCState {
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

	/// Set the parameter r. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	#[setter(r)]
	fn py_set_r(&mut self, r: f64) -> Result<()> { self.set_r(r).map(|_| ()) }

	/// Set the parameter p. Useful to initialize a custom state when starting
	/// the MCMC sampler.
	#[setter(p)]
	fn py_set_p(&mut self, p: f64) -> Result<()> { self.set_p(p).map(|_| ()) }

	/// Number of clusters.
	#[getter]
	pub fn py_num_clusts(&self) -> NonZeroUsize { self.num_clusts() }
}
