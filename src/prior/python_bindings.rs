use std::num::NonZeroUsize;

use anyhow::Result;
use itertools::Itertools;
use ndarray::Array1;
use numpy::PyArray1;
use pyo3::{prelude::*, types::PyDict};
use rand::distributions::Distribution;

use super::PriorHyperParams;
use crate::*;

#[cfg(feature = "python-module")]
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

	/// Set the range of allowed values for the number of clusters.
	#[pyo3(name = "set_range_num_clusts")]
	fn py_set_range_num_clusts(
		this: Bound<'_, Self>,
		min_num_clusts: NonZeroUsize,
		max_num_clusts: NonZeroUsize,
	) -> Result<()> {
		this.borrow_mut()
			.set_range_num_clusts(min_num_clusts..=max_num_clusts)?;
		Ok(())
	}

	/// Create a default instance of this struct.
	#[new]
	pub fn new() -> Self { PriorHyperParams::default() }

	/// Sample r from its prior.
	#[pyo3(name = "sample_r")]
	fn py_sample_r(
		this: Bound<'_, Self>,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Result<Bound<'_, PyArray1<f64>>> {
		let mut rng = get_rng(rng_seed);
		let samples = this
			.borrow()
			.r_prior()?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect_vec();
		let arr = PyArray1::from_owned_array(this.py(), Array1::from_vec(samples));
		Ok(arr)
	}

	/// Sample p from its prior.
	#[pyo3(name = "sample_p")]
	fn py_sample_p(
		this: Bound<'_, Self>,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Result<Bound<'_, PyArray1<f64>>> {
		let mut rng = get_rng(rng_seed);
		let samples = this
			.borrow()
			.p_prior()?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect_vec();
		let arr = PyArray1::from_owned_array(this.py(), Array1::from_vec(samples));
		Ok(arr)
	}

	/// Sample from the induced prior on cluster sizes.
	#[pyo3(name = "sample_cluster_sizes")]
	fn py_sample_cluster_sizes(
		this: Bound<'_, Self>,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Result<Bound<'_, PyArray1<usize>>> {
		let mut rng = get_rng(rng_seed);
		let samples = this
			.borrow()
			.cluster_size_prior()?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect_vec();
		let arr = PyArray1::from_owned_array(this.py(), Array1::from_vec(samples));
		Ok(arr)
	}

	/// Sample from the induced prior on the number of clusters, conditioned on
	/// the number of points.
	#[pyo3(name = "sample_n_clusts")]
	fn py_sample_n_clusts(
		this: Bound<'_, Self>,
		n_pts: NonZeroUsize,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Result<Bound<'_, PyArray1<usize>>> {
		let mut rng = get_rng(rng_seed);
		let samples = this
			.borrow()
			.num_clusters_prior(n_pts)?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect_vec();
		let arr = PyArray1::from_owned_array(this.py(), Array1::from_vec(samples));
		Ok(arr)
	}

	/// Sample from the induced prior on within-cluster distances,
	/// marginalising over cluster-specific parameters.
	#[pyo3(name = "sample_within_cluster_dists")]
	fn py_sample_within_cluster_dists(
		this: Bound<'_, Self>,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Result<Bound<'_, PyArray1<f64>>> {
		let mut rng = get_rng(rng_seed);
		let samples = this
			.borrow()
			.intra_cluster_dissimilarity_prior()?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect_vec();
		let arr = PyArray1::from_owned_array(this.py(), Array1::from_vec(samples));
		Ok(arr)
	}

	/// Sample from the induced prior on inter-cluster distances, marginalising
	/// over cluster-specific parameters.
	#[pyo3(name = "sample_inter_cluster_dists")]
	fn py_sample_inter_cluster_dists(
		this: Bound<'_, Self>,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Result<Bound<'_, PyArray1<f64>>> {
		let mut rng = get_rng(rng_seed);
		let samples = this
			.borrow()
			.inter_cluster_dissimilarity_prior()?
			.sample_iter(&mut rng)
			.take(n_samples)
			.collect_vec();
		let arr = PyArray1::from_owned_array(this.py(), Array1::from_vec(samples));
		Ok(arr)
	}

	fn __repr__(&self) -> String {
		format!(
			"PriorHyperParams(delta1={}, delta2={}, alpha={}, beta={}, zeta={}, gamma={}, eta={}, \
			 sigma={}, proposalsd_r={}, u={}, v={}, repulsion={}, min_num_clusts={}, \
			 max_num_clusts={})",
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
			self.repulsion(),
			self.min_num_clusts(),
			self.max_num_clusts(),
		)
	}

	/// Convert the PriorHyperParams object to a dictionary.
	fn as_dict(this: Bound<'_, Self>) -> Result<Bound<'_, PyDict>> {
		let py = this.py();
		let this = this.borrow();
		let dict: Bound<'_, PyDict> = PyDict::new(py);
		dict.set_item("delta1", this.delta1())?;
		dict.set_item("delta2", this.delta2())?;
		dict.set_item("alpha", this.alpha())?;
		dict.set_item("beta", this.beta())?;
		dict.set_item("zeta", this.zeta())?;
		dict.set_item("gamma", this.gamma())?;
		dict.set_item("eta", this.eta())?;
		dict.set_item("sigma", this.sigma())?;
		dict.set_item("proposalsd_r", this.proposalsd_r())?;
		dict.set_item("u", this.u())?;
		dict.set_item("v", this.v())?;
		dict.set_item("repulsion", this.repulsion())?;
		dict.set_item("min_num_clusts", this.min_num_clusts())?;
		dict.set_item("max_num_clusts", this.max_num_clusts())?;
		Ok(dict)
	}
}
