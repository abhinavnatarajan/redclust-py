use std::{iter::zip, num::NonZeroUsize};

use anyhow::Result;
use itertools::izip;
use ndarray::{Array1, Array2, ArrayView1, s};
use numpy::PyArray1;
use pyo3::prelude::*;
use rand::{SeedableRng, distributions::Distribution, rngs::SmallRng};
use statrs::{
	distribution::{Beta, Continuous, Gamma, LogNormal, NegativeBinomial, Uniform},
	function::{beta::ln_beta, gamma::ln_gamma},
};

use crate::{
	MCMCData,
	mcmc::findall,
	types::{ClusterLabel, MCMCState, PriorHyperParams},
};

pub(crate) unsafe fn row_sum(mat: &Array2<f64>, row: usize, cols: &[usize]) -> f64 {
	cols.iter().map(|&j| mat.uget((row, j))).sum()
}

fn sample_from_ln_probs(p: &ArrayView1<f64>, rng: &mut SmallRng) -> usize {
	let p = p - p.iter().fold(f64::INFINITY, |a, &b| a.min(b));
	let u = unsafe {
		Uniform::new(0.0, 1.0)
			.unwrap_unchecked()
			.sample_iter(rng)
			.take(p.len())
			.collect::<Array1<_>>()
	};
	(-(-(u.ln())).ln() + p)
		.iter()
		.enumerate()
		.fold(
			(0, f64::NEG_INFINITY),
			|(i, x), (j, &y)| if x < y { (j, y) } else { (i, x) },
		)
		.0
}

impl MCMCState {
	/// Sample p from its conditional posterior.
	pub(crate) unsafe fn sample_p_conditional(
		&mut self,
		params: &PriorHyperParams,
		rng: &mut SmallRng,
	) -> &mut Self {
		let n_clusts = self.n_clusts().get() as f64;
		let n_pts = self.clust_labels.len() as f64;
		self.p = Beta::new(n_pts - n_clusts + params.u, self.r * n_clusts + params.v)
			.unwrap_unchecked()
			.sample(rng);
		self
	}

	/// Sample r from its conditional posterior.
	pub(crate) unsafe fn sample_r_conditional(
		&mut self,
		params: &PriorHyperParams,
		rng: &mut SmallRng,
	) -> &mut Self {
		// This is called in a hot loop so we take off the safety goggles.
		let r = self.r;
		let p = self.p;
		let n_clusts = self.n_clusts().get() as f64;
		let eta = params.eta;
		let sigma = params.sigma;
		let proposalsd_r = params.proposalsd_r;
		let clust_list = &self.clust_list;
		let clust_sizes = &self.clust_sizes;
		let proposal_distr = LogNormal::new(r.ln(), proposalsd_r.ln()).unwrap_unchecked();
		let r_candidate = proposal_distr.sample(rng);
		let reverse_distr = LogNormal::new(r_candidate.ln(), proposalsd_r.ln()).unwrap_unchecked();

		// Calculate the transition probability
		let mut log_prob_candidate = (eta - 1.0) * r_candidate.ln()
			+ n_clusts * (r_candidate * (1.0 - p).ln() - ln_gamma(r_candidate))
			- r_candidate * sigma;
		let mut log_prob_current =
			(eta - 1.0) * r.ln() + n_clusts * (r * (1.0 - p).ln() - ln_gamma(r)) - r * sigma;
		clust_list.iter().for_each(|&j| {
			log_prob_candidate +=
				ln_gamma(*clust_sizes.get_unchecked(j as usize) as f64 + r_candidate - 1.0);
			log_prob_current += ln_gamma(*clust_sizes.get_unchecked(j as usize) as f64 + r - 1.0);
		});
		let log_acceptance_prob = reverse_distr.ln_pdf(r) - proposal_distr.ln_pdf(r_candidate)
			+ log_prob_candidate
			- log_prob_current;
		let toss = Uniform::new(0.0, 1.0).unwrap_unchecked().sample(rng).ln();
		self.r_accepted = false;
		if toss < log_acceptance_prob {
			self.r = r_candidate;
			self.r_accepted = true;
		}
		self
	}

	pub(crate) unsafe fn sample_clusters_gibbs(
		&mut self,
		data: &MCMCData,
		params: &PriorHyperParams,
		rng: &mut SmallRng,
	) -> &mut Self {
		let n_pts = self.clust_labels.len();

		// Pre-compute some quantities to speed up subsequent calculations.
		let alpha_beta_ratio = params.alpha * params.beta.ln() - ln_gamma(params.alpha);
		let zeta_gamma_ratio = params.zeta * params.gamma.ln() - ln_gamma(params.zeta);
		let ln_gamma_delta1 = ln_gamma(params.delta1);
		let ln_gamma_delta2 = ln_gamma(params.delta2);
		let ln_p = self.p.ln();
		let ln_1_minus_p = (1.0 - self.p).ln();

		// The list of candidate clusters for each point. Since we assume
		// that the cluster labels are in the range [0, n_pts), we can allocate a size
		// of n_pts in advance, instead of allocating a new vector for each point.
		let mut candidate_clusts = Vec::<ClusterLabel>::with_capacity(n_pts);

		// Pre-allocate space for vectors that represent mappings ClusterLabel -> f64.
		// Use vectors instead of hashmaps for performance and simplicity.
		let mut l3_i = Vec::<f64>::with_capacity(n_pts);
		let mut ln_probs = Vec::<f64>::with_capacity(n_pts);

		// Start
		let clust_labels = &mut self.clust_labels;
		let clust_sizes = &mut self.clust_sizes;
		let clust_list = &mut self.clust_list;
		for point_idx in 0..n_pts {
			// This is a hot loop so we skip bounds checking for clust_labels and
			// clust_sizes. Unfortunately we cannot use iter_mut to avoid this because we
			// need to borrow both variables immutably as well.
			let current_label = clust_labels.get_unchecked_mut(point_idx);

			// We cannot move the ith point to a different cluster if it is in a singleton
			// cluster and we are already at the minimum number of clusters
			if clust_list.len() == params.n_clusts_min.get()
				&& *clust_sizes.get_unchecked(*current_label as usize) == 1
			{
				continue;
			}

			// Remove the ith point from the clustering
			*clust_sizes.get_unchecked_mut(*current_label as usize) -= 1;
			if *clust_sizes.get_unchecked_mut(*current_label as usize) == 0 {
				clust_list.remove(current_label);
			}

			*current_label = n_pts as ClusterLabel; // Treat this cluster label as "hidden"
			// Number of clusters after removing the ith point
			let k_i = clust_list.len() as f64;

			// Update the candidate list
			candidate_clusts.clear(); // clear + extend avoids re-allocation
			candidate_clusts.extend(clust_list.iter());

			// Begin
			ln_probs.resize(clust_list.len(), 0.0);
			for (&k, l3_ik, ln_prob_k) in izip!(
				candidate_clusts.iter(),
				l3_i.iter_mut(),
				ln_probs.iter_mut()
			) {
				let elems_clust_k = findall(clust_labels, |&x| x == k);
				let sz_k = *clust_sizes.get_unchecked(k as usize) as f64;
				let alpha_ik = params.alpha + params.delta1 * sz_k;
				let beta_ik = params.beta + row_sum(&data.diss_mat.0, point_idx, &elems_clust_k);
				let zeta_ik = params.zeta + params.delta2 * sz_k;
				let gamma_ik = params.gamma + row_sum(&data.diss_mat.0, point_idx, &elems_clust_k);
				let sum_ln_diss_ik = row_sum(&data.ln_diss_mat.0, point_idx, &elems_clust_k);

				let l1_ik = ln_gamma(alpha_ik) - alpha_ik * beta_ik.ln()
					+ alpha_beta_ratio
					+ (params.delta1 - 1.0) * sum_ln_diss_ik
					- sz_k * ln_gamma_delta1;
				*ln_prob_k =
					(sz_k + 1.0).ln() + ln_p + (sz_k - 1.0 + self.r).ln() - sz_k.ln() + l1_ik;

				*l3_ik = ln_gamma(zeta_ik) - zeta_ik * gamma_ik.ln()
					+ zeta_gamma_ratio
					+ (params.delta2 - 1.0) * sum_ln_diss_ik
					- sz_k * ln_gamma_delta2;
			}
			let sum_l3_ik: f64 = l3_i.iter().sum();
			for (l3_ik, ln_prob_k) in izip!(l3_i.iter(), ln_probs.iter_mut()) {
				*ln_prob_k += sum_l3_ik - *l3_ik;
			}

			// If we are under the maximum number of clusters, we can consider inserting a
			// new cluster
			if clust_list.len() < params.n_clusts_max.get() {
				candidate_clusts
					.push(clust_sizes.iter().position(|&x| x == 0).unwrap() as ClusterLabel);
				ln_probs.push((k_i + 1.0).ln() + self.r * ln_1_minus_p + sum_l3_ik);
			}

			let new_clust = *candidate_clusts
				.get_unchecked(sample_from_ln_probs(&ArrayView1::from(&ln_probs), rng));
			*clust_labels.get_unchecked_mut(point_idx) = new_clust;
			*clust_sizes.get_unchecked_mut(new_clust as usize) += 1;
			clust_list.insert(new_clust);
		}
		self
	}
}

impl PriorHyperParams {
	/// Sample r from its prior.
	pub fn sample_r(&self, n_samples: usize, rng: &mut SmallRng) -> Array1<f64> {
		Gamma::new(self.eta, self.sigma)
			.unwrap()
			.sample_iter(rng)
			.take(n_samples)
			.collect()
	}

	/// Sample p from its prior.
	pub fn sample_p(&self, n_samples: usize, rng: &mut SmallRng) -> Array1<f64> {
		Beta::new(self.u, self.v)
			.unwrap()
			.sample_iter(rng)
			.take(n_samples)
			.collect()
	}

	/// Sample from the induced prior on cluster sizes.
	pub fn sample_cluster_sizes(&self, n_samples: usize, rng: &mut SmallRng) -> Array1<usize> {
		let r_samples = self.sample_r(n_samples, rng);
		let p_samples = self.sample_p(n_samples, rng);
		zip(r_samples, p_samples)
			.map(|(r, p)| NegativeBinomial::new(r, p).unwrap().sample(rng) as usize + 1)
			.collect()
	}

	/// Sample from the induced prior on the number of clusters, conditioned on
	/// the number of points.
	pub fn sample_n_clusts(
		&self,
		n_pts: NonZeroUsize,
		n_samples: NonZeroUsize,
		rng: &mut SmallRng,
	) -> Array1<usize> {
		let n_pts = n_pts.get();
		let n_samples = n_samples.get();
		let mut samples = Array1::<usize>::zeros(n_samples);
		let k = Array1::linspace(1.0, n_pts as f64, n_pts);
		let mut logprobs = Array1::<f64>::zeros(n_pts);
		let n = n_pts as f64;
		for i in 0..n_samples {
			let r = self.sample_r(1, rng)[0];
			let p = self.sample_p(1, rng)[0];
			logprobs.slice_mut(s![0..n_pts - 1]).assign(
				&(r * &k * (1.0 - p).ln() + (n - &k) * p.ln()
					- (n - &k).ln() - zip(r * &k, n - &k)
					.map(|(x, y)| ln_beta(x, y))
					.collect::<Array1<f64>>()),
			);
			logprobs[n_pts - 1] = r * n * (1.0 - p).ln();
			samples[i] = sample_from_ln_probs(&logprobs.view(), rng);
		}
		samples
	}

	/// Sample from the induced prior on within-cluster distances,
	/// marginalising over cluster-specific parameters.
	pub fn sample_within_cluster_dists(&self, n_samples: usize, rng: &mut SmallRng) -> Array1<f64> {
		let lambda_prior = Gamma::new(self.alpha, self.beta).unwrap();
		let lambda = lambda_prior
			.sample_iter(&mut *rng)
			.take(n_samples)
			.collect::<Vec<f64>>();
		let mut samples = Array1::from_vec(Vec::with_capacity(n_samples));
		(0..n_samples).for_each(|i| {
			let dist_prior = Gamma::new(self.delta1, lambda[i]).unwrap();
			samples[i] = dist_prior.sample(rng);
		});
		samples
	}

	/// Sample from the induced prior on inter-cluster distances, marginalising
	/// over cluster-specific parameters.
	pub fn sample_inter_cluster_dists(&self, n_samples: usize, rng: &mut SmallRng) -> Array1<f64> {
		let theta_prior = Gamma::new(self.zeta, self.gamma).unwrap();
		let theta = theta_prior
			.sample_iter(&mut *rng)
			.take(n_samples)
			.collect::<Vec<f64>>();
		let mut samples = Array1::from_vec(Vec::with_capacity(n_samples));
		(0..n_samples).for_each(|i| {
			let dist_prior = Gamma::new(self.delta2, theta[i]).unwrap();
			samples[i] = dist_prior.sample(rng);
		});
		samples
	}
}

#[pymethods]
impl PriorHyperParams {
	/// Sample r from its prior.
	#[pyo3(name = "sample_r")]
	fn py_sample_r(this: Bound<'_, Self>, n_samples: usize) -> Result<Bound<'_, PyArray1<f64>>> {
		let mut rng = SmallRng::from_entropy();
		let samples = this.borrow().sample_r(n_samples, &mut rng);
		let arr = PyArray1::from_owned_array(this.py(), samples);
		Ok(arr)
	}

	/// Sample p from its prior.
	#[pyo3(name = "sample_p")]
	fn py_sample_p(this: Bound<'_, Self>, n_samples: usize) -> Bound<'_, PyArray1<f64>> {
		let mut rng = SmallRng::from_entropy();
		let samples = this.borrow().sample_p(n_samples, &mut rng);
		let arr = PyArray1::from_owned_array(this.py(), samples);
		arr
	}

	/// Sample from the induced prior on cluster sizes.
	#[pyo3(name = "sample_cluster_sizes")]
	fn py_sample_cluster_sizes(
		this: Bound<'_, Self>,
		n_samples: usize,
	) -> Bound<'_, PyArray1<usize>> {
		let mut rng = SmallRng::from_entropy();
		let samples = this.borrow().sample_cluster_sizes(n_samples, &mut rng);
		let arr = PyArray1::from_owned_array(this.py(), samples);
		arr
	}

	/// Sample from the induced prior on the number of clusters, conditioned on
	/// the number of points.
	#[pyo3(name = "sample_n_clusts")]
	fn py_sample_n_clusts(
		this: Bound<'_, Self>,
		n_pts: NonZeroUsize,
		n_samples: NonZeroUsize,
	) -> Bound<'_, PyArray1<usize>> {
		let mut rng = SmallRng::from_entropy();
		let samples = this.borrow().sample_n_clusts(n_pts, n_samples, &mut rng);
		let arr = PyArray1::from_owned_array(this.py(), samples);
		arr
	}

	/// Sample from the induced prior on within-cluster distances,
	/// marginalising over cluster-specific parameters.
	#[pyo3(name = "sample_within_cluster_dists")]
	fn py_sample_within_cluster_dists(
		this: Bound<'_, Self>,
		n_samples: usize,
	) -> Bound<'_, PyArray1<f64>> {
		let mut rng = SmallRng::from_entropy();
		let samples = this
			.borrow()
			.sample_within_cluster_dists(n_samples, &mut rng);
		let arr = PyArray1::from_owned_array(this.py(), samples);
		arr
	}

	/// Sample from the induced prior on inter-cluster distances, marginalising
	/// over cluster-specific parameters.
	#[pyo3(name = "sample_inter_cluster_dists")]
	fn py_sample_inter_cluster_dists(
		this: Bound<'_, Self>,
		n_samples: usize,
	) -> Bound<'_, PyArray1<f64>> {
		let mut rng = SmallRng::from_entropy();
		let samples = this
			.borrow()
			.sample_inter_cluster_dists(n_samples, &mut rng);
		let arr = PyArray1::from_owned_array(this.py(), samples);
		arr
	}
}
