use std::iter::zip;

use anyhow::Result;
use ndarray::{Array1, s};
use numpy::PyArray1;
use pyo3::prelude::*;
use rand::{SeedableRng, distributions::Distribution, rngs::SmallRng};
use statrs::{
	distribution::{Beta, Continuous, Gamma, LogNormal, NegativeBinomial, Uniform},
	function::{beta::ln_beta, gamma::ln_gamma},
};

use crate::types::{MCMCState, PriorHyperParams};

fn sample_from_logprobs(p: &Array1<f64>, rng: &mut SmallRng) -> usize {
	let p = p - p.iter().fold(f64::INFINITY, |a, &b| a.min(b));
	let u = Uniform::new(0.0, 1.0)
		.unwrap()
		.sample_iter(rng)
		.take(p.len())
		.collect::<Array1<_>>();
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
	pub(crate) fn sample_p(&mut self, params: &PriorHyperParams, rng: &mut SmallRng) -> &mut Self {
		let n_clusts = self.n_clusts() as f64;
		let n_pts = self.clust_labels.len() as f64;
		self.p = Beta::new(n_pts - n_clusts + params.u, self.r * n_clusts + params.v)
			.unwrap()
			.sample(rng);
		self
	}

	/// Sample r from its conditional posterior.
	pub(crate) fn sample_r(&mut self, params: &PriorHyperParams, rng: &mut SmallRng) -> &mut Self {
		let r = self.r;
		let p = self.p;
		let n_clusts = self.n_clusts() as f64;
		let eta = params.eta;
		let sigma = params.sigma;
		let proposalsd_r = params.proposalsd_r;
		let clust_list = &self.clust_list;
		let clust_sizes = &self.clust_sizes;
		let proposal_distr = LogNormal::new(r.ln(), proposalsd_r.ln()).unwrap();
		let r_candidate = proposal_distr.sample(rng);
		let reverse_distr = LogNormal::new(r_candidate.ln(), proposalsd_r.ln()).unwrap();

		// Calculate the transition probability
		let mut log_prob_candidate = (eta - 1.0) * r_candidate.ln()
			+ n_clusts * (r_candidate * (1.0 - p).ln() - ln_gamma(r_candidate))
			- r_candidate * sigma;
		let mut log_prob_current =
			(eta - 1.0) * r.ln() + n_clusts * (r * (1.0 - p).ln() - ln_gamma(r)) - r * sigma;
		clust_list.iter().for_each(|j| {
			log_prob_candidate += ln_gamma(clust_sizes[j] as f64 + r_candidate - 1.0);
			log_prob_current += ln_gamma(clust_sizes[j] as f64 + r - 1.0);
		});
		let log_acceptance_prob = reverse_distr.ln_pdf(r) - proposal_distr.ln_pdf(r_candidate)
			+ log_prob_candidate
			- log_prob_current;
		let toss = Uniform::new(0.0, 1.0).unwrap().sample(rng).ln();
		self.r_accepted = false;
		if toss < log_acceptance_prob {
			self.r = r_candidate;
			self.r_accepted = true;
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
		n_pts: usize,
		n_samples: usize,
		rng: &mut SmallRng,
	) -> Array1<usize> {
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
			samples[i] = sample_from_logprobs(&logprobs, rng);
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
		n_pts: usize,
		n_samples: usize,
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
