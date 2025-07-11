use std::{
	fmt::{Debug, Display, Formatter},
	iter::{repeat_n, zip},
	num::NonZeroUsize,
	ops::RangeInclusive,
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use itertools::Itertools;
use ndarray::{Array1, Array2, s};
use numpy::PyArray1;
use pyo3::{prelude::*, types::PyDict};
use rand::{Rng, distributions::Distribution};
use statrs::{
	distribution::{Beta, Gamma, NegativeBinomial},
	function::beta::ln_beta,
};

use super::{ClusterLabel, MCMCData, MCMCOptions, MCMCState};
use crate::utils::{fit_beta_mle, fit_gamma_mle, get_rng, knee_pos, pmf, sample_from_ln_probs};

const DEFAULT_DELTA1: f64 = 1.0;
const DEFAULT_DELTA2: f64 = 1.0;
const DEFAULT_ALPHA: f64 = 1.0;
const DEFAULT_BETA: f64 = 1.0;
const DEFAULT_ZETA: f64 = 1.0;
const DEFAULT_GAMMA: f64 = 1.0;
const DEFAULT_ETA: f64 = 1.0;
const DEFAULT_SIGMA: f64 = 1.0;
const DEFAULT_PROPOSALSD_R: f64 = 1.0;
const DEFAULT_U: f64 = 1.0;
const DEFAULT_V: f64 = 1.0;
const DEFAULT_REPULSION: bool = true;
const DEFAULT_N_CLUSTS_RANGE: RangeInclusive<NonZeroUsize> =
	NonZeroUsize::new(1).unwrap()..=NonZeroUsize::new(usize::MAX).unwrap();

const NONZERO_THOUSAND: NonZeroUsize = NonZeroUsize::new(1000).unwrap();
const DEFAULT_MCMC_ITERS_RP_FITPRIOR: NonZeroUsize = NONZERO_THOUSAND;
const DEFAULT_MLE_ITERS_FITPRIOR: NonZeroUsize = NONZERO_THOUSAND;
const DEFAULT_MCMC_ITERS_N_CLUSTS_FITPRIOR: NonZeroUsize = NONZERO_THOUSAND;

/// Prior hyper-parameters for the Bayesian distance clustering algorithm.
#[derive(Debug, Clone, Accessors, PartialEq)]
#[access(get, defaults(get(cp)))]
#[pyclass(str, eq)]
pub struct PriorHyperParams {
	/// Shape hyperparameter for the Gamma likelihood on intra-cluster
	/// dissimilarities. Smaller values lead to greater within-cluster
	/// cohesion.
	#[pyo3(get)]
	delta1: f64,

	/// Shape hyperparameter for the Gamma likelihood on inter-cluster
	/// dissimilarities. Larger values lead to greater inter-cluster repulsion.
	#[pyo3(get)]
	delta2: f64,

	/// Shape hyperparameter for the Gamma prior on
	/// the rate parameters of the Gamma likelihood for intra-cluster
	/// dissimilarities. on intra-cluster distances in the k-th cluster.
	/// Smaller values for alpha lead to more cohesion and less variability
	/// within clusters.
	#[pyo3(get)]
	alpha: f64,

	/// Rate hyperparameter for the Gamma prior on the
	/// rate parameters of the Gamma likelihood on intra-cluster distances.
	/// Larger values for beta lead to less cohesion and greater variability
	/// within clusters.
	#[pyo3(get)]
	beta: f64,

	/// Rate hyperparameter for the Gamma prior on the
	/// rate parameters of the Gamma likelihood on inter-cluster distances.
	/// Smaller values for zeta lead to less repulsion between clusters,
	/// but also sharper cluster boundaries.
	#[pyo3(get)]
	zeta: f64,

	/// Rate hyperparameter for the Gamma prior on the
	/// rate parameters of the Gamma likelihood on inter-cluster distances.
	/// Larger values for gamma lead to greater repulsion between clusters,
	/// but possibly fuzzier cluster boundaries.
	#[pyo3(get)]
	gamma: f64,

	/// Shape hyperparameter for the Gamma prior on the stopping count
	/// in the negative binomial likelihood for each of the cluster sizes.
	/// Greater values for eta leads to greater and more variable cluster sizes.
	#[pyo3(get)]
	eta: f64,

	/// Rate hyperparameter for the Gamma prior on the stopping count r
	/// in the negative binomial likelihood for each of the cluster sizes.
	/// Greater values for sigma lead to greater and more variable cluster
	/// sizes.
	#[pyo3(get)]
	sigma: f64,

	/// Standard deviation of the proposal distribution when sampling
	/// the stopping count r in the MCMC.
	#[pyo3(get)]
	proposalsd_r: f64,

	/// Hyperparameter in the Beta prior for the prior on the success
	/// probability p in the negative binomial likelihood for each of the
	/// cluster sizes. Greater values for u leads to larger and more variable
	/// cluster sizes.
	#[pyo3(get)]
	u: f64,

	/// Hyperparameter in the Beta prior for the prior on the success
	/// probability p in the negative binomial likelihood for each of the
	/// cluster sizes. Greater values for v leads to smaller and less variable
	/// cluster sizes.
	#[pyo3(get)]
	v: f64,

	/// Whether to use repulsive terms in the likelihood function when
	/// clustering.
	#[pyo3(get, set)]
	#[access(set)]
	repulsion: bool,

	/// Range for number of clusters to allow in the clustering.
	#[access(get(cp = false))]
	n_clusts_range: RangeInclusive<NonZeroUsize>,
}

impl PriorHyperParams {
	/// Set the hyperparameter delta1.
	pub fn set_delta1(&mut self, delta1: f64) -> Result<&mut Self> {
		if delta1 <= 0.0 {
			return Err(anyhow!("delta1 must be positive"));
		}
		if delta1.is_nan() {
			return Err(anyhow!("delta1 must not be NaN"));
		}
		self.delta1 = delta1;
		Ok(self)
	}

	/// Set the hyperparameter delta2.
	pub fn set_delta2(&mut self, delta2: f64) -> Result<&mut Self> {
		if delta2 <= 0.0 {
			return Err(anyhow!("delta2 must be positive"));
		}
		if delta2.is_nan() {
			return Err(anyhow!("delta2 must not be NaN"));
		}
		self.delta2 = delta2;
		Ok(self)
	}

	/// Set the hyperparameter alpha.
	pub fn set_alpha(&mut self, alpha: f64) -> Result<&mut Self> {
		if alpha <= 0.0 {
			return Err(anyhow!("alpha must be positive"));
		}
		if alpha.is_nan() {
			return Err(anyhow!("alpha must not be NaN"));
		}
		self.alpha = alpha;
		Ok(self)
	}

	/// Set the hyperparameter beta.
	pub fn set_beta(&mut self, beta: f64) -> Result<&mut Self> {
		if beta <= 0.0 {
			return Err(anyhow!("beta must be positive"));
		}
		if beta.is_nan() {
			return Err(anyhow!("beta must not be NaN"));
		}
		self.beta = beta;
		Ok(self)
	}

	/// Set the hyperparameter zeta.
	pub fn set_zeta(&mut self, zeta: f64) -> Result<&mut Self> {
		if zeta <= 0.0 {
			return Err(anyhow!("zeta must be positive"));
		}
		if zeta.is_nan() {
			return Err(anyhow!("zeta must not be NaN"));
		}
		self.zeta = zeta;
		Ok(self)
	}

	/// Set the hyperparameter gamma.
	pub fn set_gamma(&mut self, gamma: f64) -> Result<&mut Self> {
		if gamma <= 0.0 {
			return Err(anyhow!("gamma must be positive"));
		}
		if gamma.is_nan() {
			return Err(anyhow!("gamma must not be NaN"));
		}
		self.gamma = gamma;
		Ok(self)
	}

	/// Set the hyperparameter eta.
	pub fn set_eta(&mut self, eta: f64) -> Result<&mut Self> {
		if eta <= 0.0 {
			return Err(anyhow!("eta must be positive"));
		}
		if eta.is_nan() {
			return Err(anyhow!("eta must not be NaN"));
		}
		self.eta = eta;
		Ok(self)
	}

	/// Set the hyperparameter sigma.
	pub fn set_sigma(&mut self, sigma: f64) -> Result<&mut Self> {
		if sigma <= 0.0 {
			return Err(anyhow!("sigma must be positive"));
		}
		if sigma.is_nan() {
			return Err(anyhow!("sigma must not be NaN"));
		}
		self.sigma = sigma;
		Ok(self)
	}

	/// Set the hyperparameter u.
	pub fn set_u(&mut self, u: f64) -> Result<&mut Self> {
		if u <= 0.0 {
			return Err(anyhow!("u must be positive"));
		}
		if u.is_nan() {
			return Err(anyhow!("u must not be NaN"));
		}
		self.u = u;
		Ok(self)
	}

	/// Set the hyperparameter v.
	pub fn set_v(&mut self, v: f64) -> Result<&mut Self> {
		if v <= 0.0 {
			return Err(anyhow!("v must be positive"));
		}
		if v.is_nan() {
			return Err(anyhow!("v must not be NaN"));
		}
		self.v = v;
		Ok(self)
	}

	/// Set the hyperparameter proposalsd_r.
	pub fn set_proposalsd_r(&mut self, proposalsd_r: f64) -> Result<&mut Self> {
		if proposalsd_r <= 0.0 {
			return Err(anyhow!("proposalsd_r must be positive"));
		}
		if proposalsd_r.is_nan() {
			return Err(anyhow!("proposalsd_r must not be NaN"));
		}
		self.proposalsd_r = proposalsd_r;
		Ok(self)
	}

	/// Set the range of allowed values for the number of clusters.
	pub fn set_n_clusts_range(
		&mut self,
		n_clusts_range: RangeInclusive<NonZeroUsize>,
	) -> Result<&mut Self> {
		if n_clusts_range.is_empty() {
			return Err(anyhow!("Range must be non-empty"));
		}
		self.n_clusts_range = n_clusts_range;
		Ok(self)
	}

	/// Sample r from its prior.
	pub fn sample_r<R: Rng>(&self, n_samples: usize, rng: &mut R) -> Vec<f64> {
		Gamma::new(self.eta, self.sigma)
			.unwrap()
			.sample_iter(rng)
			.take(n_samples)
			.collect()
	}

	/// Sample p from its prior.
	pub fn sample_p<R: Rng>(&self, n_samples: usize, rng: &mut R) -> Vec<f64> {
		Beta::new(self.u, self.v)
			.unwrap()
			.sample_iter(rng)
			.take(n_samples)
			.collect()
	}

	/// Sample from the induced prior on cluster sizes.
	pub fn sample_cluster_sizes<R: Rng>(&self, n_samples: usize, rng: &mut R) -> Vec<usize> {
		let r_samples = self.sample_r(n_samples, rng);
		let p_samples = self.sample_p(n_samples, rng);
		zip(r_samples, p_samples)
			.map(|(r, p)| NegativeBinomial::new(r, p).unwrap().sample(rng) as usize + 1)
			.collect()
	}

	/// Sample from the induced prior on the number of clusters, conditioned on
	/// the number of points.
	pub fn sample_n_clusts<R: Rng>(
		&self,
		n_pts: NonZeroUsize,
		n_samples: NonZeroUsize,
		rng: &mut R,
	) -> Result<Vec<usize>> {
		let n_pts = n_pts.get();
		let n_samples = n_samples.get();
		let mut samples = vec![0; n_samples];
		let k = Array1::from_iter((1..n_pts).map(|x| x as f64));
		let mut logprobs = Array1::<f64>::zeros(n_pts);
		let n = n_pts as f64;
		for k_sample in samples.iter_mut() {
			let r = self.sample_r(1, rng)[0];
			let p = self.sample_p(1, rng)[0];
			logprobs.slice_mut(s![0..n_pts - 1]).assign(
				&(r * &k * (1.0 - p).ln() + (n - &k) * p.ln()
					- (n - &k).ln() - zip(r * &k, n - &k)
					.map(|(x, y)| ln_beta(x, y))
					.collect::<Array1<f64>>()),
			);
			logprobs[n_pts - 1] = r * n * (1.0 - p).ln();
			*k_sample = sample_from_ln_probs(&logprobs.view(), rng)? + 1;
		}
		Ok(samples)
	}

	/// Sample from the induced prior on within-cluster distances,
	/// marginalising over cluster-specific parameters.
	pub fn sample_within_cluster_dists<R: Rng>(
		&self,
		n_samples: usize,
		rng: &mut R,
	) -> Array1<f64> {
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
	pub fn sample_inter_cluster_dists<R: Rng>(&self, n_samples: usize, rng: &mut R) -> Array1<f64> {
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

	/// Fit a prior hyperparameters from the data, and return an
	/// initialized state for the sampler.
	/// This will modify all prior hyperparameters except for repulsion and the
	/// minimum/maximum number of clusters allowed.
	pub fn fit_from_data<R: Rng>(
		&mut self,
		data: &MCMCData,
		mcmc_iters_rp: NonZeroUsize,
		mle_iters: NonZeroUsize,
		mcmc_iters_n_clusts: NonZeroUsize,
		// method: PriorClusteringMethod,
		rng: &mut R,
	) -> Result<MCMCState> {
		let n_pts = data.n_pts().get();
		let n_clusts_min = self.n_clusts_range.start().get();
		let n_clusts_max = self.n_clusts_range.end().get();
		if !(n_clusts_min..=n_clusts_max).contains(&n_pts) {
			return Err(anyhow!(
				"Number of clusters must be between {} and {}.",
				n_clusts_min,
				n_clusts_max
			));
		}

		// Set of within-cluster distances
		let mut a = Vec::<f64>::new();
		// Set of inter-cluster distances
		let mut b = Vec::<f64>::new();
		// Number of within-cluster distances for each clustering
		let mut sz_a = Vec::<usize>::with_capacity(n_clusts_max + 1 - n_clusts_min);
		let mut sz_b = Vec::<usize>::with_capacity(n_clusts_max + 1 - n_clusts_min);
		let mut losses = Vec::<f64>::with_capacity(n_clusts_max + 1 - n_clusts_min);

		// For each possible number of clusters, create a corresponding clustering
		// and record the within-cluster and inter-cluster distances, and the clustering
		// loss.
		for i in n_clusts_min..=n_clusts_max {
			let mut medoids = kmedoids::random_initialization(n_pts, i, rng);
			let (loss, clust_labels, ..) =
				kmedoids::fasterpam::<Array2<f64>, f64, f64>(data.diss_mat(), &mut medoids, 1000);
			losses.push(loss);
			let clust_labels = clust_labels
				.into_iter()
				.map(|x| x as ClusterLabel)
				.collect_vec();
			let within_cluster_dists = data.within_cluster_dists(&clust_labels).unwrap();
			sz_a.push(within_cluster_dists.len());
			a.extend(within_cluster_dists);
			let inter_cluster_dists = data.inter_cluster_dists(&clust_labels).unwrap();
			sz_b.push(inter_cluster_dists.len());
			b.extend(inter_cluster_dists);
		}
		// Use the elbow method to find the optimal number of clusters.
		let n_clusts_init = knee_pos(&losses).unwrap() + n_clusts_min;

		// Get a clustering with the optimal number of clusters for the initial state.
		let clust_labels = {
			let mut medoids = kmedoids::random_initialization(n_pts, n_clusts_init, rng);
			let (_, clust_labels, ..): (f64, _, _, _) =
				kmedoids::rand_fasterpam(data.diss_mat(), &mut medoids, 1000, rng);
			clust_labels
		}
		.into_iter()
		.map(|x| x as ClusterLabel)
		.collect_vec();

		// Given the optimal clustering, sample r and p from the conditional posterior
		// using default values for their hyperpriors.
		let (r_samples, p_samples) = pre_sample_rp(
			&clust_labels,
			self,
			&MCMCOptions {
				n_iter: mcmc_iters_rp.get(),
				n_burnin: 0,
				..MCMCOptions::default()
			},
			rng,
		)?;
		// Use the posterior samples to find maximum likelihood estimates for the
		// hyperprior parameters.
		let proposalsd_r = r_samples.std(0.0);
		let (eta, sigma) = fit_gamma_mle(
			&r_samples,
			&Array1::<f64>::ones(r_samples.len()),
			mle_iters.get(),
			None,
		)
		.map_err(|e| anyhow!("Error fitting eta and sigma: {}", e))?;
		let (u, v) = fit_beta_mle(&p_samples, mle_iters.get(), None)
			.map_err(|e| anyhow!("Error fitting u and v: {}", e))?;

		(|| -> Result<()> {
			self.set_proposalsd_r(proposalsd_r)?
				.set_eta(eta)?
				.set_sigma(sigma)?
				.set_u(u)?
				.set_v(v)?;
			Ok(())
		})()
		.map_err(|e| anyhow!("Error setting prior parameters: {}", e))?;

		// Set the initial state
		let init_state = MCMCState::new(
			clust_labels,
			self.sample_r(1, rng)[0],
			self.sample_p(1, rng)[0],
		)
		.map_err(|e| anyhow!("Error setting initial state: {}", e))?;

		// Use the posterior samples of r and p to sample from the induced distribution
		// on the number of clusters.
		let n_clusts_prior = pmf(
			&self
				.sample_n_clusts(NonZeroUsize::new(n_pts).unwrap(), mcmc_iters_n_clusts, rng)
				.map_err(|e| anyhow!("Error when computing pmf of n_clusters: {}", e))?,
			n_clusts_max,
		);

		// For each possible number of clusters, get maximum likelihood estimates for
		// the hyperpriors for the within-cluister and inter-cluster distances that we
		// computed earlier, weighted by the induced prior on the number of clusters
		// that we sampled above.
		let a = Array1::from_vec(a);
		let b = Array1::from_vec(b);
		let wts_a = Array1::from_vec(
			(n_clusts_min..=n_clusts_max)
				.map(|i| repeat_n(n_clusts_prior[i] as f64, sz_a[i - n_clusts_min]).collect_vec())
				.concat(),
		);
		let (delta1, alpha, beta) = {
			if a.is_empty() {
				(1.0, 1.0, 1.0)
			} else {
				let (delta1, _) = fit_gamma_mle(&a, &wts_a, 1000, None)
					.map_err(|e| anyhow!("Error fitting delta1: {}", e))?;
				let alpha = delta1
					* (n_clusts_min..=n_clusts_max)
						.map(|i| n_clusts_prior[i] * sz_a[i - n_clusts_min] as f64)
						.sum::<f64>();
				let beta = a.dot(&wts_a);
				(delta1, alpha, beta)
			}
		};
		(|| -> Result<()> {
			self.set_delta1(delta1)?.set_alpha(alpha)?.set_beta(beta)?;
			Ok(())
		})()
		.map_err(|e| anyhow!("Error fitting prior: {}", e))?;
		if self.repulsion {
			let wts_b = Array1::from_vec(
				(n_clusts_min..=n_clusts_max)
					.map(|i| {
						repeat_n(n_clusts_prior[i] as f64, sz_b[i - n_clusts_min]).collect_vec()
					})
					.concat(),
			);
			let (delta2, zeta, gamma) = {
				if b.is_empty() {
					(1.0, 1.0, 1.0)
				} else {
					let (delta2, _) = fit_gamma_mle(&b, &wts_b, 1000, None)
						.map_err(|e| anyhow!("Error fitting delta2: {}", e))?;
					let zeta = delta2
						* (n_clusts_min..=n_clusts_max)
							.map(|i| n_clusts_prior[i] * sz_b[i - n_clusts_min] as f64)
							.sum::<f64>();
					let gamma = b.dot(&wts_b);
					(delta2, zeta, gamma)
				}
			};
			(|| -> Result<()> {
				self.set_delta2(delta2)?.set_zeta(zeta)?.set_gamma(gamma)?;
				Ok(())
			})()
			.map_err(|e| anyhow!("Error fitting prior: {}", e))?;
		}
		Ok(init_state)
	}
}

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

	/// Get the range of allowed values for the number of clusters.
	#[getter(n_clusts_range)]
	fn py_get_n_clusts_range(&self) -> (NonZeroUsize, NonZeroUsize) {
		(*self.n_clusts_range.start(), *self.n_clusts_range.end())
	}

	/// Set the range of allowed values for the number of clusters.
	#[setter(n_clusts_range)]
	fn py_set_n_clusts_range(
		this: Bound<'_, Self>,
		n_clusts_range: (NonZeroUsize, NonZeroUsize),
	) -> Result<()> {
		this.borrow_mut()
			.set_n_clusts_range(n_clusts_range.0..=n_clusts_range.1)?;
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
	) -> Bound<'_, PyArray1<f64>> {
		let mut rng = get_rng(rng_seed);
		let samples = this.borrow().sample_r(n_samples, &mut rng);
		let arr = PyArray1::from_vec(this.py(), samples);
		arr
	}

	/// Sample p from its prior.
	#[pyo3(name = "sample_p")]
	fn py_sample_p(
		this: Bound<'_, Self>,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Bound<'_, PyArray1<f64>> {
		let mut rng = get_rng(rng_seed);
		let samples = this.borrow().sample_p(n_samples, &mut rng);
		let arr = PyArray1::from_vec(this.py(), samples);
		arr
	}

	/// Sample from the induced prior on cluster sizes.
	#[pyo3(name = "sample_cluster_sizes")]
	fn py_sample_cluster_sizes(
		this: Bound<'_, Self>,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Bound<'_, PyArray1<usize>> {
		let mut rng = get_rng(rng_seed);
		let samples = this.borrow().sample_cluster_sizes(n_samples, &mut rng);
		let arr = PyArray1::from_vec(this.py(), samples);
		arr
	}

	/// Sample from the induced prior on the number of clusters, conditioned on
	/// the number of points.
	#[pyo3(name = "sample_n_clusts")]
	fn py_sample_n_clusts(
		this: Bound<'_, Self>,
		n_pts: NonZeroUsize,
		n_samples: NonZeroUsize,
		rng_seed: Option<u64>,
	) -> PyResult<Bound<'_, PyArray1<usize>>> {
		let mut rng = get_rng(rng_seed);
		let samples = this.borrow().sample_n_clusts(n_pts, n_samples, &mut rng)?;
		let arr = PyArray1::from_vec(this.py(), samples);
		Ok(arr)
	}

	/// Sample from the induced prior on within-cluster distances,
	/// marginalising over cluster-specific parameters.
	#[pyo3(name = "sample_within_cluster_dists")]
	fn py_sample_within_cluster_dists(
		this: Bound<'_, Self>,
		n_samples: usize,
		rng_seed: Option<u64>,
	) -> Bound<'_, PyArray1<f64>> {
		let mut rng = get_rng(rng_seed);
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
		rng_seed: Option<u64>,
	) -> Bound<'_, PyArray1<f64>> {
		let mut rng = get_rng(rng_seed);
		let samples = this
			.borrow()
			.sample_inter_cluster_dists(n_samples, &mut rng);
		let arr = PyArray1::from_owned_array(this.py(), samples);
		arr
	}

	/// Fit a prior hyperparameters from the data, and return an
	/// initialized state for the sampler.
	/// This will modify all prior hyperparameters except for repulsion and the
	/// minimum/maximum number of clusters allowed.
	#[pyo3(name = "fit_from_data",
		signature = (
			data,
			mcmc_iters_rp=DEFAULT_MCMC_ITERS_RP_FITPRIOR,
			mle_iters=DEFAULT_MLE_ITERS_FITPRIOR,
			mcmc_iters_n_clusts=DEFAULT_MCMC_ITERS_N_CLUSTS_FITPRIOR,
			rng_seed=None,
			)
		)]
	fn py_fit_from_data(
		&mut self,
		data: &MCMCData,
		mcmc_iters_rp: NonZeroUsize,
		mle_iters: NonZeroUsize,
		mcmc_iters_n_clusts: NonZeroUsize,
		rng_seed: Option<u64>,
	) -> Result<MCMCState> {
		let mut rng = get_rng(rng_seed);
		self.fit_from_data(
			data,
			mcmc_iters_rp,
			mle_iters,
			mcmc_iters_n_clusts,
			&mut rng,
		)
	}

	fn __repr__(&self) -> String {
		format!(
			"PriorHyperParams(delta1={}, delta2={}, alpha={}, beta={}, zeta={}, gamma={}, eta={}, \
			 sigma={}, proposalsd_r={}, u={}, v={}, repulsion={}, n_clusts_range=({}, {}))",
			self.delta1,
			self.delta2,
			self.alpha,
			self.beta,
			self.zeta,
			self.gamma,
			self.eta,
			self.sigma,
			self.proposalsd_r,
			self.u,
			self.v,
			self.repulsion,
			self.n_clusts_range.start(),
			self.n_clusts_range.end(),
		)
	}

	/// Convert the PriorHyperParams object to a dictionary.
	fn as_dict(this: Bound<'_, Self>) -> Result<Bound<'_, PyDict>> {
		let slf = this.borrow();
		let dict: Bound<'_, PyDict> = PyDict::new(this.py());
		dict.set_item("delta1", slf.delta1)?;
		dict.set_item("delta2", slf.delta2)?;
		dict.set_item("alpha", slf.alpha)?;
		dict.set_item("beta", slf.beta)?;
		dict.set_item("zeta", slf.zeta)?;
		dict.set_item("gamma", slf.gamma)?;
		dict.set_item("eta", slf.eta)?;
		dict.set_item("sigma", slf.sigma)?;
		dict.set_item("proposalsd_r", slf.proposalsd_r)?;
		dict.set_item("u", slf.u)?;
		dict.set_item("v", slf.v)?;
		dict.set_item("repulsion", slf.repulsion)?;
		dict.set_item(
			"n_clusts_range",
			(slf.n_clusts_range.start(), slf.n_clusts_range.end()),
		)?;
		Ok(dict)
	}
}

impl Default for PriorHyperParams {
	fn default() -> Self {
		PriorHyperParams {
			delta1: DEFAULT_DELTA1,
			delta2: DEFAULT_DELTA2,
			alpha: DEFAULT_ALPHA,
			beta: DEFAULT_BETA,
			zeta: DEFAULT_ZETA,
			gamma: DEFAULT_GAMMA,
			eta: DEFAULT_ETA,
			sigma: DEFAULT_SIGMA,
			proposalsd_r: DEFAULT_PROPOSALSD_R,
			u: DEFAULT_U,
			v: DEFAULT_V,
			repulsion: DEFAULT_REPULSION,
			n_clusts_range: DEFAULT_N_CLUSTS_RANGE,
		}
	}
}

impl Display for PriorHyperParams {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
		write!(
			f,
			"PriorHyperParams {{\ndelta1: {},\ndelta2: {},\nalpha: {},\nbeta: {},\nzeta: \
			 {},\ngamma: {},\neta: {},\nsigma: {},\nproposalsd_r: {},\nu: {},\nv: {},\nrepulsion: \
			 {},\nn_clusts_range: {:#?}\n}}",
			self.delta1,
			self.delta2,
			self.alpha,
			self.beta,
			self.zeta,
			self.gamma,
			self.eta,
			self.sigma,
			self.proposalsd_r,
			self.u,
			self.v,
			self.repulsion,
			self.n_clusts_range,
		)
	}
}

fn pre_sample_rp<R: Rng>(
	clust_labels: &[ClusterLabel],
	params: &PriorHyperParams,
	options: &MCMCOptions,
	rng: &mut R,
) -> Result<(Array1<f64>, Array1<f64>)> {
	let r = params.sample_r(1, rng)[0];
	let p = params.sample_p(1, rng)[0];
	let mut state = MCMCState::new(clust_labels.to_owned(), r, p)?;
	let n_samples = options.n_samples();
	let (mut r_samples, mut p_samples) =
		(Vec::with_capacity(n_samples), Vec::with_capacity(n_samples));
	if n_samples != 0 {
		for _ in 0..options.n_burnin {
			state
				.sample_r_conditional(params, rng)?
				.sample_p_conditional(params, rng)?;
		}
		for i in options.n_burnin..options.n_iter {
			state
				.sample_r_conditional(params, rng)?
				.sample_p_conditional(params, rng)?;
			if (i - options.n_burnin) % options.thinning == 0 {
				r_samples.push(state.r);
				p_samples.push(state.p);
			}
		}
	}
	Ok((Array1::from_vec(r_samples), Array1::from_vec(p_samples)))
}
