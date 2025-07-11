use std::{
	iter::{repeat_n, zip},
	num::NonZeroUsize,
};

use anyhow::{Result, anyhow};
use itertools::Itertools;
use ndarray::{Array1, Array2, s};
#[cfg(feature = "python-module")]
use pyo3::prelude::*;
use rand::{Rng, distributions::Distribution};
use statrs::{
	distribution::{Beta, Gamma, NegativeBinomial},
	function::beta::ln_beta,
};

use crate::{
	ClusterLabel,
	MCMCData,
	MCMCOptions,
	MCMCState,
	utils::{fit_beta_mle, fit_gamma_mle, get_rng, knee_pos, pmf, sample_from_ln_probs},
};

mod prior_hyper_params;
#[cfg(feature = "python-module")]
mod python_bindings;

pub use prior_hyper_params::PriorHyperParams;
use prior_hyper_params::{
	DEFAULT_MAX_NUM_CLUSTS,
	DEFAULT_MIN_NUM_CLUSTS,
	DEFAULT_REPULSION,
};

const NONZERO_THOUSAND: NonZeroUsize = NonZeroUsize::new(1000).unwrap();
pub(super) const DEFAULT_MCMC_ITERS_RP_FITPRIOR: NonZeroUsize = NONZERO_THOUSAND;
pub(super) const DEFAULT_MLE_ITERS_FITPRIOR: NonZeroUsize = NONZERO_THOUSAND;
pub(super) const DEFAULT_MCMC_ITERS_N_CLUSTS_FITPRIOR: NonZeroUsize = NONZERO_THOUSAND;

/// Distribution on cluster sizes induced by the prior hyperparameters.
#[derive(Clone)]
pub struct ClusterSizePrior {
	r_prior: Gamma,
	p_prior: Beta,
}

impl Distribution<usize> for ClusterSizePrior {
	fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
		let r = self.r_prior.sample(rng);
		let p = self.p_prior.sample(rng);
		NegativeBinomial::new(r, p).unwrap().sample(rng) as usize
	}
}

/// Distribution on the number of clusters induced by the prior hyperparameters.
#[derive(Clone)]
pub struct NumClustersPrior {
	n_pts: usize,
	r_prior: Gamma,
	p_prior: Beta,
}

impl Distribution<usize> for NumClustersPrior {
	fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
		let r = self.r_prior.sample(rng);
		let p = self.p_prior.sample(rng);
		let mut logprobs = Array1::<f64>::zeros(self.n_pts);
		let k = Array1::from_iter((1..self.n_pts).map(|x| x as f64));
		let n = self.n_pts as f64;
		logprobs.slice_mut(s![0..self.n_pts - 1]).assign(
			&(r * &k * (1.0 - p).ln() + (n - &k) * p.ln()
				- (n - &k).ln()
				- zip(r * &k, n - &k)
					.map(|(x, y)| ln_beta(x, y))
					.collect::<Array1<f64>>()),
		);
		logprobs[self.n_pts - 1] = r * n * (1.0 - p).ln();
		sample_from_ln_probs(&logprobs.view(), rng).unwrap() + 1
	}
}

/// Distribution on the within-cluster dissimilarities induced by the prior
/// hyperparameters.
#[derive(Clone)]
pub struct IntraClusterDissimilarityPrior {
	delta1: f64,
	lambda_prior: Gamma,
}

impl Distribution<f64> for IntraClusterDissimilarityPrior {
	fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		let lambda = self.lambda_prior.sample(rng);
		Gamma::new(self.delta1, lambda).unwrap().sample(rng)
	}
}

/// Distribution on the inter-cluster dissimilarities induced by the prior
/// hyperparameters.
#[derive(Clone)]
pub struct InterClusterDissimilarityPrior {
	delta2: f64,
	theta_prior: Gamma,
}

impl Distribution<f64> for InterClusterDissimilarityPrior {
	fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		let theta = self.theta_prior.sample(rng);
		Gamma::new(self.delta2, theta).unwrap().sample(rng)
	}
}

/// Fit a prior hyperparameters from the data, and return an
/// initialized state for the sampler.
pub fn init_from_data<R: Rng>(
	data: &MCMCData,
	min_num_clusts: NonZeroUsize,
	max_num_clusts: NonZeroUsize,
	repulsion: bool,
	mcmc_iters_rp: NonZeroUsize,
	mle_iters: NonZeroUsize,
	mcmc_iters_n_clusts: NonZeroUsize,
	rng: &mut R,
) -> Result<(MCMCState, PriorHyperParams)> {
	let mut result =
		PriorHyperParams::default().with_range_num_clusts(min_num_clusts..=max_num_clusts)?;
	result.set_repulsion(repulsion);
	let n_pts = data.num_points();
	let min_num_clusts = result.min_num_clusts().get();
	let max_num_clusts = result.max_num_clusts().get();
	if !(min_num_clusts..=max_num_clusts).contains(&n_pts.get()) {
		return Err(anyhow!(
			"Number of clusters must be between {} and {}.",
			min_num_clusts,
			max_num_clusts
		));
	}

	// Set of within-cluster distances
	let mut a = Vec::<f64>::new();
	// Set of inter-cluster distances
	let mut b = Vec::<f64>::new();
	// Number of within-cluster distances for each clustering
	let mut sz_a = Vec::<usize>::with_capacity(max_num_clusts + 1 - min_num_clusts);
	let mut sz_b = Vec::<usize>::with_capacity(max_num_clusts + 1 - min_num_clusts);
	let mut losses = Vec::<f64>::with_capacity(max_num_clusts + 1 - min_num_clusts);

	// For each possible number of clusters, create a corresponding clustering
	// and record the within-cluster and inter-cluster distances, and the clustering
	// loss.
	for i in min_num_clusts..=max_num_clusts {
		let mut medoids = kmedoids::random_initialization(n_pts.get(), i, rng);
		let (loss, clust_labels, ..) = kmedoids::fasterpam::<Array2<f64>, f64, f64>(
			data.dissimilarities(),
			&mut medoids,
			1000,
		);
		losses.push(loss);
		let clust_labels = clust_labels
			.into_iter()
			.map(|x| x as ClusterLabel)
			.collect_vec();
		let within_cluster_dists = data.within_cluster_dissimilarities(&clust_labels).unwrap();
		sz_a.push(within_cluster_dists.len());
		a.extend(within_cluster_dists);
		let inter_cluster_dists = data.inter_cluster_dissimilarities(&clust_labels).unwrap();
		sz_b.push(inter_cluster_dists.len());
		b.extend(inter_cluster_dists);
	}
	// Use the elbow method to find the optimal number of clusters.
	let n_clusts_init = knee_pos(&losses).unwrap() + min_num_clusts;

	// Get a clustering with the optimal number of clusters for the initial state.
	let clust_labels = {
		let mut medoids = kmedoids::random_initialization(n_pts.get(), n_clusts_init, rng);
		let (_, clust_labels, ..): (f64, _, _, _) =
			kmedoids::rand_fasterpam(data.dissimilarities(), &mut medoids, 1000, rng);
		clust_labels
	}
	.into_iter()
	.map(|x| x as ClusterLabel)
	.collect_vec();

	// Given the optimal clustering, sample r and p from the conditional posterior
	// using default values for their hyperpriors.
	let (r_samples, p_samples) = pre_sample_rp(
		&clust_labels,
		&result,
		&MCMCOptions {
			num_iter: mcmc_iters_rp.get(),
			num_burnin: 0,
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
		result
			.set_proposalsd_r(proposalsd_r)?
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
		Some(result.r_prior()?.sample(rng)),
		Some(result.p_prior()?.sample(rng)),
	)
	.map_err(|e| anyhow!("Error setting initial state: {}", e))?;

	// Use the posterior samples of r and p to sample from the induced distribution
	// on the number of clusters.
	let n_clusts_prior = pmf(
		&result
			.num_clusters_prior(n_pts)?
			.sample_iter(rng)
			.take(mcmc_iters_n_clusts.get())
			.collect_vec(),
		// .map_err(|e| anyhow!("Error when computing pmf of n_clusters: {}", e))?,
		max_num_clusts,
	);

	// For each possible number of clusters, get maximum likelihood estimates for
	// the hyperpriors for the within-cluister and inter-cluster distances that we
	// computed earlier, weighted by the induced prior on the number of clusters
	// that we sampled above.
	let a = Array1::from_vec(a);
	let b = Array1::from_vec(b);
	let wts_a = Array1::from_vec(
		(min_num_clusts..=max_num_clusts)
			.map(|i| repeat_n(n_clusts_prior[i] as f64, sz_a[i - min_num_clusts]).collect_vec())
			.concat(),
	);
	let (delta1, alpha, beta) = {
		if a.is_empty() {
			(1.0, 1.0, 1.0)
		} else {
			let (delta1, _) = fit_gamma_mle(&a, &wts_a, 1000, None)
				.map_err(|e| anyhow!("Error fitting delta1: {}", e))?;
			let alpha = delta1
				* (min_num_clusts..=max_num_clusts)
					.map(|i| n_clusts_prior[i] * sz_a[i - min_num_clusts] as f64)
					.sum::<f64>();
			let beta = a.dot(&wts_a);
			(delta1, alpha, beta)
		}
	};
	(|| -> Result<()> {
		result
			.set_delta1(delta1)?
			.set_alpha(alpha)?
			.set_beta(beta)?;
		Ok(())
	})()
	.map_err(|e| anyhow!("Error fitting prior: {}", e))?;
	let wts_b = Array1::from_vec(
		(min_num_clusts..=max_num_clusts)
			.map(|i| repeat_n(n_clusts_prior[i] as f64, sz_b[i - min_num_clusts]).collect_vec())
			.concat(),
	);
	let (delta2, zeta, gamma) = {
		if b.is_empty() {
			(1.0, 1.0, 1.0)
		} else {
			let (delta2, _) = fit_gamma_mle(&b, &wts_b, 1000, None)
				.map_err(|e| anyhow!("Error fitting delta2: {}", e))?;
			let zeta = delta2
				* (min_num_clusts..=max_num_clusts)
					.map(|i| n_clusts_prior[i] * sz_b[i - min_num_clusts] as f64)
					.sum::<f64>();
			let gamma = b.dot(&wts_b);
			(delta2, zeta, gamma)
		}
	};
	(|| -> Result<()> {
		result
			.set_delta2(delta2)?
			.set_zeta(zeta)?
			.set_gamma(gamma)?;
		Ok(())
	})()
	.map_err(|e| anyhow!("Error fitting prior: {}", e))?;
	Ok((init_state, result))
}

/// Fit a prior hyperparameters from the data, and return an
/// initialized state for the sampler.
/// This will modify all prior hyperparameters except for repulsion and the
/// minimum/maximum number of clusters allowed.
#[pyfunction(name = "init_from_data",
		signature = (
			data,
			min_num_clusts=DEFAULT_MIN_NUM_CLUSTS,
			max_num_clusts=DEFAULT_MAX_NUM_CLUSTS,
			repulsion=DEFAULT_REPULSION,
			mcmc_iters_rp=DEFAULT_MCMC_ITERS_RP_FITPRIOR,
			mle_iters=DEFAULT_MLE_ITERS_FITPRIOR,
			mcmc_iters_n_clusts=DEFAULT_MCMC_ITERS_N_CLUSTS_FITPRIOR,
			rng_seed=None,
			)
		)]
fn py_init_from_data(
	data: &MCMCData,
	min_num_clusts: NonZeroUsize,
	max_num_clusts: NonZeroUsize,
	repulsion: bool,
	mcmc_iters_rp: NonZeroUsize,
	mle_iters: NonZeroUsize,
	mcmc_iters_n_clusts: NonZeroUsize,
	rng_seed: Option<u64>,
) -> Result<(MCMCState, PriorHyperParams)> {
	let mut rng = get_rng(rng_seed);
	init_from_data(
		data,
		min_num_clusts,
		max_num_clusts,
		repulsion,
		mcmc_iters_rp,
		mle_iters,
		mcmc_iters_n_clusts,
		&mut rng,
	)
}

#[cfg(feature = "python-module")]
#[pymodule]
pub(crate) fn prior_pymodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(py_init_from_data, m)?)?;
	Ok(())
}

fn pre_sample_rp<R: Rng>(
	clust_labels: &[ClusterLabel],
	params: &PriorHyperParams,
	options: &MCMCOptions,
	rng: &mut R,
) -> Result<(Array1<f64>, Array1<f64>)> {
	let r = params.r_prior()?.sample(rng);
	let p = params.p_prior()?.sample(rng);
	let mut state = MCMCState::new(clust_labels.to_owned(), Some(r), Some(p))?;
	let n_samples = options.num_samples();
	let (mut r_samples, mut p_samples) =
		(Vec::with_capacity(n_samples), Vec::with_capacity(n_samples));
	if n_samples != 0 {
		for _ in 0..options.num_burnin {
			state
				.sample_r_conditional(params, rng)?
				.sample_p_conditional(params, rng)?;
		}
		for i in options.num_burnin..options.num_iter {
			state
				.sample_r_conditional(params, rng)?
				.sample_p_conditional(params, rng)?;
			if (i - options.num_burnin) % options.thinning == 0 {
				r_samples.push(state.r);
				p_samples.push(state.p);
			}
		}
	}
	Ok((Array1::from_vec(r_samples), Array1::from_vec(p_samples)))
}
