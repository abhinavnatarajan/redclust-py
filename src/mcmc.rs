use std::thread;

use anyhow::{Result, anyhow};
use itertools::Itertools;
#[cfg(feature = "python-module")]
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use statrs::{
	distribution::{Beta, Continuous, Gamma},
	function::gamma::ln_gamma,
};

use crate::*;
/// Log-likelihood of the clustering, which depends on the data and the cluster
/// labels. Will only consider the first n points in the data where
/// ``n=state.clust_labels.len()``.
pub fn ln_likelihood(
	data: &MCMCData,
	mcmc_state: &MCMCState,
	prior_params: &PriorHyperParams,
	likelihood_options: &LikelihoodOptions,
) -> f64 {
	let clust_labels = &mcmc_state.clust_labels;
	let clust_list = mcmc_state.clust_list.iter().copied().collect_vec();
	let diss_mat = data.dissimilarities();
	let ln_diss_mat = data.ln_dissimilarities();
	let alpha = prior_params.alpha();
	let beta = prior_params.beta();
	let alpha_beta_ratio = alpha * beta.ln() - ln_gamma(alpha);
	let delta1 = prior_params.delta1();
	let lgamma_delta1 = ln_gamma(delta1);

	// Cohesive part of likelihood
	let lik_cohesive = clust_list
		.iter()
		.map(|k| {
			let clust_k = clust_labels.iter().positions(|x| *x == *k).collect_vec();
			let sz_k = clust_k.len();
			let num_pairs_k = num_pairs(sz_k as u64) as f64;
			let a = alpha + delta1 * num_pairs_k;
			let b = beta + symm_mat_sum(diss_mat, &clust_k, &clust_k);
			(delta1 - 1.0) * symm_mat_sum(ln_diss_mat, &clust_k, &clust_k) / 2.0
				- num_pairs_k * lgamma_delta1
				+ alpha_beta_ratio
				+ ln_gamma(a)
				- a * b.ln()
		})
		.sum::<f64>();

	if !likelihood_options.repulsion() {
		return lik_cohesive;
	};

	// Repulsive part of likelihood
	let zeta = prior_params.zeta();
	let gamma = prior_params.gamma();
	let zeta_gamma_ratio = zeta * gamma.ln() - ln_gamma(zeta);
	let delta2 = prior_params.delta2();
	let lgamma_delta2 = ln_gamma(delta2);
	let n_clusts = mcmc_state.num_clusts().get();
	let lik_repulsive = (0..n_clusts)
		.into_par_iter()
		.map(|k| {
			let clust_k = clust_labels
				.iter()
				.positions(|x| *x == clust_list[k])
				.collect_vec();
			let sz_k = clust_k.len();
			(k + 1..n_clusts)
				.map(|t| {
					let clust_t = clust_labels
						.iter()
						.positions(|x| *x == clust_list[t])
						.collect_vec();
					let sz_t = clust_t.len();
					let num_pairs_kt = (sz_k * sz_t) as f64;
					let z = zeta + delta2 * num_pairs_kt;
					let g = gamma + symm_mat_sum(diss_mat, &clust_k, &clust_t);
					(delta2 - 1.0) * symm_mat_sum(ln_diss_mat, &clust_k, &clust_t)
						- num_pairs_kt * lgamma_delta2
						+ zeta_gamma_ratio + ln_gamma(z)
						- z * g.ln()
				})
				.sum::<f64>()
		})
		.sum::<f64>();

	lik_cohesive + lik_repulsive
}

/// Log-prior of the clustering, which depends only on the cluster sizes.
pub fn ln_prior(mcmc_state: &MCMCState, prior_params: &PriorHyperParams) -> f64 {
	let n_pts = mcmc_state.clust_labels.len() as f64;
	let clust_sizes = &mcmc_state.clust_sizes;
	let n_clusts = mcmc_state.num_clusts().get() as f64;
	let r = mcmc_state.r;
	let p = mcmc_state.p;
	let eta = prior_params.eta();
	let sigma = prior_params.sigma();
	let u = prior_params.u();
	let v = prior_params.v();

	ln_gamma(n_clusts + 1.0) + (n_pts - n_clusts) * p.ln() + (r * n_clusts) * (1.0 - p).ln()
		- n_clusts * ln_gamma(r)
		+ Gamma::new(eta, sigma).unwrap().pdf(r)
		+ Beta::new(u, v).unwrap().pdf(p)
		+ mcmc_state
			.clust_list
			.iter()
			.map(|&j| {
				let nj = clust_sizes[j as usize] as f64;
				nj.ln() + ln_gamma(nj + r - 1.0)
			})
			.sum::<f64>()
}

fn run_chain<R: Rng>(
	data: &MCMCData,
	prior_params: &PriorHyperParams,
	likelihood_options: &LikelihoodOptions,
	init_state: &MCMCState,
	mcmc_options: &MCMCOptions,
	rng: &mut R,
) -> Result<MCMCResult> {
	let mut state = init_state.clone();
	let n_samples = mcmc_options.num_samples();
	let n_pts = data.num_points().get();
	let mut result = MCMCResult::with_capacity(n_pts, n_samples);
	if n_samples != 0 {
		for _ in 0..mcmc_options.num_burnin {
			state
				.sample_r_conditional(prior_params, rng)?
				.sample_p_conditional(prior_params, rng)?;
		}
		let mut j = 0;
		for i in mcmc_options.num_burnin..mcmc_options.num_iter {
			state
				.sample_r_conditional(prior_params, rng)?
				.sample_p_conditional(prior_params, rng)?
				.sample_clusters_gibbs(data, prior_params, likelihood_options, rng)?;
			if state.r_accepted {
				result.r_acceptance_rate += 1.0;
			}
			if (i - mcmc_options.num_burnin) % mcmc_options.thinning == 0 {
				result.clusts[j].extend(state.clust_labels.iter());
				result.num_clusts[j] = state.num_clusts().get();
				result.r[j] = state.r;
				result.p[j] = state.p;
				result.ln_likelihood[j] =
					ln_likelihood(data, &state, prior_params, likelihood_options);
				result.ln_posterior[j] = result.ln_likelihood[j] + ln_prior(&state, prior_params);
				j += 1;
			}
		}
		result.r_acceptance_rate /= mcmc_options.num_iter as f64;
	}
	Ok(result)
}

pub fn run_sampler(
	data: &MCMCData,
	prior_params: &PriorHyperParams,
	likelihood_options: &LikelihoodOptions,
	init_state: &MCMCState,
	mcmc_options: &MCMCOptions,
) -> Result<MCMCResult> {
	// Check that data and params are compatible
	if data.num_points() < likelihood_options.min_num_clusts() {
		return Err(anyhow!(
			"Parameter min_num_clusts = {} which is greater than the number of points ({})",
			likelihood_options.min_num_clusts(),
			data.num_points()
		));
	}
	// Check that data and init_state are compatible
	if init_state.clust_labels().len() != data.num_points().get() {
		return Err(anyhow!(
			"Invalid initial state: found {} cluster labels, but the data has {} points",
			init_state.clust_labels().len(),
			data.num_points()
		));
	}
	// Check that params and init_state are compatible
	if !(likelihood_options.min_num_clusts()..=likelihood_options.max_num_clusts())
		.contains(&init_state.num_clusts())
	{
		return Err(anyhow!(
			"Initial state has {} clusters, which is incompatible with the parameters given where \
			 the number of clusters must be in [{} and {}]",
			init_state.num_clusts(),
			likelihood_options.min_num_clusts(),
			likelihood_options.max_num_clusts()
		));
	}
	let mut rng = get_rng(mcmc_options.rng_seed);
	let results_vec = thread::scope(|s| {
		(0..mcmc_options.num_chains.get())
			.map(|_| {
				rng.jump();
				let mut thread_rng = rng.clone();
				s.spawn(move || {
					run_chain(
						data,
						prior_params,
						likelihood_options,
						init_state,
						mcmc_options,
						&mut thread_rng,
					)
				})
			})
			.map(|handle| handle.join().unwrap())
			.collect::<Vec<_>>()
	});

	{
		if results_vec.iter().all(|x| x.is_err()) {
			Err(anyhow!("All chains failed."))
		} else {
			Ok(results_vec
				.into_iter()
				.filter_map(|x| x.ok())
				.reduce(|acc, x| acc.merge_with(x))
				.unwrap())
		}
	}
}

#[pyfunction]
#[pyo3(name = "run_sampler")]
pub(crate) fn py_run_sampler(
	data: &MCMCData,
	prior_params: &PriorHyperParams,
	likelihood_options: &LikelihoodOptions,
	mcmc_options: &MCMCOptions,
	init_state: &mut MCMCState,
) -> Result<MCMCResult> {
	run_sampler(
		data,
		prior_params,
		likelihood_options,
		init_state,
		mcmc_options,
	)
}
