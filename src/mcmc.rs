use std::thread;

use anyhow::{Result, anyhow};
use itertools::Itertools;
use ndarray::{Array1, s};
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use statrs::{
	distribution::{Beta, Continuous, Gamma},
	function::gamma::ln_gamma,
};

use crate::*;

// Mean, variance, autocorrelation function, integrated autocorrelation time,
// and effective sample size.
fn aux_stats(x: &Array1<f64>) -> (f64, f64, Vec<f64>, f64, f64) {
	if x.is_empty() {
		return (0.0, 0.0, vec![], 0.0, 0.0);
	}
	let lags = (0..(x.len() - 1).min(10 * (x.len() as f64).log10() as usize)).collect_vec();
	let n = x.len();
	let mean = x.mean().unwrap();
	let var = x.var(0.0);
	let y = x - mean;
	let mut acf = Vec::with_capacity(lags.len());
	for lag in lags {
		if lag >= n {
			acf.push(0.0);
		} else {
			acf.push(y.slice(s![..n - lag]).dot(&y.slice(s![lag..])) / var);
		}
	}
	let iac = acf.iter().sum::<f64>() * 2.0;
	let ess = x.len() as f64 / iac;
	(mean, var, acf, iac, ess)
}

/// Log-likelihood of the clustering, which depends on the data and the cluster
/// labels. Will only consider the first n points in the data where
/// ``n=state.clust_labels.len()``.
pub fn ln_likelihood(data: &MCMCData, state: &MCMCState, params: &PriorHyperParams) -> f64 {
	let clust_labels = &state.clust_labels;
	let clust_list = state.clust_list.iter().copied().collect_vec();
	let diss_mat = data.dissimilarities();
	let ln_diss_mat = data.ln_dissimilarities();
	let alpha = params.alpha();
	let beta = params.beta();
	let alpha_beta_ratio = alpha * beta.ln() - ln_gamma(alpha);
	let delta1 = params.delta1();
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

	if !params.repulsion() {
		return lik_cohesive;
	};

	// Repulsive part of likelihood
	let zeta = params.zeta();
	let gamma = params.gamma();
	let zeta_gamma_ratio = zeta * gamma.ln() - ln_gamma(zeta);
	let delta2 = params.delta2();
	let lgamma_delta2 = ln_gamma(delta2);
	let n_clusts = state.num_clusts().get();
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
pub fn ln_prior(state: &MCMCState, params: &PriorHyperParams) -> f64 {
	let n_pts = state.clust_labels.len() as f64;
	let clust_sizes = &state.clust_sizes;
	let n_clusts = state.num_clusts().get() as f64;
	let r = state.r;
	let p = state.p;
	let eta = params.eta();
	let sigma = params.sigma();
	let u = params.u();
	let v = params.v();

	ln_gamma(n_clusts + 1.0) + (n_pts - n_clusts) * p.ln() + (r * n_clusts) * (1.0 - p).ln()
		- n_clusts * ln_gamma(r)
		+ Gamma::new(eta, sigma).unwrap().pdf(r)
		+ Beta::new(u, v).unwrap().pdf(p)
		+ state
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
	params: &PriorHyperParams,
	init_state: &MCMCState,
	options: &MCMCOptions,
	rng: &mut R,
) -> Result<MCMCResult> {
	let mut state = init_state.clone();
	let n_samples = options.num_samples();
	let n_pts = data.num_points().get();
	let mut result = MCMCResult::with_capacity(n_pts, n_samples);
	if n_samples != 0 {
		for _ in 0..options.num_burnin {
			state
				.sample_r_conditional(params, rng)?
				.sample_p_conditional(params, rng)?;
		}
		let mut j = 0;
		for i in options.num_burnin..options.num_iter {
			state
				.sample_r_conditional(params, rng)?
				.sample_p_conditional(params, rng)?
				.sample_clusters_gibbs(data, params, rng)?;
			if state.r_accepted {
				result.r_acceptance_rate += 1.0;
			}
			if (i - options.num_burnin) % options.thinning == 0 {
				result.clusts[j].extend(state.clust_labels.iter());
				result.num_clusts[j] = state.num_clusts().get();
				result.r[j] = state.r;
				result.p[j] = state.p;
				result.ln_likelihood[j] = ln_likelihood(data, &state, params);
				result.ln_posterior[j] = result.ln_likelihood[j] + ln_prior(&state, params);
				j += 1;
			}
		}
		result.r_acceptance_rate /= options.num_iter as f64;
	}
	Ok(result)
}

pub fn run_sampler(
	data: &MCMCData,
	params: &PriorHyperParams,
	init_state: &MCMCState,
	options: &MCMCOptions,
) -> Result<MCMCResult> {
	// Check that data and params are compatible
	if data.num_points() < params.min_num_clusts() {
		return Err(anyhow!(
			"Parameter min_num_clusts = {} which is greater than the number of points ({})",
			params.min_num_clusts(),
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
	if !(params.min_num_clusts()..=params.max_num_clusts()).contains(&init_state.num_clusts()) {
		return Err(anyhow!(
			"Initial state has {} clusters, which is incompatible with the parameters given where \
			 the number of clusters must be in [{} and {}]",
			init_state.num_clusts(),
			params.min_num_clusts(),
			params.max_num_clusts()
		));
	}
	let mut rng = get_rng(options.rng_seed);
	let results_vec = thread::scope(|s| {
		(0..options.num_chains.get())
			.map(|_| {
				rng.jump();
				let mut thread_rng = rng.clone();
				s.spawn(move || run_chain(data, params, init_state, options, &mut thread_rng))
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
	options: &MCMCOptions,
	init_state: &mut MCMCState,
	params: &PriorHyperParams,
) -> Result<MCMCResult> {
	run_sampler(data, params, init_state, options)
}
