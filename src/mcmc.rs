use std::time::Instant;

use anyhow::{Result, anyhow};
use itertools::Itertools;
use ndarray::{Array1, Array2, s};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use statrs::{
	distribution::{Beta, Continuous, Gamma},
	function::gamma::ln_gamma,
};

use crate::{
	MCMCOptions,
	MCMCResult,
	types::{Array2Wrapper, MCMCData, MCMCState, PriorHyperParams},
	utils::{num_pairs, symm_mat_sum},
};

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
	let diss_mat = data.diss_mat();
	let ln_diss_mat = data.ln_diss_mat();
	let alpha = params.alpha();
	let beta = params.beta();
	let alpha_beta_ratio = alpha * beta.ln() - ln_gamma(alpha);
	let delta1 = params.delta1();
	let lgamma_delta1 = ln_gamma(delta1);

	// Cohesive part of likelihood
	let lik_cohesive = clust_list
		.par_iter()
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
	let n_clusts = state.n_clusts().get();
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
	let n_clusts = state.n_clusts().get() as f64;
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

pub fn run_sampler<R: Rng>(
	data: &MCMCData,
	options: &MCMCOptions,
	init_state: &mut MCMCState,
	params: &PriorHyperParams,
	rng: &mut R,
) -> Result<MCMCResult> {
	let state = init_state;
	let n_samples = options.n_samples();
	let n_pts = data.n_pts().get();
	if params.n_clusts_range().start().get() > n_pts {
		return Err(anyhow!(
			"Parameter n_clusts_min ({}) is greater than number of points ({})",
			params.n_clusts_range().start().get(),
			n_pts
		));
	}
	if state.clust_labels().len() == n_pts {
		return Err(anyhow!(
			"Invalid initial state: found {} cluster labels, but the data has {} points",
			state.clust_labels().len(),
			n_pts
		));
	}
	if !params.n_clusts_range().contains(&state.n_clusts()) {
		return Err(anyhow!(
			"Initial state has {} clusters, which is incompatible with the parameters given where \
			 the number of clusters must be in [{} and {}]",
			state.n_clusts(),
			params.n_clusts_range().start().get(),
			params.n_clusts_range().end().get()
		));
	}
	let mut result = MCMCResult {
		clusts: vec![Vec::with_capacity(n_pts); n_samples],
		posterior_coclustering: Array2Wrapper(Array2::zeros((n_pts, n_pts))),
		n_clusts: vec![0; n_samples],
		n_clusts_ess: 0.0,
		n_clusts_acf: Vec::with_capacity(n_samples),
		n_clusts_iac: 0.0,
		n_clusts_mean: 0.0,
		n_clusts_variance: 0.0,
		r: Vec::with_capacity(n_samples),
		r_ess: 0.0,
		r_acf: Vec::with_capacity(n_samples),
		r_iac: 0.0,
		r_mean: 0.0,
		r_variance: 0.0,
		p: Vec::with_capacity(n_samples),
		p_ess: 0.0,
		p_acf: Vec::with_capacity(n_samples),
		p_iac: 0.0,
		p_mean: 0.0,
		p_variance: 0.0,
		splitmerge_acceptances: Vec::with_capacity(n_samples), // todo
		splitmerge_acceptance_rate: 0.0,                       // todo
		splitmerge_splits: Vec::with_capacity(n_samples),      // todo
		r_acceptances: Vec::with_capacity(n_samples),
		r_acceptance_rate: 0.0,
		runtime: 0.0,
		mean_iter_time: 0.0,
		ln_lik: Vec::with_capacity(n_samples),
		ln_posterior: Vec::with_capacity(n_samples),
		options: *options,
		params: params.clone(),
	};
	if n_samples != 0 {
		let start_time = Instant::now();
		for _ in 0..options.n_burnin {
			state
				.sample_r_conditional(params, rng)?
				.sample_p_conditional(params, rng)?;
		}
		let mut j = 0;
		for i in options.n_burnin..options.n_iter {
			state
				.sample_r_conditional(params, rng)?
				.sample_p_conditional(params, rng)?
				.sample_clusters_gibbs(data, params, rng)?;
			if (i - options.n_burnin) % options.thinning == 0 {
				result.clusts[j].extend(state.clust_labels.iter());
				result.n_clusts[j] = state.n_clusts().get();
				result.r[j] = state.r;
				result.p[j] = state.p;
				result.r_acceptances[j] = state.r_accepted;
				result.ln_lik[j] = ln_likelihood(data, state, params);
				result.ln_posterior[j] = result.ln_lik[j] + ln_prior(state, params);
				j += 1;
			}
		}
		result.runtime = start_time.elapsed().as_secs_f64();
		result.mean_iter_time = result.runtime / options.n_iter as f64;
		(
			result.n_clusts_mean,
			result.n_clusts_variance,
			result.n_clusts_acf,
			result.n_clusts_iac,
			result.n_clusts_ess,
		) = aux_stats(&Array1::from_iter(
			result.n_clusts.iter().map(|x| *x as f64),
		));
		(
			result.r_mean,
			result.r_variance,
			result.r_acf,
			result.r_iac,
			result.r_ess,
		) = aux_stats(&Array1::from_iter(result.r.iter().copied()));
		(
			result.p_mean,
			result.p_variance,
			result.p_acf,
			result.p_iac,
			result.p_ess,
		) = aux_stats(&Array1::from_iter(result.p.iter().copied()));
	}
	todo!();
}

#[pyfunction]
#[pyo3(name = "run_sampler")]
pub(crate) fn py_run_sampler(
	data: &MCMCData,
	options: &MCMCOptions,
	init_state: &mut MCMCState,
	params: &PriorHyperParams,
) -> Result<MCMCResult> {
	let mut rng = SmallRng::from_entropy();
	run_sampler(data, options, init_state, params, &mut rng)
}
