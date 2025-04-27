use itertools::Itertools;
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use statrs::{
	distribution::{Beta, Continuous, Gamma},
	function::gamma::ln_gamma,
};

use crate::types::{MCMCData, MCMCState, PriorHyperParams};

/// Non-short-circuiting version of iter().position()
fn findall<T, F>(v: &[T], f: F) -> Vec<usize>
where
	T: Copy,
	F: Fn(&T) -> bool,
{
	v.iter()
		.enumerate()
		.filter_map(|(i, x)| if f(x) { Some(i) } else { None })
		.collect_vec()
}

fn num_pairs(n: u64) -> u64 { n * (n - 1) / 2 }

fn matsum(mat: &Array2<f64>, rows: &[usize], cols: &[usize]) -> f64 {
	let mut res = 0.0;
	for i in rows.iter() {
		let r = mat.row(*i);
		for j in cols.iter() {
			res += r[*j];
		}
	}
	res
}

/// Log-likelihood of the clustering, which depends on the data and the cluster
/// labels.
#[pyfunction]
pub fn ln_likelihood(data: &MCMCData, state: &MCMCState, params: &PriorHyperParams) -> f64 {
	let clust_labels = &state.clust_labels;
	let clust_list = &state.clust_list;
	let diss_mat = &data.diss_mat.0;
	let log_diss_mat = &data.log_diss_mat.0;
	let alpha = params.alpha;
	let beta = params.beta;
	let alpha_beta_ratio = alpha * beta.ln() - ln_gamma(alpha);
	let delta1 = params.delta1();
	let lgamma_delta1 = ln_gamma(delta1);

	// Cohesive part of likelihood
	let lik_cohesive = clust_list
		.par_iter()
		.map(|k| {
			let clust_k = findall(clust_labels, |x| *x == *k);
			let sz_k = clust_k.len();
			let num_pairs_k = num_pairs(sz_k as u64) as f64;
			let a = alpha + delta1 * num_pairs_k;
			let b = beta + matsum(diss_mat, &clust_k, &clust_k);
			(delta1 - 1.0) * matsum(log_diss_mat, &clust_k, &clust_k) / 2.0
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
	let n_clusts = state.n_clusts();
	let lik_repulsive = (0..n_clusts)
		.into_par_iter()
		.map(|k| {
			let clust_k = findall(clust_labels, |x| *x == clust_list[k]);
			let sz_k = clust_k.len();
			(k + 1..n_clusts)
				.map(|t| {
					let clust_t = findall(clust_labels, |x| *x == clust_list[t]);
					let sz_t = clust_t.len();
					let num_pairs_kt = (sz_k * sz_t) as f64;
					let z = zeta + delta2 * num_pairs_kt;
					let g = gamma + matsum(diss_mat, &clust_k, &clust_t);
					(delta2 - 1.0) * matsum(log_diss_mat, &clust_k, &clust_t)
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
#[pyfunction]
pub fn ln_prior(state: &MCMCState, params: &PriorHyperParams) -> f64 {
	let n_pts = state.clust_labels.len() as f64;
	let clust_sizes = &state.clust_sizes;
	let n_clusts = state.n_clusts() as f64;
	let r = state.r;
	let p = state.p;
	let eta = params.eta;
	let sigma = params.sigma;
	let u = params.u;
	let v = params.v;

	ln_gamma(n_clusts + 1.0) + (n_pts - n_clusts) * p.ln() + (r * n_clusts) * (1.0 - p).ln()
		- n_clusts * ln_gamma(r)
		+ Gamma::new(eta, sigma).unwrap().pdf(r)
		+ Beta::new(u, v).unwrap().pdf(p)
		+ state
			.clust_list
			.iter()
			.map(|j| {
				let nj = clust_sizes[j] as f64;
				nj.ln() + ln_gamma(nj + r - 1.0)
			})
			.sum::<f64>()
}
