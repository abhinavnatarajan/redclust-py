use std::{collections::HashMap, iter::repeat_n};

use anyhow::{Result, anyhow};
use itertools::Itertools;
use ndarray::{Array1, array};
use ndarray_linalg::SolveH;
use rand::rngs::SmallRng;
use special::Gamma;

use crate::types::{ClusterLabel, MCMCData, MCMCOptions, MCMCState, PriorHyperParams};

impl MCMCData {
	/// Fit a PriorHyperParams instance based on the data, and return an
	/// initialized state for the sampler.
	pub fn fit_prior(
		&self,
		kmin: usize,
		kmax: usize,
		repulsion: bool,
		// method: PriorClusteringMethod,
		rng: &mut SmallRng,
	) -> Result<(PriorHyperParams, MCMCState)> {
		let n_pts = self.diss_mat.0.nrows();
		let kmin = kmin.min(n_pts);
		let kmax = kmax.min(n_pts);
		let mut res = PriorHyperParams::default();
		res.set_n_clusts_range(kmin..=kmax)?;
		// Set of within-cluster distances
		let mut a = Vec::<f64>::new();
		// Set of inter-cluster distances
		let mut b = Vec::<f64>::new();
		// Number of within-cluster distances for each clustering
		let mut sz_a = Vec::<usize>::with_capacity(kmax + 1 - kmin);
		let mut sz_b = Vec::<usize>::with_capacity(kmax + 1 - kmin);
		let mut losses = Vec::<f64>::with_capacity(kmax + 1 - kmin);

		for i in kmin..=kmax {
			let mut medoids = kmedoids::random_initialization(n_pts, i, rng);
			let (loss, clust_labels, ..) =
				kmedoids::rand_fasterpam(&self.diss_mat.0, &mut medoids, 1000, rng);
			losses.push(loss);
			let clust_labels = clust_labels
				.into_iter()
				.map(|x| x as ClusterLabel)
				.collect_vec();
			let within_cluster_dists = self.within_cluster_dists(&clust_labels).unwrap();
			sz_a.push(within_cluster_dists.len());
			a.extend(within_cluster_dists);
			let inter_cluster_dists = self.inter_cluster_dists(&clust_labels).unwrap();
			sz_b.push(inter_cluster_dists.len());
			b.extend(inter_cluster_dists);
		}
		let n_clusts_init = losses
			.iter()
			.enumerate()
			.fold((0, f64::INFINITY), |(i, min_l), (j, &l)| {
				if l < min_l { (j, l) } else { (i, min_l) }
			})
			.0;
		let clust_labels = {
			let mut medoids = kmedoids::random_initialization(n_pts, n_clusts_init, rng);
			let (_, clust_labels, ..): (f64, _, _, _) =
				kmedoids::rand_fasterpam(&self.diss_mat.0, &mut medoids, 1000, rng);
			clust_labels
		}
		.into_iter()
		.map(|x| x as ClusterLabel)
		.collect_vec();

		// Partition prior parameters
		let (r_samples, p_samples) = pre_sample_rp(
			&clust_labels,
			&PriorHyperParams::default(),
			&MCMCOptions::default(),
			rng,
		)?;
		let proposalsd_r = r_samples.std(0.0);
		let (eta, sigma) = fit_gamma(
			&r_samples,
			&Array1::<f64>::ones(r_samples.len()),
			1000,
			None,
		);
		let (u, v) = fit_beta(&p_samples, 1000, None);

		let mut params = PriorHyperParams::default();
		(|| -> Result<()> {
			params.set_eta(eta)?.set_sigma(sigma)?.set_u(u)?.set_v(v)?;
			Ok(())
		})()
		.map_err(|e| anyhow!("Error setting prior parameters: {}", e))?;

		// Create the init state
		let init_state = MCMCState::new(
			clust_labels,
			params.sample_r(1, rng)[0],
			params.sample_p(1, rng)[0],
		)
		.unwrap();

		// Use the partition prior parameters to sample the induced prior on the number
		// of clusters
		let k_prior = pmf(params
			.sample_n_clusts(n_pts, 10000, rng)
			.as_slice()
			.unwrap());

		// Sample
		let a = Array1::from_vec(a);
		let b = Array1::from_vec(b);
		let wts_a = Array1::from_vec(
			(kmin..=kmax)
				.map(|i| repeat_n(k_prior[&i] as f64, sz_a[i - kmin]).collect_vec())
				.concat(),
		);
		let wts_b = Array1::from_vec(
			(kmin..=kmax)
				.map(|i| repeat_n(k_prior[&i] as f64, sz_b[i - kmin]).collect_vec())
				.concat(),
		);
		let (delta1, alpha, beta) = {
			if a.is_empty() {
				(1.0, 1.0, 1.0)
			} else {
				let (delta1, _) = fit_gamma(&a, &wts_a, 1000, None);
				let alpha = delta1
					* (kmin..=kmax)
						.map(|i| k_prior[&i] * sz_a[i - kmin] as f64)
						.sum::<f64>();
				let beta = a.dot(&wts_a);
				(delta1, alpha, beta)
			}
		};
		let (delta2, zeta, gamma) = {
			if b.is_empty() {
				(1.0, 1.0, 1.0)
			} else {
				let (delta2, _) = fit_gamma(&b, &wts_b, 1000, None);
				let zeta = delta2
					* (kmin..=kmax)
						.map(|i| k_prior[&i] * sz_b[i - kmin] as f64)
						.sum::<f64>();
				let gamma = b.dot(&wts_b);
				(delta2, zeta, gamma)
			}
		};

		res.repulsion = repulsion;
		(|| -> Result<(PriorHyperParams, MCMCState)> {
			res.set_delta1(delta1)?
				.set_delta2(delta2)?
				.set_delta2(delta2)?
				.set_alpha(alpha)?
				.set_beta(beta)?
				.set_zeta(zeta)?
				.set_gamma(gamma)?
				.set_eta(eta)?
				.set_sigma(sigma)?
				.set_proposalsd_r(proposalsd_r)?
				.set_u(u)?
				.set_v(v)?;
			Ok((res, init_state))
		})()
		.map_err(|e| anyhow!("Error fitting prior: {}", e))
	}
}

pub(crate) fn pre_sample_rp(
	clust_labels: &[ClusterLabel],
	params: &PriorHyperParams,
	options: &MCMCOptions,
	rng: &mut SmallRng,
) -> Result<(Array1<f64>, Array1<f64>)> {
	let r = params.sample_r(1, rng)[0];
	let p = params.sample_r(1, rng)[0];
	let mut state = MCMCState::new(clust_labels.to_owned(), r, p)?;
	let n_samples = options.n_samples();
	let n_iter = options.n_iter;
	let n_burnin = options.n_burnin;
	let thinning = options.thinning;
	let (mut r_samples, mut p_samples) =
		(Vec::with_capacity(n_samples), Vec::with_capacity(n_samples));
	if n_samples != 0 {
		for _ in 0..n_burnin {
			state.sample_r(params, rng).sample_p(params, rng);
		}
		for i in n_burnin..n_iter {
			state.sample_r(params, rng).sample_p(params, rng);
			if (i - n_burnin) % thinning == 0 {
				r_samples.push(state.r);
				p_samples.push(state.p);
			}
		}
	}
	Ok((Array1::from_vec(r_samples), Array1::from_vec(p_samples)))
}

fn fit_gamma(x: &Array1<f64>, wts: &Array1<f64>, max_iter: usize, tol: Option<f64>) -> (f64, f64) {
	// https://github.com/JuliaStats/Distributions.jl/blob/master/src/univariate/continuous/gamma.jl
	let sx = x.dot(wts);
	let slogx = x.mapv(|x| x.ln()).dot(wts);
	let tw = wts.sum();

	let mx = sx / tw;
	let logmx = mx.ln();
	let mlogx = slogx / tw;
	let mut a = (logmx - mlogx) / 2.0;

	let tol = tol.unwrap_or(1e-16);

	for _ in 0..max_iter {
		let a_old = a;
		a = {
			let ia = 1.0 / a;
			let z = ia + (mlogx - logmx + a.ln() - a.digamma()) / (a.powi(2) * (ia - a.trigamma()));
			1.0 / z
		};
		if (a - a_old).abs() <= tol {
			break;
		}
	}
	(a, mx / a)
}

fn fit_beta(x: &Array1<f64>, max_iter: usize, tol: Option<f64>) -> (f64, f64) {
	// https://github.com/JuliaStats/Distributions.jl/blob/master/src/univariate/continuous/beta.jl
	let tol = tol.unwrap_or(1e-16);
	let (a, b) = {
		let mean = x.mean().unwrap();
		let var = x.var(0.0);
		let temp = (mean * (1.0 - mean) / var) - 1.0;
		(mean * temp, (1.0 - mean) * temp)
	};
	let g1 = x.mapv(|x| x.ln()).mean().unwrap();
	let g2 = x.mapv(|x| (1.0 - x).ln()).mean().unwrap();
	let mut theta = array![a, b];
	for _ in 0..max_iter {
		let temp1 = (theta[0] + theta[1]).digamma();
		let temp2 = (theta[0] + theta[1]).trigamma();
		let grad = array![
			g1 + temp1 - theta[0].digamma(),
			temp1 + g2 - theta[1].digamma()
		];
		let hess = array![
			[temp2 - theta[0].trigamma(), temp2],
			[temp2, temp2 - theta[1].trigamma()]
		];
		let delta_theta = hess.solveh_into(grad).unwrap();
		theta -= &delta_theta;
		if delta_theta.mapv(|x| x.powi(2)).sum() < 2.0 * tol {
			break;
		}
	}
	//
	(theta[0], theta[1])
}

fn pmf<T>(v: &[T]) -> HashMap<T, f64>
where
	T: std::hash::Hash + Eq + Copy,
{
	v.iter()
		.counts()
		.into_iter()
		.map(|(k, count)| (*k, count as f64 / v.len() as f64))
		.collect()
}
