use std::{collections::BTreeSet, mem, num::NonZeroUsize};

use anyhow::{Result, anyhow};
use itertools::{Itertools, izip};
use ndarray::{Array1, ArrayView1};
use rand::{Rng, distributions::Distribution};
use statrs::{
	distribution::{Beta, Continuous, LogNormal, Uniform},
	function::gamma::ln_gamma,
};

use crate::{
	ClusterLabel,
	LikelihoodOptions,
	InputData,
	MCMCOptions,
	PriorHyperParams,
	State,
	utils::{num_pairs, row_sum, sample_from_ln_probs, symm_mat_sum},
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StateHelper<'a> {
	/// Dissimilarity matrix and related data.
	data: &'a InputData,
	/// Number of points.
	n_pts: usize,
	/// Prior hyperparameters.
	prior_params: &'a PriorHyperParams,
	/// Options related to the likelihood.
	likelihood_options: &'a LikelihoodOptions,
	/// Options related to the MCMC like number of iterations, etc.
	mcmc_options: &'a MCMCOptions,
	/// Current state of the chain.
	pub(crate) state: State,
	/// Cluster sizes.
	clust_sizes: Vec<usize>,
	/// List of labels of non-empty clusters.
	nonempty_clusts: BTreeSet<ClusterLabel>,
	/// ln(beta^alpha / Gamma(alpha))
	alpha_beta_ratio: f64,
	/// ln(gamma^zeta/ Gamma(zeta))
	zeta_gamma_ratio: f64,
	/// ln(Gamma(delta_1))
	ln_gamma_delta1: f64,
	/// ln(Gamma(delta_2))
	ln_gamma_delta2: f64,
	/// Current value of ln(p)
	ln_p: f64,
	/// Quantity L^(3)_{i,k} defined in the supplementary material of the paper.
	sum_l3_ik: f64,
	/// Number of split-merge proposals that were accepted
	/// in the most recent label sampling step.
	pub(crate) splitmerge_accepted: usize,
	/// Whether the last proposal for the parameter r was accepted.
	pub(crate) r_accepted: bool,
}

impl<'a> StateHelper<'a> {
	pub(crate) fn new(
		data: &'a InputData,
		prior_params: &'a PriorHyperParams,
		likelihood_options: &'a LikelihoodOptions,
		mcmc_options: &'a MCMCOptions,
		state: State,
	) -> Self {
		// Pre-compute some quantities to speed up subsequent calculations.
		let alpha_beta_ratio =
			prior_params.alpha() * prior_params.beta().ln() - ln_gamma(prior_params.alpha());
		let zeta_gamma_ratio =
			prior_params.zeta() * prior_params.gamma().ln() - ln_gamma(prior_params.zeta());
		let ln_gamma_delta1 = ln_gamma(prior_params.delta1());
		let ln_gamma_delta2 = ln_gamma(prior_params.delta2());
		let ln_p = state.p.ln();
		let n_pts = state.clust_labels.len();
		let mut clust_sizes = vec![0; n_pts];
		let nonempty_clusts = state.clust_set();
		state
			.clust_labels
			.iter()
			.counts()
			.into_iter()
			.for_each(|(k, v)| {
				clust_sizes[*k as usize] = v;
			});
		Self {
			data,
			n_pts: data.num_points().get(),
			prior_params,
			likelihood_options,
			mcmc_options,
			state,
			clust_sizes,
			nonempty_clusts,
			alpha_beta_ratio,
			zeta_gamma_ratio,
			ln_gamma_delta1,
			ln_gamma_delta2,
			ln_p,
			sum_l3_ik: 0.0,
			splitmerge_accepted: 0,
			r_accepted: false,
		}
	}

	/// Sample p from its conditional posterior.
	pub(crate) fn sample_p_conditional<R: Rng>(&mut self, rng: &mut R) -> Result<&mut Self> {
		let n_clusts = self.num_clusts().get() as f64;
		let n_pts = self.n_pts as f64;
		let u = self.prior_params.u();
		let v = self.prior_params.v();
		let r = self.state.r();
		self.state.p = Beta::new(n_pts - n_clusts + u, r * n_clusts + v)?.sample(rng);
		self.ln_p = self.state.p.ln();
		Ok(self)
	}

	/// Sample r from its conditional posterior.
	pub(crate) fn sample_r_conditional<R: Rng>(&mut self, rng: &mut R) -> Result<&mut Self> {
		// This is called in a hot loop so we take off the safety goggles.
		let r = self.state.r;
		let p = self.state.p;
		let n_clusts = self.num_clusts().get() as f64;
		let eta = self.prior_params.eta();
		let sigma = self.prior_params.sigma();
		let proposalsd_r = self.prior_params.proposalsd_r();
		let nonempty_clusts = &self.nonempty_clusts;
		let clust_sizes = &self.clust_sizes;

		let proposal_distr = LogNormal::new(r, proposalsd_r)?;
		let r_proposed = proposal_distr.sample(rng);
		let reverse_distr = LogNormal::new(r_proposed, proposalsd_r)?;

		// Calculate the transition probability
		let mut ln_prob_proposal = (eta - 1.0) * r_proposed.ln()
			+ n_clusts * (r_proposed * (1.0 - p).ln() - ln_gamma(r_proposed))
			- r_proposed * sigma;
		let mut ln_prob_current =
			(eta - 1.0) * r.ln() + n_clusts * (r * (1.0 - p).ln() - ln_gamma(r)) - r * sigma;
		nonempty_clusts.iter().map(|j| *j as usize).for_each(|j| {
			ln_prob_proposal += ln_gamma(clust_sizes[j] as f64 + r_proposed - 1.0);
			ln_prob_current += ln_gamma(clust_sizes[j] as f64 + r - 1.0);
		});
		let ln_accept_prob = reverse_distr.ln_pdf(r) - proposal_distr.ln_pdf(r_proposed)
			+ ln_prob_proposal
			- ln_prob_current;
		if ln_accept_prob.is_nan() {
			return Err(anyhow::anyhow!("Encountered NaN"));
		}
		let toss = Uniform::new(0.0, 1.0)?.sample(rng).ln();
		self.r_accepted = false;
		if toss < ln_accept_prob {
			self.state.r = r_proposed;
			self.r_accepted = true;
		}
		Ok(self)
	}

	/// Remove the cluster assignment for a data point.
	/// Does nothing if the point is already unassigned.
	fn delete_point(&mut self, point_idx: usize) {
		let item_clust = &mut self.state.clust_labels[point_idx];
		if *item_clust != ClusterLabel::MAX {
			let old_clust_label = *item_clust;
			let old_clust_size = &mut self.clust_sizes[old_clust_label as usize];
			*old_clust_size -= 1;
			if *old_clust_size == 0 {
				self.nonempty_clusts.remove(&old_clust_label);
			}
			*item_clust = ClusterLabel::MAX;
		}
	}

	/// Updates the cluster assignment for a data point.
	fn update_clust(&mut self, point_idx: usize, new_clust_label: ClusterLabel) {
		if new_clust_label != ClusterLabel::MAX {
			let old_clust_label = self.state.clust_labels[point_idx];
			let old_clust_size = &mut self.clust_sizes[old_clust_label as usize];
			*old_clust_size -= 1;
			if *old_clust_size == 0 {
				self.nonempty_clusts.remove(&old_clust_label);
			}
			self.state.clust_labels[point_idx] = new_clust_label;
			self.clust_sizes[new_clust_label as usize] += 1;
			self.nonempty_clusts.insert(new_clust_label);
		} else {
			self.delete_point(point_idx);
		}
	}

	/// Find the first empty cluster label.
	fn first_empty_cluster(&self) -> Option<ClusterLabel> {
		self.clust_sizes
			.iter()
			.position(|&x| x == 0)
			.map(|x| x as ClusterLabel)
			.filter(|&x| x != ClusterLabel::MAX)
	}

	/// Number of clusters.
	#[inline(always)]
	pub fn num_clusts(&self) -> NonZeroUsize {
		NonZeroUsize::new(self.nonempty_clusts.len()).unwrap()
	}

	fn update_ln_probs(
		&mut self,
		point_idx: usize,
		candidate_clusts: &[ClusterLabel],
		ln_probs: &mut [f64],
		l3_i: &mut [f64],
	) {
		let state = &self.state;
		let clust_sizes = &self.clust_sizes;
		let prior_params = self.prior_params;
		let alpha = prior_params.alpha();
		let beta = prior_params.beta();
		let alpha_beta_ratio = self.alpha_beta_ratio;
		let delta1 = prior_params.delta1();
		let ln_gamma_delta1 = self.ln_gamma_delta1;
		let zeta = prior_params.zeta();
		let gamma = prior_params.gamma();
		let zeta_gamma_ratio = self.zeta_gamma_ratio;
		let delta2 = prior_params.delta2();
		let ln_gamma_delta2 = self.ln_gamma_delta2;
		let r = state.r;
		let ln_p = self.ln_p;
		let diss = self.data.dissimilarities();
		let ln_diss = self.data.ln_dissimilarities();
		let repulsion = self.likelihood_options.repulsion;

		// Compute cohesive likelihood component for each proposed cluster.
		for (&k, l3_ik, ln_prob_k) in izip!(
			candidate_clusts.iter(),
			l3_i.iter_mut(),
			ln_probs.iter_mut()
		) {
			let elems_clust_k = self.state.items_with_label(k);
			let beta_ik = beta + row_sum(diss, point_idx, &elems_clust_k);
			let sum_ln_diss_ik = row_sum(ln_diss, point_idx, &elems_clust_k);
			let sz_k = clust_sizes[k as usize] as f64;
			let alpha_ik = alpha + delta1 * sz_k;
			let l1_ik = ln_gamma(alpha_ik) - alpha_ik * beta_ik.ln()
				+ alpha_beta_ratio
				+ (delta1 - 1.0) * sum_ln_diss_ik
				- sz_k * ln_gamma_delta1;
			*ln_prob_k = (sz_k + 1.0).ln() + ln_p + (sz_k - 1.0 + r).ln() - sz_k.ln() + l1_ik;
			if repulsion {
				// Compute the repulsive part of the likelihood.
				let zeta_ik = zeta + delta2 * sz_k;
				let gamma_ik = gamma + row_sum(diss, point_idx, &elems_clust_k);
				*l3_ik = ln_gamma(zeta_ik) - zeta_ik * gamma_ik.ln()
					+ zeta_gamma_ratio
					+ (delta2 - 1.0) * sum_ln_diss_ik
					- sz_k * ln_gamma_delta2;
			}
		}
		// Add the repulsive part of the likelihood to the log-probabilities.
		// If repulsion is off, l3_i will be all zeros so this will have no effect.
		self.sum_l3_ik = l3_i.iter().sum();
		for (l3_ik, ln_prob_k) in izip!(l3_i.iter(), ln_probs.iter_mut()) {
			*ln_prob_k += self.sum_l3_ik - *l3_ik;
		}
	}

	pub(crate) fn sample_clusters_gibbs_restricted<R: Rng>(
		&mut self,
		items_to_reallocate: &[usize],
		candidate_clusts: &[ClusterLabel],
		final_clusts: Option<&[ClusterLabel]>,
		rng: &mut R,
	) -> Result<f64> {
		// Pre-compute some quantities to speed up subsequent calculations.
		let mut ln_probs = Array1::from_elem(2, 0.0);
		let mut l3_i = Array1::from_elem(2, 0.0);
		let mut ln_transition_prob = 0.0;

		for &point_idx in items_to_reallocate.iter() {
			self.delete_point(point_idx);
			ln_probs.fill(0.0);
			l3_i.fill(0.0);

			self.update_ln_probs(
				point_idx,
				candidate_clusts,
				ln_probs
					.as_slice_mut()
					.ok_or_else(|| anyhow!("Something went desperately wrong."))?,
				l3_i.as_slice_mut()
					.ok_or_else(|| anyhow!("Something went desperately wrong."))?,
			);

			// Assign the new cluster
			let new_clust: ClusterLabel;
			let choice: usize;
			if let Some(final_clusts_inner) = final_clusts {
				new_clust = final_clusts_inner[point_idx];
				choice = candidate_clusts
					.iter()
					.position(|&x| x == new_clust)
					.ok_or_else(|| {
						anyhow!(
							"final_clusts contained {} which is not in candidate_clusts.",
							new_clust
						)
					})?;
			} else {
				choice = sample_from_ln_probs(&ln_probs.view(), rng)?;
				new_clust = candidate_clusts[choice];
			}
			self.update_clust(point_idx, new_clust);

			// Calculate the transition probability
			ln_probs += ln_probs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
			let mut probs = ln_probs.exp();
			probs /= probs.sum();
			ln_transition_prob += probs[choice].ln();
		}
		Ok(ln_transition_prob)
	}

	pub(crate) fn sample_clusters_gibbs<R: Rng>(&mut self, rng: &mut R) -> Result<&mut Self> {
		let n_pts = self.n_pts;
		let r = self.state.r;
		let sum_l3_ik = self.sum_l3_ik;
		let ln_1_minus_p = (1.0 - self.state.p).ln();

		// The list of candidate clusters for each point.
		// Since we assume that there are at most `n_pts` clusters,
		// we can allocate a size of n_pts in advance,
		// instead of allocating a new vector for each point.
		let mut candidate_clusts = Vec::<ClusterLabel>::with_capacity(n_pts);

		// Pre-allocate space for vectors that represent mappings ClusterLabel -> f64.
		// Use sparse vectors instead of "dense" hashmaps for performance and
		// simplicity.
		let mut l3_i = Vec::<f64>::with_capacity(n_pts);
		let mut ln_probs = Vec::<f64>::with_capacity(n_pts);

		// Start
		for point_idx in 0..n_pts {
			let current_clust_sz = self.clust_sizes[self.state.clust_labels[point_idx] as usize];

			// We cannot move the ith point to a different cluster if it is in a singleton
			// cluster and we are already at the minimum number of clusters.
			if self.nonempty_clusts.len() == self.likelihood_options.min_num_clusts().get()
				&& current_clust_sz == 1
			{
				continue;
			}

			// Remove the ith point from the clustering
			self.delete_point(point_idx);

			// Number of clusters after removing the ith point.
			let k_i = self.num_clusts().get();

			// Update the candidate list.
			// clear + extend avoids re-allocation.
			candidate_clusts.clear();
			candidate_clusts.extend(self.nonempty_clusts.iter());
			// These will hold the log-probability for each candidate cluster.
			ln_probs.resize(k_i, 0.0);
			l3_i.resize(k_i, 0.0);

			self.update_ln_probs(
				point_idx,
				&candidate_clusts,
				ln_probs.as_mut_slice(),
				l3_i.as_mut_slice(),
			);

			// If we are under the maximum number of clusters, we can consider inserting a
			// new cluster.
			if k_i < self.likelihood_options.max_num_clusts().get() && k_i < n_pts {
				candidate_clusts.push(self.first_empty_cluster().ok_or_else(|| {
					anyhow!("Couldn't find an empty cluster when there should be one!")
				})?);
				ln_probs.push((k_i as f64 + 1.0).ln() + r * ln_1_minus_p + sum_l3_ik);
			}

			let new_clust = candidate_clusts
				[sample_from_ln_probs(&ArrayView1::from(ln_probs.as_slice()), rng)?];
			self.update_clust(point_idx, new_clust);
		}
		Ok(self)
	}

	pub(crate) fn sample_cluster_labels<R: Rng>(&mut self, rng: &mut R) -> Result<&mut Self> {
		let n_pts = self.n_pts;
		let ln_1_minus_p = (1.0 - self.state.p).ln();
		let ln_p = self.ln_p;
		let r = self.state.r;
		let likelihood_options = self.likelihood_options;
		let num_clusts = self.num_clusts().get();
		let max_num_clusts = likelihood_options.max_num_clusts().get();
		self.splitmerge_accepted = 0;

		let mut ln_lik_cur = self.ln_likelihood();
		for _ in 0..self.mcmc_options.num_mh_steps {
			let orig_num_clusts = self.num_clusts().get() as f64;
			let mut split = false;

			// Pick chaperones
			let range = 0..n_pts;
			let (i, j): (usize, usize) = (rng.gen_range(range.clone()), rng.gen_range(range));
			let ci = self.state.clust_labels[i];
			let cj = self.state.clust_labels[j];
			if ci == cj {
				split = true;
				if num_clusts >= max_num_clusts || num_clusts == n_pts {
					continue;
				}
			}
			let sz_ci = self.clust_sizes[ci as usize] as f64;
			let sz_cj = self.clust_sizes[cj as usize] as f64;
			let items_to_reallocate: Vec<usize> = self
				.state
				.items_with_label(ci)
				.into_iter()
				.chain(self.state.items_with_label(cj).into_iter())
				.filter(|&idx| idx != i && idx != j)
				.collect();

			// Init the launch state.
			let mut launch_state = self.clone();
			if split {
				// Split proposal, move point i to a new cluster.
				launch_state.update_clust(i, launch_state.first_empty_cluster().unwrap());
			}
			let candidate_clusts = [
				launch_state.state.clust_labels[i],
				launch_state.state.clust_labels[j],
			];
			for &item in items_to_reallocate.iter() {
				// Randomly distribute the points in clusters
				// ci and cj into the clusters occupied by i and j.
				launch_state.update_clust(item, candidate_clusts[rng.gen_range(0..2)]);
			}

			// Restricted Gibbs scans:
			// Reallocate the items in clusters ci and cj into
			// clusters occupied by i and j using the conditional
			// reallocation probabilities from the model.
			for _ in 0..self.mcmc_options.num_gibbs_passes {
				launch_state.sample_clusters_gibbs_restricted(
					&items_to_reallocate,
					&candidate_clusts,
					None,
					rng,
				)?;
			}

			let ln_proposal_ratio: f64;
			let ln_prior_ratio: f64;
			let mut final_state: StateHelper;
			if split {
				// Split proposal.
				ln_proposal_ratio = launch_state.sample_clusters_gibbs_restricted(
					&items_to_reallocate,
					&candidate_clusts,
					None,
					rng,
				)?;
				final_state = launch_state;
				let ci_final = final_state.state.clust_labels[i] as usize;
				let cj_final = final_state.state.clust_labels[j] as usize;
				let sz_ci_final = final_state.clust_sizes[ci_final] as f64;
				let sz_cj_final = final_state.clust_sizes[cj_final] as f64;
				ln_prior_ratio = (orig_num_clusts + 1.0).ln() + r * ln_1_minus_p
					- ln_p - ln_gamma(r)
					- ln_gamma(sz_ci - 1.0 + r)
					- sz_ci.ln() + ln_gamma(sz_ci_final - 1.0 + r)
					+ ln_gamma(sz_cj_final - 1.0 + r)
					+ sz_ci_final.ln()
					+ sz_cj_final.ln();
			} else {
				final_state = launch_state.clone();
				let clust_i = final_state.state.items_with_label(ci);
				clust_i
					.iter()
					.for_each(|&idx| final_state.update_clust(idx, cj));
				let sz_cj_final = final_state.clust_sizes[cj as usize] as f64;
				// TODO
				ln_prior_ratio = -(orig_num_clusts.ln() + r * ln_1_minus_p - ln_p - ln_gamma(r))
					+ ln_gamma(sz_cj_final - 1.0 + r)
					+ sz_cj_final.ln()
					- (ln_gamma(sz_ci - 1.0 + r)
						+ ln_gamma(sz_cj - 1.0 + r)
						+ sz_ci.ln() + sz_cj.ln());
				ln_proposal_ratio = -launch_state.sample_clusters_gibbs_restricted(
					&items_to_reallocate,
					&candidate_clusts,
					Some(&self.state.clust_labels),
					rng,
				)?;
			}
			let ln_lik_new = final_state.ln_likelihood();
			let ln_lik_ratio = ln_lik_new - ln_lik_cur;
			let ln_acceptance_ratio = [0.0, ln_prior_ratio + ln_lik_ratio - ln_proposal_ratio]
				.iter()
				.fold(f64::INFINITY, |a, &b| a.min(b));
			if Uniform::new(0.0, 1.0).unwrap().sample(rng).ln() < ln_acceptance_ratio {
				// Accept the proposal.
				mem::swap(self, &mut final_state);
				self.splitmerge_accepted += 1;
				ln_lik_cur = ln_lik_new;
			}
		}
		// Final Gibbs scan.
		self.sample_clusters_gibbs(rng)
	}

	/// Log likelihood of the current state.
	/// Faster than calling mcmc::ln_likelihood because it re-uses pre-computed
	/// quantities.
	pub(crate) fn ln_likelihood(&self) -> f64 {
		let nonempty_clusts = &self.nonempty_clusts.iter().copied().collect_vec();
		let diss_mat = self.data.dissimilarities();
		let ln_diss_mat = self.data.ln_dissimilarities();
		let alpha = self.prior_params.alpha();
		let beta = self.prior_params.beta();
		let alpha_beta_ratio = self.alpha_beta_ratio;
		let delta1 = self.prior_params.delta1();
		let lgamma_delta1 = self.ln_gamma_delta1;
		let repulsion = self.likelihood_options.repulsion;

		// Cohesive part of likelihood
		let lik_cohesive = nonempty_clusts
			.iter()
			.map(|k| {
				let clust_k = self.state.items_with_label(*k);
				let sz_k = clust_k.len();
				let num_pairs_k = num_pairs(sz_k as u64) as f64;
				let a = alpha + delta1 * num_pairs_k;
				let b = beta + symm_mat_sum(diss_mat, &clust_k, &clust_k);
				(delta1 - 1.0) * symm_mat_sum(ln_diss_mat, &clust_k, &clust_k) / 2.0
					- num_pairs_k * lgamma_delta1
					+ alpha_beta_ratio
					+ ln_gamma(a) - a * b.ln()
			})
			.sum::<f64>();

		if !repulsion {
			return lik_cohesive;
		};

		// Repulsive part of likelihood
		let zeta = self.prior_params.zeta();
		let gamma = self.prior_params.gamma();
		let zeta_gamma_ratio = self.zeta_gamma_ratio;
		let delta2 = self.prior_params.delta2();
		let lgamma_delta2 = ln_gamma(delta2);
		let n_clusts = self.num_clusts().get();
		let lik_repulsive = (0..n_clusts)
			.map(|k| {
				let clust_k = self.state.items_with_label(nonempty_clusts[k]);
				let sz_k = clust_k.len();
				(k + 1..n_clusts)
					.map(|t| {
						let clust_t = self.state.items_with_label(nonempty_clusts[t]);
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
}
