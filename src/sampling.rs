use anyhow::Result;
use itertools::{Itertools, izip};
use ndarray::ArrayView1;
use rand::{Rng, distributions::Distribution};
use statrs::{
	distribution::{Beta, Continuous, LogNormal, Uniform},
	function::gamma::ln_gamma,
};

use crate::{
	MCMCData,
	types::{ClusterLabel, MCMCState, PriorHyperParams},
	utils::{row_sum, sample_from_ln_probs},
};

impl MCMCState {
	/// Sample p from its conditional posterior.
	pub(crate) fn sample_p_conditional<R: Rng>(
		&mut self,
		params: &PriorHyperParams,
		rng: &mut R,
	) -> Result<&mut Self> {
		let n_clusts = self.n_clusts().get() as f64;
		let n_pts = self.clust_labels.len() as f64;
		self.p = Beta::new(
			n_pts - n_clusts + params.u(),
			self.r * n_clusts + params.v(),
		)?
		.sample(rng);
		Ok(self)
	}

	/// Sample r from its conditional posterior.
	pub(crate) fn sample_r_conditional<R: Rng>(
		&mut self,
		params: &PriorHyperParams,
		rng: &mut R,
	) -> Result<&mut Self> {
		// This is called in a hot loop so we take off the safety goggles.
		let r = self.r;
		let p = self.p;
		let n_clusts = self.n_clusts().get() as f64;
		let eta = params.eta();
		let sigma = params.sigma();
		let proposalsd_r = params.proposalsd_r();
		let clust_list = &self.clust_list;
		let clust_sizes = &self.clust_sizes;
		let proposal_distr = LogNormal::new(r, proposalsd_r).unwrap();
		let r_proposed = proposal_distr.sample(rng);
		let reverse_distr = LogNormal::new(r_proposed, proposalsd_r).unwrap();

		// Calculate the transition probability
		let mut ln_prob_proposal = (eta - 1.0) * r_proposed.ln()
			+ n_clusts * (r_proposed * (1.0 - p).ln() - ln_gamma(r_proposed))
			- r_proposed * sigma;
		let mut ln_prob_current =
			(eta - 1.0) * r.ln() + n_clusts * (r * (1.0 - p).ln() - ln_gamma(r)) - r * sigma;
		clust_list.iter().for_each(|&j| {
			ln_prob_proposal += ln_gamma(clust_sizes[j as usize] as f64 + r_proposed - 1.0);
			ln_prob_current += ln_gamma(clust_sizes[j as usize] as f64 + r - 1.0);
		});
		let ln_accept_prob = reverse_distr.ln_pdf(r) - proposal_distr.ln_pdf(r_proposed)
			+ ln_prob_proposal
			- ln_prob_current;
		if ln_accept_prob.is_nan() {
			return Err(anyhow::anyhow!("Encountered NaN"));
		}
		let toss = Uniform::new(0.0, 1.0).unwrap().sample(rng).ln();
		self.r_accepted = false;
		if toss < ln_accept_prob {
			self.r = r_proposed;
			self.r_accepted = true;
		}
		Ok(self)
	}

	pub(crate) fn sample_clusters_gibbs_restricted<R: Rng>(
		&mut self,
		data: &MCMCData,
		params: &PriorHyperParams,
		rng: &mut R,
	) -> Result<&mut Self> {
		let n_pts = self.clust_labels.len();
		todo!()
	}

	pub(crate) fn sample_clusters_gibbs<R: Rng>(
		&mut self,
		data: &MCMCData,
		params: &PriorHyperParams,
		rng: &mut R,
	) -> Result<&mut Self> {
		let n_pts = self.clust_labels.len();

		// Pre-compute some quantities to speed up subsequent calculations.
		let alpha_beta_ratio = params.alpha() * params.beta().ln() - ln_gamma(params.alpha());
		let zeta_gamma_ratio = params.zeta() * params.gamma().ln() - ln_gamma(params.zeta());
		let ln_gamma_delta1 = ln_gamma(params.delta1());
		let ln_gamma_delta2 = ln_gamma(params.delta2());
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
			let current_label = &mut clust_labels[point_idx];

			// We cannot move the ith point to a different cluster if it is in a singleton
			// cluster and we are already at the minimum number of clusters.
			if clust_list.len() == params.n_clusts_range().start().get()
				&& clust_sizes[*current_label as usize] == 1
			{
				continue;
			}

			// Remove the ith point from the clustering
			clust_sizes[*current_label as usize] -= 1;
			if clust_sizes[*current_label as usize] == 0 {
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
				let elems_clust_k = clust_labels.iter().positions(|&x| x == k).collect_vec();
				let sz_k = clust_sizes[k as usize] as f64;
				let alpha_ik = params.alpha() + params.delta1() * sz_k;
				let beta_ik = params.beta() + row_sum(data.diss_mat(), point_idx, &elems_clust_k);
				let zeta_ik = params.zeta() + params.delta2() * sz_k;
				let gamma_ik = params.gamma() + row_sum(data.diss_mat(), point_idx, &elems_clust_k);
				let sum_ln_diss_ik = row_sum(data.ln_diss_mat(), point_idx, &elems_clust_k);

				let l1_ik = ln_gamma(alpha_ik) - alpha_ik * beta_ik.ln()
					+ alpha_beta_ratio
					+ (params.delta1() - 1.0) * sum_ln_diss_ik
					- sz_k * ln_gamma_delta1;
				*ln_prob_k =
					(sz_k + 1.0).ln() + ln_p + (sz_k - 1.0 + self.r).ln() - sz_k.ln() + l1_ik;

				*l3_ik = ln_gamma(zeta_ik) - zeta_ik * gamma_ik.ln()
					+ zeta_gamma_ratio
					+ (params.delta2() - 1.0) * sum_ln_diss_ik
					- sz_k * ln_gamma_delta2;
			}
			let sum_l3_ik: f64 = l3_i.iter().sum();
			for (l3_ik, ln_prob_k) in izip!(l3_i.iter(), ln_probs.iter_mut()) {
				*ln_prob_k += sum_l3_ik - *l3_ik;
			}

			// If we are under the maximum number of clusters, we can consider inserting a
			// new cluster
			if clust_list.len() < params.n_clusts_range().end().get() {
				candidate_clusts
					.push(clust_sizes.iter().position(|&x| x == 0).unwrap() as ClusterLabel);
				ln_probs.push((k_i + 1.0).ln() + self.r * ln_1_minus_p + sum_l3_ik);
			}

			let new_clust =
				candidate_clusts[sample_from_ln_probs(&ArrayView1::from(&ln_probs), rng)?];
			clust_labels[point_idx] = new_clust;
			clust_sizes[new_clust as usize] += 1;
			clust_list.insert(new_clust);
		}
		Ok(self)
	}
}
