use std::ops::AddAssign;

use anyhow::{Result, anyhow};
use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::SolveH;
use num_traits::identities::Zero;
use numpy::array;
use rand::{
	Rng,
	RngCore,
	SeedableRng,
	distributions::{Distribution, Uniform},
	rngs::OsRng,
};
use rand_xoshiro::Xoshiro256PlusPlus;
use special::Gamma;

/// Sum of the elements in a row of a matrix, given the row index and the column
/// indices to sum over.
/// Panics if the row index or any of the column indices are out of bounds.
pub fn row_sum(mat: &Array2<f64>, row: usize, cols: &[usize]) -> f64 {
	cols.iter().map(|&j| mat[(row, j)]).sum()
}

pub fn get_rng(rng_seed: Option<u64>) -> Xoshiro256PlusPlus {
	let seed = rng_seed.unwrap_or_else(|| OsRng.next_u64());
	Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// Sample from discrete distribution, in the form of a vector of
/// log-probabilities, using the Gumbel-max trick. Will return an error if the
/// vector contains any NaNs.
pub fn sample_from_ln_probs<R: Rng + ?Sized>(p: &ArrayView1<f64>, rng: &mut R) -> Result<usize> {
	let p = p - p.iter().fold(f64::INFINITY, |a, &b| a.min(b));
	let u = Uniform::from(0.0..=1.0)
		.sample_iter(rng)
		.take(p.len())
		.collect::<Array1<_>>();
	unsafe {
		float_vec_max(
			(-(-(u.ln())).ln() + p)
				.as_slice()
				// safe to unwrap if p is contiguous or in standard order, which
				// we guarantee because we create p in this function
				.unwrap_unchecked(),
		)
		.map(|(idx, _val)| idx)
	}
}

pub fn num_pairs(n: u64) -> u64 {
	if n < 2 {
		return 0;
	}
	n * (n - 1) / 2
}

/// Sum of a submatrix in a symmetric matrix, spanned by the given row and
/// column indices.
pub fn symm_mat_sum<T: Copy + AddAssign + Zero>(
	mat: &Array2<T>,
	rows: &[usize],
	cols: &[usize],
) -> T {
	let mut sum = T::zero();
	for row in rows {
		for col in cols {
			sum += mat[(*row, *col)];
		}
	}
	sum
}

/// Find the index and value of the maximum element in a vector of f64 values.
pub fn float_vec_max(v: &[f64]) -> Result<(usize, f64)> {
	if v.is_empty() {
		return Err(anyhow!("Empty vector"));
	}
	if v.iter().find_position(|x| x.is_nan()).is_some() {
		return Err(anyhow!("Encountered NaN"));
	}
	Ok(float_vec_max_unchecked(v))
}

/// Same as float_vec_max, but does not check for NaN or empty vector.
pub fn float_vec_max_unchecked(v: &[f64]) -> (usize, f64) {
	v.iter()
		.enumerate()
		.fold((0, f64::NEG_INFINITY), |(i, max), (j, &val)| {
			if val > max { (j, val) } else { (i, max) }
		})
}

fn forward_diffs(v: &[f64]) -> Vec<f64> {
	let n = v.len();
	if n == 0 {
		return vec![];
	} else if n == 1 {
		return vec![0.0];
	}
	let mut diffs = vec![0.0; n];
	for i in 0..n - 1 {
		diffs[i] = v[i + 1] - v[i];
	}
	diffs[n - 1] = diffs[n - 2];
	diffs
}

fn backward_diffs(v: &[f64]) -> Vec<f64> {
	let n = v.len();
	if n == 0 {
		return vec![];
	} else if n == 1 {
		return vec![0.0];
	}
	let mut diffs = vec![0.0; n];
	for i in 1..n {
		diffs[i] = v[i] - v[i - 1];
	}
	diffs[0] = diffs[1];
	diffs
}

/// Find the index of the knee point in a vector of f64 values.
/// The knee point is defined as the point where the second derivative
/// is maximized, which is approximated by the maximum of the
/// backward differences of the forward differences.
pub fn knee_pos(vals: &[f64]) -> Result<usize> {
	if vals.is_empty() {
		return Err(anyhow!("Input vector is empty"));
	}
	let second_der = backward_diffs(&forward_diffs(vals));
	float_vec_max(&second_der).map(|(index, _value)| index)
}

pub fn fit_gamma_mle(
	x: &Array1<f64>,
	wts: &Array1<f64>,
	max_iter: usize,
	tol: Option<f64>,
) -> Result<(f64, f64)> {
	// https://github.com/JuliaStats/Distributions.jl/blob/master/src/univariate/continuous/gamma.jl
	let sx = x.dot(wts);
	let slogx = x.mapv(|x| x.ln()).dot(wts);
	let tw = wts.sum();

	let mx = sx / tw;
	let logmx = mx.ln();
	let mlogx = slogx / tw;
	let mut a = (logmx - mlogx) / 2.0;

	let tol = tol.unwrap_or(1e-16);

	let mut iter = 0;
	while iter < max_iter {
		let a_old = a;
		a = {
			let ia = 1.0 / a;
			let z = ia + (mlogx - logmx + a.ln() - a.digamma()) / (a.powi(2) * (ia - a.trigamma()));
			1.0 / z
		};
		if (a - a_old).abs() <= tol {
			break;
		}
		iter += 1;
	}
	if iter == max_iter {
		return Err(anyhow!("MLE failed to converge!"));
	}
	if a.is_nan() || a <= 0.0 {
		return Err(anyhow!("Numerical instability in the MLE, got {a:?}."));
	}
	Ok((a, mx / a))
}

pub fn fit_beta_mle(x: &Array1<f64>, max_iter: usize, tol: Option<f64>) -> Result<(f64, f64)> {
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
	let mut iter = 0;
	while iter < max_iter {
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
		iter += 1;
	}
	if iter == max_iter {
		return Err(anyhow!("MLE failed to converge!"));
	}
	if theta.iter().any(|&x| x.is_nan() || x <= 0.0) {
		return Err(anyhow!(
			"Numerical instability in the MLE, got ({:?}, {:?}).",
			theta[0],
			theta[1]
		));
	}
	Ok((theta[0], theta[1]))
}

pub fn pmf(v: &[usize], kmax: usize) -> Vec<f64>
where
{
	let n = v.len() as f64;
	let mut pmf = vec![0.0; kmax + 1];
	v.iter().copied().for_each(|k| pmf[k] += 1.0);
	pmf.iter_mut().for_each(|x| *x /= n);
	pmf
}
