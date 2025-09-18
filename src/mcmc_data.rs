use std::{
	fmt::{Debug, Display, Formatter},
	num::NonZeroUsize,
};

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Zip};
use ndarray_linalg::Norm;
#[cfg(feature = "python-module")]
use numpy::{PyArray2, PyArrayMethods};
#[cfg(feature = "python-module")]
use pyo3::{prelude::*, types::PyType};

use crate::{
	ClusterLabel,
	utils::{Array1Wrapper, Array2Wrapper},
};

/// Struct to hold the dissimilarities matrix.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "python-module", pyclass(str))]
pub struct MCMCData {
	/// Dissimlarities matrix.
	dissimilarities: Array2Wrapper<f64>,

	/// Element-wise log of the dissimilarities matrix.
	ln_dissimilarities: Array2Wrapper<f64>,
}

impl MCMCData {
	/// Check whether the dissimilarity matrix is non-empty, square, and that
	/// the dissimilarities are symmetric positive-definite.
	fn validate_dissimilarities(diss_mat: Array2<f64>) -> Result<Array2<f64>> {
		if diss_mat.is_empty() {
			// empty
			return Err(anyhow!("Dissimilarity matrix must be non-empty."));
		}
		if diss_mat.nrows() != diss_mat.ncols() {
			// not square
			return Err(anyhow!(
				"Dissimilarity matrix must be square, but found shape {}x{}",
				diss_mat.nrows(),
				diss_mat.ncols()
			));
		}
		if Zip::from(&diss_mat).and(&diss_mat.t()).any(|&x, &y| x != y) {
			return Err(anyhow!("Dissimilarity matrix must be symmetric."));
		}
		if Zip::from(&diss_mat).any(|&x| x.is_nan()) {
			return Err(anyhow!("Found NaN in the dissimilarity matrix."));
		}
		if Zip::from(&diss_mat.diag()).any(|&x| x != 0.0) {
			return Err(anyhow!(
				"Dissimilarity matrix must have zeros along the diagonal."
			));
		}
		if diss_mat.iter().any(|x| *x < 0.0) {
			return Err(anyhow!("Off-diagonal entries must be strictly positive."));
		}
		Ok(diss_mat)
	}

	/// Create a new MCMCData instance with the specified dissimilarities
	/// matrix. The matrix must be non-empty, symmetric, with non-negative
	/// entries that are zero on the diagonal.
	pub fn from_dissimilarities(diss_mat: Array2<f64>) -> Result<Self> {
		Self::validate_dissimilarities(diss_mat)
			.and_then(|diss_mat| {
				let ln_diss_mat = diss_mat.mapv(|x| x.ln());
				if ln_diss_mat.iter().any(|x| x.is_nan()) {
					return Err(anyhow!("Found NaN in the log of the dissimilarity matrix."));
				}
				Ok(MCMCData {
					dissimilarities: Array2Wrapper(diss_mat),
					ln_dissimilarities: Array2Wrapper(ln_diss_mat),
				})
			})
			.map_err(|e| anyhow!("Error initialising MCMCData: {}", e))
	}

	/// Create a new MCMCData instance with the specified point cloud using the
	/// standard Euclidean 2-norm. The input should be a non-empty 2D array of
	/// shape (n_pts, n_dims).
	pub fn from_points(points: Array2<f64>) -> Result<Self> {
		let n_pts = points.nrows();
		if Zip::from(&points).any(|&x| x.is_nan()) {
			return Err(anyhow!("Found NaN entry in the points matrix."));
		}
		if n_pts == 0 {
			return Err(anyhow!("Point cloud cannot be empty"));
		}

		let diss_mat = {
			let temp = Array2::<f64>::from_shape_fn((n_pts, n_pts), |(i, j)| {
				if i <= j {
					0.0
				} else {
					(&points.row(i) - &points.row(j)).norm_l2()
				}
			});
			&temp + &temp.t()
		};
		Self::from_dissimilarities(diss_mat)
	}

	/// Get a reference to the dissimilarities matrix owned by this object.
	#[inline(always)]
	pub fn dissimilarities(&self) -> &Array2<f64> {
		&self.dissimilarities.0
	}

	/// Get a reference to the log-dissimilarities matrix owned by this object.
	#[inline(always)]
	pub(crate) fn ln_dissimilarities(&self) -> &Array2<f64> {
		&self.ln_dissimilarities.0
	}

	/// Number of points in the data.
	#[inline(always)]
	pub fn num_points(&self) -> NonZeroUsize {
		NonZeroUsize::new(self.dissimilarities.0.nrows()).unwrap()
	}

	/// Given cluster labels, return the set of all within-cluster
	/// dissimilarities.
	pub fn within_cluster_dissimilarities(
		&self,
		clust_labels: &[ClusterLabel],
	) -> Result<Array1<f64>> {
		if clust_labels.len() != self.dissimilarities.0.nrows() {
			return Err(anyhow!(
				"Expected {} cluster labels, but found {}",
				self.dissimilarities.0.nrows(),
				clust_labels.len()
			));
		}
		Ok(self
			.dissimilarities
			.0
			.indexed_iter()
			.filter_map(|((i, j), x)| {
				if i < j && clust_labels[i] == clust_labels[j] {
					Some(*x)
				} else {
					None
				}
			})
			.collect::<Array1<_>>())
	}

	/// Given the cluster labels, return the set of all inter-cluster
	/// dissimilarities.
	pub fn inter_cluster_dissimilarities(
		&self,
		clust_labels: &[ClusterLabel],
	) -> Result<Array1<f64>> {
		if clust_labels.len() != self.dissimilarities.0.nrows() {
			return Err(anyhow!(
				"Expected {} cluster labels, but found {}",
				self.dissimilarities.0.nrows(),
				clust_labels.len()
			));
		}
		Ok(self
			.dissimilarities
			.0
			.indexed_iter()
			.filter_map(|((i, j), x)| {
				if i < j && clust_labels[i] != clust_labels[j] {
					Some(*x)
				} else {
					None
				}
			})
			.collect::<Array1<_>>())
	}
}

impl Display for MCMCData {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
		write!(f, "MCMCData {{\ndiss_mat:\n{}\n}}", self.dissimilarities)
	}
}

#[cfg(feature = "python-module")]
#[pymethods]
impl MCMCData {
	/// Create a new MCMCData instance with the specified dissimilarities
	/// matrix. The matrix must be non-empty, symmetric, with non-negative
	/// entries that are zero on the diagonal.
	#[classmethod]
	#[pyo3(name = "from_dissimilarities")]
	fn py_from_dissimilarities(
		_: Bound<'_, PyType>,
		diss_mat: Bound<'_, PyArray2<f64>>,
	) -> Result<Self> {
		Self::from_dissimilarities(diss_mat.to_owned_array())
	}

	/// Create a new MCMCData instance with the specified point cloud using the
	/// standard Euclidean 2-norm. The input should be a non-empty 2D array of
	/// shape (n_pts, n_dims).
	#[classmethod]
	#[pyo3(name = "from_points")]
	fn py_from_points(_: Bound<'_, PyType>, points: Bound<'_, PyArray2<f64>>) -> Result<Self> {
		Self::from_points(points.to_owned_array())
	}

	/// Get a copy of the dissimilarities matrix.
	#[getter(dissimilarities)]
	fn py_get_dissimilarities(&self) -> &Array2Wrapper<f64> {
		&self.dissimilarities
	}

	/// Given cluster labels, return the set of all within-cluster
	/// dissimilarities.
	#[pyo3(name = "within_cluster_dissimilarities")]
	fn py_within_cluster_dissimilarities(
		&self,
		clust_labels: Vec<ClusterLabel>,
	) -> Result<Array1Wrapper<f64>> {
		self.within_cluster_dissimilarities(&clust_labels)
			.map(Array1Wrapper)
	}

	/// Given cluster labels, return the set of all inter-cluster
	/// dissimilarities.
	#[pyo3(name = "inter_cluster_dissimilarities")]
	fn py_inter_cluster_dissimilarities(
		&self,
		clust_labels: Vec<ClusterLabel>,
	) -> Result<Array1Wrapper<f64>> {
		self.inter_cluster_dissimilarities(&clust_labels)
			.map(Array1Wrapper)
	}

	fn __repr__(&self) -> String {
		format!("MCMCData(diss_mat={:#?})", self.dissimilarities.0)
	}
}
