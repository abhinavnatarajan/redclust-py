use std::{
	fmt::{Debug, Display, Formatter},
	num::NonZeroUsize,
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Zip};
use ndarray_linalg::Norm;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::{prelude::*, types::PyType};

use super::ClusterLabel;
use crate::types::Array2Wrapper;

/// Struct to hold the dissimilarities matrix.
#[derive(Debug, Clone, Accessors, PartialEq)]
#[pyclass(str)]
pub struct MCMCData {
	/// Dissimlarities matrix.
	pub(crate) diss_mat: Array2Wrapper<f64>,

	/// Element-wise log of the dissimilarities matrix.
	pub(crate) ln_diss_mat: Array2Wrapper<f64>,
}

impl MCMCData {
	/// Check whether a matrix is nonempty, and has symmetric non-negative
	/// entries that are zero on the diagonal.
	fn validate_diss_mat(diss_mat: Array2<f64>) -> Result<Array2<f64>> {
		if diss_mat.nrows() != diss_mat.ncols() // not square
			|| diss_mat.is_empty() // empty
			|| Zip::from(&diss_mat).any(|&x| x < 0.0)
			|| Zip::from(&diss_mat.diag()).any(|&x| x != 0.0)
			|| Zip::from(&diss_mat).and(&diss_mat.t()).any(|&x, &y| x != y)
		{
			return Err(anyhow!(
				"Dissimilarities must be non-empty, and symmetric with nonnegative entries and \
				 zero diagonal."
			));
		}
		Ok(diss_mat)
	}

	/// Create a new MCMCData instance with the specified dissimilarities
	/// matrix. The matrix must be non-empty, symmetric, with non-negative
	/// entries that are zero on the diagonal.
	pub fn from_diss_mat(diss_mat: Array2<f64>) -> Result<Self> {
		Self::validate_diss_mat(diss_mat)
			.map(|diss_mat| {
				let ln_diss_mat = diss_mat.mapv(|x| x.ln());
				MCMCData {
					diss_mat: Array2Wrapper(diss_mat),
					ln_diss_mat: Array2Wrapper(ln_diss_mat),
				}
			})
			.map_err(|e| anyhow!("Error initialising MCMCData: {}", e))
	}

	/// Create a new MCMCData instance with the specified point cloud using the
	/// standard Euclidean 2-norm. The input should be a non-empty 2D array of
	/// shape (n_pts, n_dims).
	pub fn from_points(points: Array2<f64>) -> Result<Self> {
		let n_pts = points.nrows();
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

		let ln_diss_mat = diss_mat.mapv(|x| x.ln());
		Ok(MCMCData {
			diss_mat: Array2Wrapper(diss_mat),
			ln_diss_mat: Array2Wrapper(ln_diss_mat),
		})
	}

	/// Update the dissimilarities matrix. The new matrix must be non-empty,
	/// symmetric, with non-negative entries that are zero on the diagonal.
	pub fn set_diss_mat(&mut self, diss_mat: Array2<f64>) -> Result<&mut Self> {
		Self::validate_diss_mat(diss_mat)
			.map(|diss_mat| {
				self.diss_mat = Array2Wrapper(diss_mat);
				self.ln_diss_mat = Array2Wrapper(self.diss_mat.0.mapv(|x| x.ln()));
				self
			})
			.map_err(|e| anyhow!("Error setting dissimilarities matrix: {}", e))
	}

	/// Get a reference to the dissimilarities matrix owned by this object.
	#[inline(always)]
	pub fn diss_mat(&self) -> &Array2<f64> { &self.diss_mat.0 }

	/// Number of points in the data.
	#[inline(always)]
	pub fn n_pts(&self) -> NonZeroUsize {
		unsafe { NonZeroUsize::new_unchecked(self.diss_mat.0.nrows()) }
	}

	/// Given cluster labels, return the set of all within-cluster
	/// dissimilarities.
	pub fn within_cluster_dists(&self, clust_labels: &[ClusterLabel]) -> Result<Array1<f64>> {
		if clust_labels.len() != self.diss_mat.0.nrows() {
			return Err(anyhow!(
				"Expected {} cluster labels, but found {}",
				self.diss_mat.0.nrows(),
				clust_labels.len()
			));
		}
		Ok(self
			.diss_mat
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
	pub fn inter_cluster_dists(&self, clust_labels: &[ClusterLabel]) -> Result<Array1<f64>> {
		if clust_labels.len() != self.diss_mat.0.nrows() {
			return Err(anyhow!(
				"Expected {} cluster labels, but found {}",
				self.diss_mat.0.nrows(),
				clust_labels.len()
			));
		}
		Ok(self
			.diss_mat
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

#[pymethods]
impl MCMCData {
	/// Create a new MCMCData instance with the specified dissimilarities
	/// matrix. The matrix must be non-empty, symmetric, with non-negative
	/// entries that are zero on the diagonal.
	#[classmethod]
	#[pyo3(name = "from_diss_mat")]
	fn py_from_diss_mat(
		_cls: Bound<'_, PyType>,
		diss_mat: Bound<'_, PyArray2<f64>>,
	) -> Result<Self> {
		Self::from_diss_mat(diss_mat.to_owned_array())
	}

	/// Create a new MCMCData instance with the specified point cloud using the
	/// standard Euclidean 2-norm. The input should be a non-empty 2D array of
	/// shape (n_pts, n_dims).
	#[classmethod]
	#[pyo3(name = "from_points")]
	fn py_from_points(_cls: Bound<'_, PyType>, points: Bound<'_, PyArray2<f64>>) -> Result<Self> {
		Self::from_points(points.to_owned_array())
	}

	/// Update the dissimilarities matrix. The new matrix must be non-empty,
	/// symmetric, with non-negative entries that are zero on the diagonal.
	#[setter(diss_mat)]
	fn py_set_diss_mat(&mut self, diss_mat: Bound<'_, PyArray2<f64>>) -> Result<()> {
		self.set_diss_mat(diss_mat.to_owned_array()).map(|_| ())
	}

	/// Get a copy of the dissimilarities matrix.
	#[getter(diss_mat)]
	fn py_get_dist_mat(this: Bound<'_, Self>) -> Bound<'_, PyArray2<f64>> {
		this.borrow().diss_mat.into_pyobject(this.py()).unwrap()
	}

	/// Given cluster labels, return the set of all within-cluster
	/// dissimilarities.
	#[pyo3(name = "within_cluster_dists")]
	fn py_within_cluster_dists(
		this: Bound<'_, Self>,
		clust_labels: Vec<ClusterLabel>,
	) -> Result<Bound<'_, PyArray1<f64>>> {
		this.borrow()
			.within_cluster_dists(&clust_labels)
			.map(|arr| PyArray1::from_owned_array(this.py(), arr))
	}

	/// Given cluster labels, return the set of all inter-cluster
	/// dissimilarities.
	#[pyo3(name = "inter_cluster_dists")]
	fn py_inter_cluster_dists(
		this: Bound<'_, Self>,
		clust_labels: Vec<ClusterLabel>,
	) -> Result<Bound<'_, PyArray1<f64>>> {
		this.borrow()
			.inter_cluster_dists(&clust_labels)
			.map(|arr| PyArray1::from_owned_array(this.py(), arr))
	}

	fn __repr__(&self) -> String {
		format!(
			"MCMCData(dis_mat={:#?}, ln_dis_mat={:#?})",
			self.diss_mat.0, self.ln_diss_mat.0
		)
	}
}

impl Display for MCMCData {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
		write!(
			f,
			"MCMCData {{\ndis_mat:\n{}\nln_dis_mat:\n{}\n}}",
			self.diss_mat, self.ln_diss_mat
		)
	}
}
