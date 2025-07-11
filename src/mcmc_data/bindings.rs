use anyhow::Result;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::{prelude::*, types::PyType};

use super::{ClusterLabel, MCMCData};

#[pymethods]
impl MCMCData {
	/// Create a new MCMCData instance with the specified dissimilarities
	/// matrix. The matrix must be non-empty, symmetric, with non-negative
	/// entries that are zero on the diagonal.
	#[classmethod]
	#[pyo3(name = "from_dissimilarities")]
	fn py_from_dissimilarities(
		_cls: Bound<'_, PyType>,
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
	fn py_get_dissimilarities(this: Bound<'_, Self>) -> Bound<'_, PyArray2<f64>> {
		(&this.borrow().dissimilarities)
			.into_pyobject(this.py())
			.unwrap()
	}

	/// Given cluster labels, return the set of all within-cluster
	/// dissimilarities.
	#[pyo3(name = "within_cluster_dissimilarities")]
	fn py_within_cluster_dissimilarities(
		this: Bound<'_, Self>,
		clust_labels: Vec<ClusterLabel>,
	) -> Result<Bound<'_, PyArray1<f64>>> {
		this.borrow()
			.within_cluster_dissimilarities(&clust_labels)
			.map(|arr| PyArray1::from_owned_array(this.py(), arr))
	}

	/// Given cluster labels, return the set of all inter-cluster
	/// dissimilarities.
	#[pyo3(name = "inter_cluster_dissimilarities")]
	fn py_inter_cluster_dissimilarities(
		this: Bound<'_, Self>,
		clust_labels: Vec<ClusterLabel>,
	) -> Result<Bound<'_, PyArray1<f64>>> {
		this.borrow()
			.inter_cluster_dissimilarities(&clust_labels)
			.map(|arr| PyArray1::from_owned_array(this.py(), arr))
	}

	fn __repr__(&self) -> String { format!("MCMCData(diss_mat={:#?})", self.dissimilarities.0) }
}
