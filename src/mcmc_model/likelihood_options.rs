use std::{
	fmt::{Debug, Display, Formatter},
	num::NonZeroUsize,
	ops::RangeInclusive,
};

use accessory::Accessors;
use anyhow::{Result, anyhow};
#[cfg(feature = "python-module")]
use pyo3::{prelude::*, types::PyDict};

/// Parameters that modify the underlying clustering algorithm.
#[derive(Debug, Clone, Accessors, PartialEq)]
#[access(get, defaults(get(cp)))]
#[cfg_attr(feature = "python-module", pyclass(get_all, str, eq))]
pub struct LikelihoodOptions {
	/// Whether to use repulsive terms in the likelihood function when
	/// clustering.
	#[access(set)]
	repulsion: bool,

	/// Minimum number of clusters to allow in the clustering.
	min_num_clusts: NonZeroUsize,

	/// Maximum number of clusters to allow in the clustering.
	max_num_clusts: NonZeroUsize,
}

impl Default for LikelihoodOptions {
	fn default() -> Self {
		Self {
			repulsion: true,
			min_num_clusts: NonZeroUsize::new(1).unwrap(),
			max_num_clusts: NonZeroUsize::new(usize::MAX).unwrap(),
		}
	}
}

impl Display for LikelihoodOptions {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
		write!(
			f,
			"repulsion: {},\nmin_num_clusts: {:#?}\nmax_num_clusts: {:#?}\n}}",
			self.repulsion, self.min_num_clusts, self.max_num_clusts,
		)
	}
}

impl LikelihoodOptions {
	/// Set the range of allowed values for the number of clusters.
	pub fn set_range_num_clusts(
		&mut self,
		range_num_clusts: RangeInclusive<NonZeroUsize>,
	) -> Result<&mut Self> {
		if range_num_clusts.is_empty() {
			return Err(anyhow!("Range must be non-empty"));
		}
		self.min_num_clusts = *range_num_clusts.start();
		self.max_num_clusts = *range_num_clusts.end();
		Ok(self)
	}

	/// Set the range of allowed values for the number of clusters.
	pub fn with_range_num_clusts(
		mut self,
		range_num_clusts: RangeInclusive<NonZeroUsize>,
	) -> Result<Self> {
		self.set_range_num_clusts(range_num_clusts)?;
		Ok(self)
	}
}

#[cfg(feature = "python-module")]
#[pymethods]
impl LikelihoodOptions {
	/// Set the range of allowed values for the number of clusters.
	#[pyo3(name = "set_range_num_clusts")]
	fn py_set_range_num_clusts(
		&mut self,
		min_num_clusts: NonZeroUsize,
		max_num_clusts: NonZeroUsize,
	) -> Result<()> {
		self.set_range_num_clusts(min_num_clusts..=max_num_clusts)?;
		Ok(())
	}

	fn as_dict(this: Bound<'_, Self>) -> Result<Bound<'_, PyDict>> {
		let this = this.borrow();
		let dict: Bound<'_, PyDict> = PyDict::new(this.py());
		dict.set_item("repulsion", this.repulsion)?;
		dict.set_item("min_num_clusts", this.min_num_clusts)?;
		dict.set_item("max_num_clusts", this.max_num_clusts)?;
		Ok(dict)
	}
}
