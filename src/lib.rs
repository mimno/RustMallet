pub mod corpus;
pub mod model;
pub mod optimize;
pub mod output;
pub mod sampler;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _rust_mallet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
