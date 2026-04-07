use pyo3::prelude::*;

#[pyfunction]
pub fn gretzky_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn gretzky_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gretzky_version, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::RngCore;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_version_is_nonempty() {
        assert!(!gretzky_version().is_empty());
    }
    #[test]
    fn test_chacha8_reproducible() {
        let mut a = ChaCha8Rng::seed_from_u64(42);
        let mut b = ChaCha8Rng::seed_from_u64(42);
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
