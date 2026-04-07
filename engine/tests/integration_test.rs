use gretzky_engine::gretzky_version;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::RngCore;

#[test]
fn test_version_from_integration() {
    assert_eq!(gretzky_version(), "0.1.0");
}

#[test]
fn test_chacha8_determinism_1000_draws() {
    let mut a = ChaCha8Rng::seed_from_u64(8675309);
    let mut b = ChaCha8Rng::seed_from_u64(8675309);
    for _ in 0..1000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
