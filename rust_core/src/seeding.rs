//! Deterministic stream separation for seeded RNGs.
//!
//! Seven modules each carried a private copy of this function. Six were
//! identical; `hurdle`'s combined its arguments with XOR rather than addition,
//! so one primitive had two behaviours and no test compared them. This is the
//! single definition.

/// Derive a stable, well-separated RNG seed for one chain of one stage.
///
/// `domain` keeps distinct stages (fitting, forecasting, posterior prediction)
/// on disjoint streams when a caller reuses one seed across them, which is the
/// common case because the stages share a default seed. The combination is
/// additive, so seeds deliberately offset by exactly a domain difference still
/// meet; ordinary seeds do not.
///
/// The body is the SplitMix64 finalizer, which spreads nearby inputs across
/// the whole 64-bit range.
pub fn chain_seed(seed: u64, chain_index: usize, domain: u64) -> u64 {
    let mut value = seed
        .wrapping_add(domain)
        .wrapping_add((chain_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Posterior-predictive draws taken from an already-fitted model.
///
/// `sampler::run` seeds chain `c` as `seed + c`, and posterior prediction
/// defaults to the same seed the fit defaulted to. Without a domain, a default
/// four-chain fit consumes streams 42..=45 and the predictive draws replay
/// chain 0's stream exactly.
pub const PREDICTIVE_SEED_DOMAIN: u64 = 0x5052_4544_5F4F_4253;

/// Prior-predictive draws, which use no fit at all.
///
/// Kept distinct from both fitting and posterior prediction so that checking a
/// prior and then fitting under the same seed does not replay one stream as
/// the other.
pub const PRIOR_PREDICTIVE_SEED_DOMAIN: u64 = 0x5052_494F_525F_5042;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domains_separate_streams_for_one_seed() {
        assert_ne!(
            chain_seed(42, 0, 0),
            chain_seed(42, 0, PREDICTIVE_SEED_DOMAIN)
        );
    }

    #[test]
    fn chains_separate_within_one_domain() {
        let seeds: Vec<u64> = (0..8).map(|c| chain_seed(42, c, 0)).collect();
        let mut unique = seeds.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), seeds.len());
    }

    #[test]
    fn predictive_stream_avoids_every_default_fit_chain() {
        // The collision this domain exists to prevent: `sampler::run` seeds
        // chain `c` as `seed + c`, so a default fit holds 42..=45.
        let predictive = chain_seed(42, 0, PREDICTIVE_SEED_DOMAIN);
        for chain in 0..64u64 {
            assert_ne!(predictive, 42u64.wrapping_add(chain));
        }
    }

    #[test]
    fn nearby_seeds_do_not_produce_nearby_streams() {
        let a = chain_seed(1, 0, 0);
        let b = chain_seed(2, 0, 0);
        assert!(a.abs_diff(b) > 1 << 32, "{a} and {b} are too close");
    }
}
