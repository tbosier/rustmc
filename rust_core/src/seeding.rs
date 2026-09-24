//! Deterministic stream separation for seeded RNGs.
//!
//! Eight modules each carried a private copy of this function, seven named
//! `chain_seed` and one `seed_for`. Six were byte-identical; `hurdle`'s combined
//! its arguments with XOR rather than addition, so one primitive had two
//! behaviours and no test compared them. This is the single definition.

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

/// Domain separator for the chains of an HMC/NUTS fit through `sampler`,
/// `"SMPL_FIT"`.
pub const SAMPLER_FIT_SEED_DOMAIN: u64 = 0x534D_504C_5F46_4954;

/// Domain separator for the random starting points of those chains,
/// `"SMPLINIT"`.
pub const SAMPLER_INIT_SEED_DOMAIN: u64 = 0x534D_504C_494E_4954;

/// Domain separator for the posterior-predictive stream, `"PRED_GEN"`.
pub const POSTERIOR_PREDICT_SEED_DOMAIN: u64 = 0x5052_4544_5F47_454E;

/// Domain separator for a prior-predictive stream, `"PRIOR_GN"`.
pub const PRIOR_PREDICT_SEED_DOMAIN: u64 = 0x5052_494F_525F_474E;

/// Re-key a caller's seed into a named RNG stream.
///
/// A simulation that seeded a generator with the raw integer would replay
/// whichever stream some other stage keyed from that integer — and passing the
/// fit seed is exactly what a caller reaches for. Mixing a domain constant in
/// through the SplitMix64 finalizer separates the streams, the way
/// [`sampler`](crate::sampler) keys its chains through
/// [`SAMPLER_FIT_SEED_DOMAIN`] and the structural, hierarchical, hurdle and
/// dynamic-GLM fits separate their fit, forecast and prior-predictive streams.
/// The finalizer is a bijection, so distinct seeds still give distinct streams
/// within a domain.
///
/// This is [`chain_seed`] for a stage that has only one stream.
pub fn stream_seed(seed: u64, domain: u64) -> u64 {
    chain_seed(seed, 0, domain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domains_separate_streams_for_one_seed() {
        assert_ne!(
            chain_seed(42, 0, 0),
            chain_seed(42, 0, POSTERIOR_PREDICT_SEED_DOMAIN)
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
        let predictive = chain_seed(42, 0, POSTERIOR_PREDICT_SEED_DOMAIN);
        for chain in 0..64 {
            assert_ne!(predictive, chain_seed(42, chain, SAMPLER_FIT_SEED_DOMAIN));
            assert_ne!(predictive, chain_seed(42, chain, SAMPLER_INIT_SEED_DOMAIN));
        }
    }

    #[test]
    fn adjacent_fit_seeds_do_not_share_chains() {
        // `seed + chain` made seed 42's chain 1 the same stream as seed 43's
        // chain 0.
        for seed in [0u64, 42, u64::MAX] {
            for chain in 0..16 {
                for other in 0..16 {
                    assert_ne!(
                        chain_seed(seed, chain + 1, SAMPLER_FIT_SEED_DOMAIN),
                        chain_seed(seed.wrapping_add(1), other, SAMPLER_FIT_SEED_DOMAIN)
                    );
                }
            }
        }
    }

    #[test]
    fn nearby_seeds_do_not_produce_nearby_streams() {
        let a = chain_seed(1, 0, 0);
        let b = chain_seed(2, 0, 0);
        assert!(a.abs_diff(b) > 1 << 32, "{a} and {b} are too close");
    }
}
