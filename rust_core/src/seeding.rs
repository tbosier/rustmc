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
/// common case because the stages share a default seed.
///
/// The three inputs enter in separate SplitMix64 finalizer rounds: the seed is
/// mixed, the domain is folded into the mixed value and mixed again, and the
/// chain's golden-ratio offset is added to that and mixed once more. Each
/// round is a bijection, so distinct chains of one seed and domain never share
/// a stream. The previous form added all three before a single round, so
/// chain `c` of seed `s` was chain 0 of seed `s + c * 0x9E37_79B9_7F4A_7C15`,
/// and two domains met at seeds offset by their difference; mixing each input
/// before the next is combined leaves no such offset. Different seeds can
/// still share a chain: for a fixed chain index and domain the map from seed
/// to stream is a bijection onto all of `u64`, so chain `c > 0` of a seed is
/// chain 0 of exactly one other seed (chain 1 of seed 14758518203450600995 is
/// chain 0 of seed 42), found only by inverting the mixing rounds rather than
/// by any simple relation between the seeds.
pub fn chain_seed(seed: u64, chain_index: usize, domain: u64) -> u64 {
    let keyed = splitmix64_finalize(splitmix64_finalize(seed) ^ domain);
    splitmix64_finalize(keyed.wrapping_add((chain_index as u64).wrapping_mul(GOLDEN_GAMMA)))
}

/// SplitMix64's increment, the odd integer nearest `2^64 / phi`.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The SplitMix64 output function: a bijection on `u64` that spreads nearby
/// inputs across the whole range.
fn splitmix64_finalize(mut value: u64) -> u64 {
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
    fn some_other_seed_always_shares_a_chain() {
        // The documented counterexample: seeds are not a partition of streams.
        assert_eq!(
            chain_seed(14758518203450600995, 1, SAMPLER_FIT_SEED_DOMAIN),
            chain_seed(42, 0, SAMPLER_FIT_SEED_DOMAIN)
        );
    }

    #[test]
    fn no_seed_offset_reproduces_another_chain_or_domain() {
        // Under the additive form each of these pairs was the same stream.
        for seed in [0u64, 1, 42, 1 << 40, u64::MAX] {
            for chain in 1..64usize {
                let offset = (chain as u64).wrapping_mul(GOLDEN_GAMMA);
                assert_ne!(
                    chain_seed(seed, chain, SAMPLER_FIT_SEED_DOMAIN),
                    chain_seed(seed.wrapping_add(offset), 0, SAMPLER_FIT_SEED_DOMAIN)
                );
            }
            let shift = POSTERIOR_PREDICT_SEED_DOMAIN.wrapping_sub(PRIOR_PREDICT_SEED_DOMAIN);
            assert_ne!(
                chain_seed(seed.wrapping_add(shift), 0, PRIOR_PREDICT_SEED_DOMAIN),
                chain_seed(seed, 0, POSTERIOR_PREDICT_SEED_DOMAIN)
            );
            let flip = POSTERIOR_PREDICT_SEED_DOMAIN ^ PRIOR_PREDICT_SEED_DOMAIN;
            assert_ne!(
                chain_seed(seed ^ flip, 0, PRIOR_PREDICT_SEED_DOMAIN),
                chain_seed(seed, 0, POSTERIOR_PREDICT_SEED_DOMAIN)
            );
        }
    }

    #[test]
    fn nearby_seeds_do_not_produce_nearby_streams() {
        let a = chain_seed(1, 0, 0);
        let b = chain_seed(2, 0, 0);
        assert!(a.abs_diff(b) > 1 << 32, "{a} and {b} are too close");
    }
}
