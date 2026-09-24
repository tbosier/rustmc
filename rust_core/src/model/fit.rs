//! A fitted graph model and everything computed from its posterior draws:
//! posterior prediction, pointwise log-likelihood and deterministics.
//!
//! The Python `FitResult` is a thin adapter over [`ModelFit`], so a fit made
//! in either language predicts, scores and persists through this one
//! implementation.
use crate::autodiff::Evaluator;
use crate::data::{DataBinding, DataInputs};
use crate::graph::{Graph, ObservationHead, ParamTransform};
use crate::sampler::SampleResult;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use super::*;

/// Posterior predictions indexed by response, chain, draw, and observation.
pub type Prediction = HashMap<String, Vec<Vec<Vec<f64>>>>;

/// Posterior draws evaluated per worker: large enough to amortise one
/// evaluator's allocation, small enough to spread a single chain over a pool.
const EVALUATION_CHUNK: usize = 64;

#[derive(Clone, Debug)]
pub struct ModelFit {
    model: GraphModel,
    binding: DataBinding,
    raw: Arc<SampleResult>,
    /// Draws in the reported parameter order. When no parameter is derived
    /// this is the same allocation as the raw draws, not a copy.
    pub samples: Arc<SampleResult>,
}

/// Simulated observations at a set of posterior draws.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictiveDraws {
    /// The posterior `(chain, draw)` each row was simulated from, in row order.
    pub coordinates: Vec<(usize, usize)>,
    /// Observations per row, for each likelihood in `likelihood_names` order.
    pub n_obs: Vec<usize>,
    /// One row-major `coordinates.len() × n_obs[i]` block per likelihood.
    pub values: Vec<Vec<f64>>,
}

/// One deterministic evaluated at every posterior draw.
#[derive(Clone, Debug, PartialEq)]
pub struct DeterministicDraws {
    pub name: String,
    /// Elements per draw; zero for a scalar.
    pub len: usize,
    /// Row-major `(chain, draw[, element])` values.
    pub values: Vec<f64>,
}

impl ModelFit {
    /// Assemble a fit from a model, the data it was fitted to and the
    /// sampler's output, deriving the reported parameters.
    pub fn new(model: GraphModel, binding: DataBinding, raw: SampleResult) -> ModelResult<Self> {
        Self::from_shared(model, binding, Arc::new(raw))
    }

    pub(crate) fn from_shared(
        model: GraphModel,
        binding: DataBinding,
        raw: Arc<SampleResult>,
    ) -> ModelResult<Self> {
        if raw.param_names != model.structure.param_names {
            return Err(ModelError::invalid(
                "posterior parameter names/order differ from the model",
            ));
        }
        binding
            .validate_for(&model.structure)
            .map_err(|e| ModelError::invalid(e.to_string()))?;
        let samples = display_sample_result(&raw, &model.display_params)?;
        Ok(Self {
            model,
            binding,
            raw,
            samples,
        })
    }

    pub fn model(&self) -> &GraphModel {
        &self.model
    }

    /// The data the model was fitted to.
    pub fn binding(&self) -> &DataBinding {
        &self.binding
    }

    /// Draws of the sampled (graph) parameters, before display derivation.
    pub fn raw_samples(&self) -> &Arc<SampleResult> {
        &self.raw
    }

    pub fn num_chains(&self) -> usize {
        self.raw.samples.len()
    }

    pub fn num_draws(&self) -> usize {
        self.raw.samples.first().map_or(0, Vec::len)
    }

    /// The model structure hydrated with its training data.
    pub fn graph(&self) -> Graph {
        self.model.structure.with_binding(&self.binding)
    }

    /// The unconstrained sampler position of one posterior draw.
    ///
    /// The stored exact position is preferred: a constrained value may have
    /// rounded onto a transform boundary, where its inverse is infinite or
    /// otherwise lossy.
    pub fn position(&self, chain: usize, draw: usize) -> Cow<'_, [f64]> {
        match &self.raw.unconstrained_samples {
            Some(positions) => Cow::Borrowed(&positions[chain][draw]),
            None => Cow::Owned(constrained_draw_to_raw(
                &self.raw.samples[chain][draw],
                &self.model.structure.param_transforms,
            )),
        }
    }

    /// The graph predictions are evaluated on: the training data when neither
    /// inputs nor sizes are given, otherwise [`bind_prediction`] over them.
    pub fn prediction_graph(
        &self,
        inputs: Option<DataInputs>,
        sizes: Option<HashMap<String, usize>>,
    ) -> ModelResult<Graph> {
        let graph = self.graph();
        if inputs.is_none() && sizes.is_none() {
            graph
                .validate_shapes()
                .map_err(|e| ModelError::invalid(e.to_string()))?;
            return Ok(graph);
        }
        bind_prediction(
            &graph,
            inputs.unwrap_or_default(),
            sizes.unwrap_or_default(),
        )
    }

    /// Predict at new inputs, keeping the posterior's chain and draw axes.
    /// Response placeholders are constructed internally.
    pub fn predict(
        &self,
        inputs: DataInputs,
        sizes: HashMap<String, usize>,
        seed: u64,
        expected: bool,
    ) -> ModelResult<Prediction> {
        let graph = self.prediction_graph(Some(inputs), Some(sizes))?;
        let draws = self.posterior_predictive(&graph, None, seed, expected)?;
        let (chains, n_draws) = (self.num_chains(), self.num_draws());
        Ok(self
            .model
            .likelihood_names
            .iter()
            .zip(draws.values)
            .zip(draws.n_obs)
            .map(|((name, values), n_obs)| {
                let mut rows = values.chunks(n_obs.max(1)).map(<[f64]>::to_vec);
                let nested = (0..chains)
                    .map(|_| rows.by_ref().take(n_draws).collect())
                    .collect();
                (name.clone(), nested)
            })
            .collect())
    }

    /// Posterior-predictive simulation on a flat draw axis.
    ///
    /// Draws are taken chain-major; when `n_samples` asks for fewer than were
    /// sampled, that many are chosen without replacement and kept in
    /// chain-major order. `expected` returns each observation's mean instead
    /// of a simulated value. The stream is keyed by `seed` alone, so a seeded
    /// call reproduces exactly.
    pub fn posterior_predictive(
        &self,
        graph: &Graph,
        n_samples: Option<usize>,
        seed: u64,
        expected: bool,
    ) -> ModelResult<PredictiveDraws> {
        let mut rng = predictive_rng(seed);
        let all_draws: Vec<(usize, usize)> = self
            .raw
            .samples
            .iter()
            .enumerate()
            .flat_map(|(chain, draws)| (0..draws.len()).map(move |draw| (chain, draw)))
            .collect();
        let coordinates = select_posterior_draw_indices(all_draws.len(), n_samples, &mut rng)
            .into_iter()
            .map(|index| all_draws[index])
            .collect();
        self.simulate(graph, coordinates, expected, &mut rng)
    }

    /// Posterior-predictive simulation on the posterior's own
    /// `(chain, draw, obs)` grid, at the training data.
    ///
    /// When `n_samples` asks for fewer draws than were sampled, the thinning
    /// is chain-stratified: one shared set of per-chain draw indices is kept
    /// in every chain, so the block stays rectangular and every chain is
    /// represented equally. A request smaller than the chain count still
    /// keeps one draw per chain. The second value holds the retained draw
    /// indices in increasing order.
    pub fn posterior_predictive_grid(
        &self,
        n_samples: Option<usize>,
        seed: u64,
    ) -> ModelResult<(PredictiveDraws, Vec<usize>)> {
        let graph = self.prediction_graph(None, None)?;
        let mut rng = predictive_rng(seed);
        let (n_chains, n_draws) = (self.num_chains(), self.num_draws());
        let per_chain = match n_samples {
            Some(requested) if n_chains > 0 => (requested / n_chains).max(1).min(n_draws),
            _ => n_draws,
        };
        let retained = select_posterior_draw_indices(n_draws, Some(per_chain), &mut rng);
        let coordinates = (0..n_chains)
            .flat_map(|chain| retained.iter().map(move |&draw| (chain, draw)))
            .collect();
        let draws = self.simulate(&graph, coordinates, false, &mut rng)?;
        Ok((draws, retained))
    }

    /// Forward-simulate every likelihood at the given posterior coordinates.
    fn simulate<R: Rng + ?Sized>(
        &self,
        graph: &Graph,
        coordinates: Vec<(usize, usize)>,
        expected: bool,
        rng: &mut R,
    ) -> ModelResult<PredictiveDraws> {
        let heads = graph.observation_heads();
        let mut evaluator =
            Evaluator::try_new(graph).map_err(|error| ModelError::invalid(error.to_string()))?;
        let mut values: Vec<Vec<f64>> = heads
            .iter()
            .map(|head| Vec::with_capacity(coordinates.len() * head.n_obs))
            .collect();
        for &(chain, draw) in &coordinates {
            evaluator.forward(graph, &self.position(chain, draw));
            for (block, head) in values.iter_mut().zip(&heads) {
                let aux = head.aux.map(|node| evaluator.scalar_at(node));
                for i in 0..head.n_obs {
                    let eta = evaluator.vec_elem(head.linpred, i, graph);
                    block.push(
                        if expected {
                            crate::observation::mean(head.family, eta, aux)
                        } else {
                            crate::observation::sample(head.family, eta, aux, rng)
                        }
                        .map_err(ModelError::invalid)?,
                    );
                }
            }
        }
        Ok(PredictiveDraws {
            coordinates,
            n_obs: heads.iter().map(|head| head.n_obs).collect(),
            values,
        })
    }

    /// Pointwise log-likelihood of every training observation at every
    /// posterior draw: one row-major `(chain, draw, obs)` block per
    /// likelihood, in `likelihood_names` order. This is the group ArviZ uses
    /// for LOO and WAIC.
    pub fn log_likelihood(&self) -> ModelResult<Vec<Vec<f64>>> {
        let graph = self.prediction_graph(None, None)?;
        let heads = graph.observation_heads();
        self.evaluate_chunks(&graph, heads.len(), |evaluator, blocks| {
            for (block, head) in blocks.iter_mut().zip(&heads) {
                pointwise_log_likelihood(evaluator, &graph, head, block)?;
            }
            Ok(())
        })
    }

    /// Every deterministic evaluated at every posterior draw on `graph`
    /// (the training data, or a [`Self::prediction_graph`]).
    ///
    /// A nonfinite value is a failed computation, not a result -- the same
    /// standard the sampler holds the parameters to.
    pub fn deterministics(&self, graph: &Graph) -> ModelResult<Vec<DeterministicDraws>> {
        let evaluator =
            Evaluator::try_new(graph).map_err(|error| ModelError::invalid(error.to_string()))?;
        let lens: Vec<usize> = graph
            .deterministics
            .iter()
            .map(|(_, node)| evaluator.node_len(*node))
            .collect();
        let values = self.evaluate_chunks(graph, lens.len(), |evaluator, blocks| {
            for ((block, (_, node)), len) in blocks.iter_mut().zip(&graph.deterministics).zip(&lens)
            {
                block.extend((0..(*len).max(1)).map(|i| evaluator.vec_elem(*node, i, graph)));
            }
            Ok(())
        })?;
        let n_draws = self.num_draws();
        graph
            .deterministics
            .iter()
            .zip(lens)
            .zip(values)
            .map(|(((name, _), len), values)| {
                if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                    let flat_draw = index / len.max(1);
                    return Err(ModelError::invalid(format!(
                        "deterministic '{name}' is nonfinite at chain {}, draw {}",
                        flat_draw / n_draws.max(1),
                        flat_draw % n_draws.max(1)
                    )));
                }
                Ok(DeterministicDraws {
                    name: name.clone(),
                    len,
                    values,
                })
            })
            .collect()
    }

    /// Run a forward pass at every posterior draw, chain-major, and let
    /// `record` append that draw's outputs to each of `n_outputs` blocks.
    ///
    /// Draws are split into chunks evaluated in parallel, each with one
    /// evaluator, and the chunks' blocks are concatenated in draw order, so
    /// the result does not depend on the pool.
    fn evaluate_chunks<F>(
        &self,
        graph: &Graph,
        n_outputs: usize,
        record: F,
    ) -> ModelResult<Vec<Vec<f64>>>
    where
        F: Fn(&mut Evaluator, &mut Vec<Vec<f64>>) -> ModelResult<()> + Sync,
    {
        let n_draws = self.num_draws();
        let total = self.num_chains() * n_draws;
        let starts: Vec<usize> = (0..total).step_by(EVALUATION_CHUNK).collect();
        let chunks = starts
            .into_par_iter()
            .map(|start| {
                let mut evaluator = Evaluator::try_new(graph)
                    .map_err(|error| ModelError::invalid(error.to_string()))?;
                let mut out = vec![Vec::new(); n_outputs];
                for flat in start..(start + EVALUATION_CHUNK).min(total) {
                    let (chain, draw) = (flat / n_draws, flat % n_draws);
                    evaluator.forward(graph, &self.position(chain, draw));
                    record(&mut evaluator, &mut out)?;
                }
                Ok(out)
            })
            .collect::<ModelResult<Vec<_>>>()?;
        let mut blocks = vec![Vec::new(); n_outputs];
        for chunk in chunks {
            for (block, part) in blocks.iter_mut().zip(chunk) {
                block.extend(part);
            }
        }
        Ok(blocks)
    }
}

fn pointwise_log_likelihood(
    evaluator: &Evaluator,
    graph: &Graph,
    head: &ObservationHead,
    out: &mut Vec<f64>,
) -> ModelResult<()> {
    let aux = head.aux.map(|node| evaluator.scalar_at(node));
    for (i, &observed) in graph.obs_vectors[head.obs_data_idx].iter().enumerate() {
        out.push(
            crate::observation::log_density(
                head.family,
                observed,
                evaluator.vec_elem(head.linpred, i, graph),
                aux,
            )
            .map_err(ModelError::invalid)?,
        );
    }
    Ok(())
}

fn predictive_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(crate::seeding::stream_seed(
        seed,
        crate::seeding::POSTERIOR_PREDICT_SEED_DOMAIN,
    ))
}

/// `n_samples` of `total_draws` indices without replacement, sorted; all of
/// them, consuming no randomness, when `n_samples` is absent or not smaller.
fn select_posterior_draw_indices<R: Rng + ?Sized>(
    total_draws: usize,
    n_samples: Option<usize>,
    rng: &mut R,
) -> Vec<usize> {
    let n = n_samples.unwrap_or(total_draws).min(total_draws);
    if n >= total_draws {
        return (0..total_draws).collect();
    }
    let mut indices: Vec<usize> = (0..total_draws).collect();
    indices.shuffle(rng);
    indices.truncate(n);
    indices.sort_unstable();
    indices
}

/// The unconstrained value a constrained draw came from.
fn unconstrain(transform: &ParamTransform, value: f64) -> f64 {
    use crate::prior_sampling::logit_stable;
    match transform {
        ParamTransform::Identity => value,
        ParamTransform::Exp => value.ln(),
        ParamTransform::Sigmoid => logit_stable(value),
        ParamTransform::BoundedSigmoid { lower, upper } => {
            logit_stable((value - lower) / (upper - lower))
        }
    }
}

pub(crate) fn constrained_draw_to_raw(draw: &[f64], transforms: &[ParamTransform]) -> Vec<f64> {
    draw.iter()
        .zip(transforms)
        .map(|(&value, transform)| unconstrain(transform, value))
        .collect()
}
