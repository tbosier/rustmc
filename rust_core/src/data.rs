//! Dataset schemas and validated, re-bindable payloads.

use std::collections::{BTreeSet, HashMap};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotKind {
    Vector,
    Observation { likelihood: String },
    Matrix { n_cols: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataSlot {
    pub key: String,
    pub kind: SlotKind,
    pub dim: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataSchema {
    pub vectors: Vec<DataSlot>,
    pub observations: Vec<DataSlot>,
    pub matrices: Vec<DataSlot>,
}

impl DataSchema {
    pub fn required_keys(&self) -> Vec<&str> {
        self.observations
            .iter()
            .chain(&self.vectors)
            .chain(&self.matrices)
            .map(|slot| slot.key.as_str())
            .collect()
    }

    pub fn describe(&self) -> Vec<(String, SlotKind)> {
        self.observations
            .iter()
            .chain(&self.vectors)
            .chain(&self.matrices)
            .map(|slot| (slot.key.clone(), slot.kind.clone()))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct MatrixBinding {
    pub data: Arc<[f64]>,
    pub n_rows: usize,
    pub n_cols: usize,
}

#[derive(Debug, Clone, Default)]
pub struct DataInputs {
    pub vectors: HashMap<String, Arc<[f64]>>,
    pub matrices: HashMap<String, MatrixBinding>,
}

#[derive(Debug, Clone)]
pub struct DataBinding {
    pub(crate) vectors: Vec<Arc<[f64]>>,
    pub(crate) observations: Vec<Arc<[f64]>>,
    pub(crate) matrices: Vec<MatrixBinding>,
    schema: DataSchema,
    n_obs: usize,
    id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BindError {
    MissingKey {
        key: String,
        kind: SlotKind,
    },
    UnexpectedKey {
        key: String,
        did_you_mean: Option<String>,
    },
    WrongKind {
        key: String,
        expected: SlotKind,
        found: SlotKind,
    },
    Empty {
        key: String,
    },
    LengthMismatch {
        key: String,
        len: usize,
        expected: usize,
        expected_from: String,
    },
    MatrixColsMismatch {
        key: String,
        n_cols: usize,
        expected: usize,
    },
    RaggedMatrix {
        key: String,
        n_rows: usize,
        n_cols: usize,
        values: usize,
    },
    NonFinite {
        key: String,
        index: usize,
        value: f64,
    },
    InvalidObservation {
        likelihood: String,
        family: String,
        index: usize,
        value: f64,
        requirement: &'static str,
    },
    SchemaMismatch,
}

impl Display for BindError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingKey { key, kind } => {
                write!(f, "missing required {:?} data key '{}'", kind, key)
            }
            Self::UnexpectedKey {
                key,
                did_you_mean: Some(s),
            } => write!(f, "unexpected data key '{}'; did you mean '{}'?", key, s),
            Self::UnexpectedKey {
                key,
                did_you_mean: None,
            } => write!(f, "unexpected data key '{}'", key),
            Self::WrongKind {
                key,
                expected,
                found,
            } => write!(
                f,
                "data key '{}' has kind {:?}, expected {:?}",
                key, found, expected
            ),
            Self::Empty { key } => write!(f, "data key '{}' must not be empty", key),
            Self::LengthMismatch {
                key,
                len,
                expected,
                expected_from,
            } => write!(
                f,
                "'{}' has length {}, expected {} (from '{}')",
                key, len, expected, expected_from
            ),
            Self::MatrixColsMismatch {
                key,
                n_cols,
                expected,
            } => write!(
                f,
                "matrix '{}' has {} columns, expected {}",
                key, n_cols, expected
            ),
            Self::RaggedMatrix {
                key,
                n_rows,
                n_cols,
                values,
            } => write!(
                f,
                "matrix '{}' declares shape {}x{} but contains {} values",
                key, n_rows, n_cols, values
            ),
            Self::NonFinite { key, index, value } => write!(
                f,
                "data key '{}' contains non-finite value {} at flat index {}",
                key, value, index
            ),
            Self::InvalidObservation {
                likelihood,
                family,
                index,
                value,
                requirement,
            } => write!(
                f,
                "{} likelihood '{}' requires {} observed values; found {} at index {}",
                family, likelihood, requirement, value, index
            ),
            Self::SchemaMismatch => write!(f, "binding was created for a different data schema"),
        }
    }
}

impl std::error::Error for BindError {}

impl DataBinding {
    /// Compatibility bridge for legacy graphs that still carry one dataset.
    pub fn from_graph(graph: &crate::graph::Graph) -> Result<Self, BindError> {
        let vectors = graph
            .data_vectors
            .iter()
            .cloned()
            .map(Arc::<[f64]>::from)
            .collect::<Vec<_>>();
        let observations = graph
            .obs_vectors
            .iter()
            .cloned()
            .map(Arc::<[f64]>::from)
            .collect::<Vec<_>>();
        let matrices = graph
            .data_matrices
            .iter()
            .map(|m| MatrixBinding {
                data: Arc::from(m.data.clone()),
                n_rows: m.n_rows,
                n_cols: m.n_cols,
            })
            .collect::<Vec<_>>();
        let mut lengths = observations
            .iter()
            .map(|v| v.len())
            .chain(vectors.iter().map(|v| v.len()))
            .chain(matrices.iter().map(|m| m.n_rows));
        let n_obs = lengths.next().unwrap_or(1);
        if n_obs == 0 {
            return Err(BindError::Empty {
                key: "dataset".to_string(),
            });
        }
        if lengths.any(|len| len != n_obs) {
            return Err(BindError::LengthMismatch {
                key: "dataset".to_string(),
                len: 0,
                expected: n_obs,
                expected_from: "first data slot".to_string(),
            });
        }
        Ok(Self {
            vectors,
            observations,
            matrices,
            schema: graph.schema.clone(),
            n_obs,
            id: "0".to_string(),
        })
    }

    pub fn bind(
        schema: &DataSchema,
        inputs: DataInputs,
        id: impl Into<String>,
        strict: bool,
        check_finite: bool,
    ) -> Result<Self, BindError> {
        for slot in schema.observations.iter().chain(&schema.vectors) {
            if !inputs.vectors.contains_key(&slot.key) {
                if inputs.matrices.contains_key(&slot.key) {
                    return Err(BindError::WrongKind {
                        key: slot.key.clone(),
                        expected: slot.kind.clone(),
                        found: SlotKind::Matrix {
                            n_cols: inputs.matrices[&slot.key].n_cols,
                        },
                    });
                }
                return Err(BindError::MissingKey {
                    key: slot.key.clone(),
                    kind: slot.kind.clone(),
                });
            }
        }
        for slot in &schema.matrices {
            if !inputs.matrices.contains_key(&slot.key) {
                if inputs.vectors.contains_key(&slot.key) {
                    return Err(BindError::WrongKind {
                        key: slot.key.clone(),
                        expected: slot.kind.clone(),
                        found: SlotKind::Vector,
                    });
                }
                return Err(BindError::MissingKey {
                    key: slot.key.clone(),
                    kind: slot.kind.clone(),
                });
            }
        }
        if strict {
            let required: BTreeSet<&str> = schema.required_keys().into_iter().collect();
            let mut extras: Vec<&str> = inputs
                .vectors
                .keys()
                .chain(inputs.matrices.keys())
                .map(String::as_str)
                .filter(|k| !required.contains(k))
                .collect();
            extras.sort_unstable();
            if let Some(key) = extras.first() {
                let suggestion = required
                    .iter()
                    .copied()
                    .find(|candidate| edit_distance(key, candidate) <= 2)
                    .map(str::to_string);
                return Err(BindError::UnexpectedKey {
                    key: (*key).to_string(),
                    did_you_mean: suggestion,
                });
            }
        }

        let mut n_obs = None;
        let mut source = String::new();
        let mut validate_len = |key: &str, len: usize| -> Result<(), BindError> {
            if len == 0 {
                return Err(BindError::Empty {
                    key: key.to_string(),
                });
            }
            if let Some(expected) = n_obs {
                if len != expected {
                    return Err(BindError::LengthMismatch {
                        key: key.to_string(),
                        len,
                        expected,
                        expected_from: source.clone(),
                    });
                }
            } else {
                n_obs = Some(len);
                source = key.to_string();
            }
            Ok(())
        };
        let mut observations = Vec::with_capacity(schema.observations.len());
        for slot in &schema.observations {
            let v = Arc::clone(&inputs.vectors[&slot.key]);
            validate_len(&slot.key, v.len())?;
            observations.push(v);
        }
        let mut vectors = Vec::with_capacity(schema.vectors.len());
        for slot in &schema.vectors {
            let v = Arc::clone(&inputs.vectors[&slot.key]);
            validate_len(&slot.key, v.len())?;
            vectors.push(v);
        }
        let mut matrices = Vec::with_capacity(schema.matrices.len());
        for slot in &schema.matrices {
            let m = inputs.matrices[&slot.key].clone();
            if m.n_rows.checked_mul(m.n_cols) != Some(m.data.len()) {
                return Err(BindError::RaggedMatrix {
                    key: slot.key.clone(),
                    n_rows: m.n_rows,
                    n_cols: m.n_cols,
                    values: m.data.len(),
                });
            }
            if let SlotKind::Matrix { n_cols } = slot.kind {
                if m.n_cols != n_cols {
                    return Err(BindError::MatrixColsMismatch {
                        key: slot.key.clone(),
                        n_cols: m.n_cols,
                        expected: n_cols,
                    });
                }
            }
            validate_len(&slot.key, m.n_rows)?;
            matrices.push(m);
        }
        if check_finite {
            for (slot, values) in schema
                .observations
                .iter()
                .zip(&observations)
                .chain(schema.vectors.iter().zip(&vectors))
            {
                if let Some((index, value)) = values
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, v)| !v.is_finite())
                {
                    return Err(BindError::NonFinite {
                        key: slot.key.clone(),
                        index,
                        value,
                    });
                }
            }
            for (slot, matrix) in schema.matrices.iter().zip(&matrices) {
                if let Some((index, value)) = matrix
                    .data
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, v)| !v.is_finite())
                {
                    return Err(BindError::NonFinite {
                        key: slot.key.clone(),
                        index,
                        value,
                    });
                }
            }
        }
        Ok(Self {
            vectors,
            observations,
            matrices,
            schema: schema.clone(),
            n_obs: n_obs.unwrap_or(1),
            id: id.into(),
        })
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn set_id(&mut self, id: impl Into<String>) {
        self.id = id.into();
    }

    /// Recheck the binding's structural identity and internal payload shapes
    /// before it enters evaluator code. `DataBinding` fields are opaque so all
    /// externally constructible values have already passed `bind()`.
    pub fn validate_for(&self, graph: &crate::graph::Graph) -> Result<(), BindError> {
        if self.schema != graph.schema {
            return Err(BindError::SchemaMismatch);
        }
        for (index, values) in self.observations.iter().chain(&self.vectors).enumerate() {
            if values.len() != self.n_obs {
                return Err(BindError::LengthMismatch {
                    key: format!("binding vector slot {index}"),
                    len: values.len(),
                    expected: self.n_obs,
                    expected_from: "binding row count".into(),
                });
            }
        }
        for (index, matrix) in self.matrices.iter().enumerate() {
            if matrix.n_rows != self.n_obs {
                return Err(BindError::LengthMismatch {
                    key: format!("binding matrix slot {index}"),
                    len: matrix.n_rows,
                    expected: self.n_obs,
                    expected_from: "binding row count".into(),
                });
            }
            if matrix.n_rows.checked_mul(matrix.n_cols) != Some(matrix.data.len()) {
                return Err(BindError::RaggedMatrix {
                    key: format!("binding matrix slot {index}"),
                    n_rows: matrix.n_rows,
                    n_cols: matrix.n_cols,
                    values: matrix.data.len(),
                });
            }
        }
        for node in &graph.nodes {
            let crate::graph::Op::ObsLogP {
                family,
                obs_data_idx,
                ..
            } = &node.op
            else {
                continue;
            };
            let values = self
                .observations
                .get(*obs_data_idx)
                .ok_or(BindError::SchemaMismatch)?;
            let (family_name, requirement, invalid): (&str, &'static str, fn(f64) -> bool) =
                match family {
                    crate::graph::ObsFamily::Normal => {
                        ("Normal", "finite", |value| !value.is_finite())
                    }
                    crate::graph::ObsFamily::BernoulliLogit => {
                        ("Bernoulli-logit", "binary (0 or 1)", |value| {
                            !value.is_finite() || (value != 0.0 && value != 1.0)
                        })
                    }
                    crate::graph::ObsFamily::PoissonLog => {
                        ("Poisson-log", "non-negative integers", |value| {
                            !value.is_finite() || value < 0.0 || value.fract() != 0.0
                        })
                    }
                    crate::graph::ObsFamily::ExponentialLog => {
                        ("Exponential", "non-negative", |value| {
                            !value.is_finite() || value < 0.0
                        })
                    }
                    crate::graph::ObsFamily::LogNormal => {
                        ("LogNormal", "strictly positive", |value| {
                            !value.is_finite() || value <= 0.0
                        })
                    }
                    crate::graph::ObsFamily::NegativeBinomialLog => {
                        ("NegativeBinomial", "non-negative integers", |value| {
                            !value.is_finite() || value < 0.0 || value.fract() != 0.0
                        })
                    }
                };
            if let Some((index, value)) = values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| invalid(*value))
            {
                let likelihood = graph
                    .schema
                    .observations
                    .get(*obs_data_idx)
                    .and_then(|slot| match &slot.kind {
                        SlotKind::Observation { likelihood } => Some(likelihood.clone()),
                        SlotKind::Vector | SlotKind::Matrix { .. } => None,
                    })
                    .unwrap_or_else(|| family_name.to_string());
                return Err(BindError::InvalidObservation {
                    likelihood,
                    family: family_name.to_string(),
                    index,
                    value,
                    requirement,
                });
            }
        }
        Ok(())
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.bytes().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.bytes().enumerate() {
            let old = row[j + 1];
            row[j + 1] = (row[j + 1] + 1)
                .min(row[j] + 1)
                .min(prev + usize::from(ca != cb));
            prev = old;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::Evaluator;
    use crate::graph::Graph;
    use std::sync::Arc;

    fn schema() -> DataSchema {
        DataSchema {
            observations: vec![DataSlot {
                key: "y".into(),
                kind: SlotKind::Observation {
                    likelihood: "y_obs".into(),
                },
                dim: "obs".into(),
            }],
            vectors: vec![DataSlot {
                key: "x".into(),
                kind: SlotKind::Vector,
                dim: "obs".into(),
            }],
            matrices: vec![DataSlot {
                key: "X".into(),
                kind: SlotKind::Matrix { n_cols: 2 },
                dim: "obs".into(),
            }],
        }
    }

    fn inputs(n: usize) -> DataInputs {
        DataInputs {
            vectors: HashMap::from([
                ("x".into(), Arc::from(vec![1.0; n])),
                ("y".into(), Arc::from(vec![2.0; n])),
            ]),
            matrices: HashMap::from([(
                "X".into(),
                MatrixBinding {
                    data: Arc::from(vec![1.0; n * 2]),
                    n_rows: n,
                    n_cols: 2,
                },
            )]),
        }
    }

    #[test]
    fn row_count_is_binding_not_structure() {
        let a = DataBinding::bind(&schema(), inputs(3), "a", true, true).unwrap();
        let b = DataBinding::bind(&schema(), inputs(7), "b", true, true).unwrap();
        assert_eq!((a.n_obs, b.n_obs), (3, 7));
    }

    #[test]
    fn reports_schema_errors_actionably() {
        let mut missing = inputs(3);
        missing.vectors.remove("y");
        assert!(DataBinding::bind(&schema(), missing, "x", true, true)
            .unwrap_err()
            .to_string()
            .contains("missing required"));
        let mut wrong_cols = inputs(3);
        wrong_cols.matrices.insert(
            "X".into(),
            MatrixBinding {
                data: Arc::from(vec![1.0; 9]),
                n_rows: 3,
                n_cols: 3,
            },
        );
        assert!(DataBinding::bind(&schema(), wrong_cols, "x", true, true)
            .unwrap_err()
            .to_string()
            .contains("expected 2"));
        let mut nan = inputs(3);
        nan.vectors
            .insert("x".into(), Arc::from(vec![1.0, f64::NAN, 2.0]));
        assert!(DataBinding::bind(&schema(), nan, "x", true, true)
            .unwrap_err()
            .to_string()
            .contains("non-finite"));
    }

    #[test]
    fn evaluator_rebinds_row_count_while_arc_shares_structure() {
        let mut graph = Graph::new();
        let beta = graph.add_param("beta");
        let zero = graph.add_constant(0.0);
        let one = graph.add_constant(1.0);
        graph.normal_logp(beta, zero, one);
        let x = graph.add_data("x", vec![1.0]);
        let mu = graph.scalar_mul_data(beta, x);
        let obs = graph.add_named_obs_data("y", "obs", vec![1.0]);
        graph.normal_obs_logp(mu, one, obs);
        let structure = Arc::new(graph.structure_only());
        let shared = Arc::clone(&structure);

        let small = DataBinding::bind(
            &structure.schema,
            DataInputs {
                vectors: HashMap::from([
                    ("x".into(), Arc::from(vec![1.0; 2])),
                    ("y".into(), Arc::from(vec![1.0; 2])),
                ]),
                matrices: HashMap::new(),
            },
            "small",
            true,
            true,
        )
        .unwrap();
        let large = DataBinding::bind(
            &structure.schema,
            DataInputs {
                vectors: HashMap::from([
                    ("x".into(), Arc::from(vec![1.0; 6])),
                    ("y".into(), Arc::from(vec![1.0; 6])),
                ]),
                matrices: HashMap::new(),
            },
            "large",
            true,
            true,
        )
        .unwrap();

        let mut evaluator = Evaluator::with_binding(&structure, small);
        evaluator.compute(&structure, &[0.5]);
        let small_logp = evaluator.total_logp;
        evaluator.rebind(&structure, large).unwrap();
        evaluator.compute(&structure, &[0.5]);
        assert!(evaluator.total_logp < small_logp);
        assert!(Arc::ptr_eq(&structure, &shared));
    }

    #[test]
    fn evaluator_rejects_a_binding_from_a_different_schema() {
        let mut graph = Graph::new();
        let beta = graph.add_param("beta");
        let one = graph.add_constant(1.0);
        let x = graph.add_data("x", vec![1.0]);
        let mu = graph.scalar_mul_data(beta, x);
        let obs = graph.add_named_obs_data("y", "obs", vec![1.0]);
        graph.normal_obs_logp(mu, one, obs);
        let structure = graph.structure_only();

        let foreign_schema = DataSchema {
            observations: structure.schema.observations.clone(),
            vectors: vec![DataSlot {
                key: "foreign_x".into(),
                kind: SlotKind::Vector,
                dim: "obs".into(),
            }],
            matrices: Vec::new(),
        };
        let wrong = DataBinding::bind(
            &foreign_schema,
            DataInputs {
                vectors: HashMap::from([
                    ("foreign_x".into(), Arc::from(vec![1.0])),
                    ("y".into(), Arc::from(vec![1.0])),
                ]),
                matrices: HashMap::new(),
            },
            "wrong",
            true,
            true,
        )
        .unwrap();
        let error = match Evaluator::try_with_binding(&structure, wrong) {
            Ok(_) => panic!("binding from another schema was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("different data schema"));
    }

    #[test]
    fn evaluator_rejects_rebound_observations_outside_likelihood_domain() {
        let mut graph = Graph::new();
        let eta = graph.add_param("eta");
        let eta_vec = graph.scalar_broadcast(eta);
        let obs = graph.add_named_obs_data("y", "conversion", vec![1.0]);
        graph.obs_logp_bernoulli_logit(eta_vec, obs);
        let structure = graph.structure_only();
        let binding = DataBinding::bind(
            &structure.schema,
            DataInputs {
                vectors: HashMap::from([("y".into(), Arc::from(vec![0.5]))]),
                matrices: HashMap::new(),
            },
            "invalid",
            true,
            true,
        )
        .unwrap();

        let error = match Evaluator::try_with_binding(&structure, binding) {
            Ok(_) => panic!("invalid Bernoulli observations were accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("requires binary"));
        assert!(error.to_string().contains("conversion"));
    }

    #[test]
    fn rebound_observation_boundaries_use_exact_support() {
        let mut lognormal = Graph::new();
        let mu = lognormal.add_param("mu");
        let mu_vec = lognormal.scalar_broadcast(mu);
        let sigma = lognormal.add_constant(1.0);
        let obs = lognormal.add_named_obs_data("y", "duration", vec![1.0]);
        lognormal.obs_logp_lognormal(mu_vec, sigma, obs);
        let structure = lognormal.structure_only();
        let tiny_positive = DataBinding::bind(
            &structure.schema,
            DataInputs {
                vectors: HashMap::from([("y".into(), Arc::from(vec![1e-12]))]),
                matrices: HashMap::new(),
            },
            "tiny-positive",
            true,
            true,
        )
        .unwrap();
        assert!(Evaluator::try_with_binding(&structure, tiny_positive).is_ok());

        let mut exponential = Graph::new();
        let eta = exponential.add_param("eta");
        let eta_vec = exponential.scalar_broadcast(eta);
        let obs = exponential.add_named_obs_data("y", "waiting_time", vec![1.0]);
        exponential.obs_logp_exponential_log(eta_vec, obs);
        let structure = exponential.structure_only();
        let tiny_negative = DataBinding::bind(
            &structure.schema,
            DataInputs {
                vectors: HashMap::from([("y".into(), Arc::from(vec![-1e-12]))]),
                matrices: HashMap::new(),
            },
            "tiny-negative",
            true,
            true,
        )
        .unwrap();
        let error = match Evaluator::try_with_binding(&structure, tiny_negative) {
            Ok(_) => panic!("negative Exponential observation was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("requires non-negative"));
    }
}
