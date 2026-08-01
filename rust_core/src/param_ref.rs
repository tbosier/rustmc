//! Build-time validation of parameter references.
//!
//! A model is described by an ordered list of *parameter declarations* (priors,
//! in declaration order) and a set of *parameter references* (hyperparameters of
//! other priors, likelihood scale parameters, terms in a linear predictor).
//!
//! Every reference must resolve to a declaration. Some references additionally
//! carry an ordering constraint: a prior's hyperparameter must have been
//! declared *before* the prior that uses it, because the computation graph is
//! built in declaration order.
//!
//! Validating this once, up front, is strictly better than discovering it
//! half-way through building a graph — or worse, silently substituting a
//! default value and returning a plausible-but-wrong posterior.

use std::fmt;

/// A single use of a parameter name somewhere in a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamReference {
    /// The referenced parameter name.
    pub name: String,
    /// Human-readable description of *where* the reference occurs, e.g.
    /// `"prior 'theta' hyperparameter mu"`. Used verbatim in error messages.
    pub context: String,
    /// If `Some(i)`, the referenced parameter must be declared strictly before
    /// declaration index `i`. `None` means order does not matter.
    pub must_precede: Option<usize>,
}

impl ParamReference {
    /// A reference with no ordering constraint (e.g. a likelihood term).
    pub fn unordered(name: impl Into<String>, context: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            context: context.into(),
            must_precede: None,
        }
    }

    /// A reference that must resolve to a declaration made before index
    /// `declaration_index`.
    pub fn ordered(
        name: impl Into<String>,
        context: impl Into<String>,
        declaration_index: usize,
    ) -> Self {
        Self {
            name: name.into(),
            context: context.into(),
            must_precede: Some(declaration_index),
        }
    }
}

/// Why a parameter reference could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamRefError {
    /// The name does not appear anywhere in the model.
    Undeclared {
        name: String,
        context: String,
        available: Vec<String>,
        suggestion: Option<String>,
    },
    /// The name exists, but is declared after the point where it is used.
    UsedBeforeDeclared {
        name: String,
        context: String,
        /// Zero-based declaration index of the referenced parameter.
        declared_at: usize,
        /// Zero-based declaration index of the declaration doing the referencing.
        used_at: usize,
    },
    /// Two declarations share a name, so a reference to it is ambiguous.
    DuplicateDeclaration {
        name: String,
        first: usize,
        second: usize,
    },
}

impl ParamRefError {
    /// The offending parameter name. Always present, in every variant, so that
    /// callers (and tests) can rely on the message naming it.
    pub fn param_name(&self) -> &str {
        match self {
            ParamRefError::Undeclared { name, .. }
            | ParamRefError::UsedBeforeDeclared { name, .. }
            | ParamRefError::DuplicateDeclaration { name, .. } => name,
        }
    }
}

impl fmt::Display for ParamRefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamRefError::Undeclared {
                name,
                context,
                available,
                suggestion,
            } => {
                write!(f, "unknown parameter '{}' referenced by {}.", name, context)?;
                if let Some(s) = suggestion {
                    write!(f, " Did you mean '{}'?", s)?;
                }
                if available.is_empty() {
                    write!(f, " This model declares no parameters.")
                } else {
                    write!(f, " Declared parameters: [{}]", available.join(", "))
                }
            }
            ParamRefError::UsedBeforeDeclared {
                name,
                context,
                declared_at,
                used_at,
            } => write!(
                f,
                "parameter '{}' is referenced by {} but is declared later \
                 (declaration #{} references declaration #{}). Declare '{}' \
                 before the prior that uses it.",
                name, context, used_at, declared_at, name
            ),
            ParamRefError::DuplicateDeclaration {
                name,
                first,
                second,
            } => write!(
                f,
                "parameter '{}' is declared twice (declarations #{} and #{}); \
                 references to it would be ambiguous.",
                name, first, second
            ),
        }
    }
}

impl std::error::Error for ParamRefError {}

/// Check that every reference in `refs` resolves to a declaration in
/// `declared`, respecting ordering constraints.
///
/// `declared` is the ordered list of parameter names the model declares.
pub fn validate_param_references(
    declared: &[String],
    refs: &[ParamReference],
) -> Result<(), ParamRefError> {
    // First declaration index for each name (and a duplicate check).
    let mut index_of: Vec<(&str, usize)> = Vec::with_capacity(declared.len());
    for (i, name) in declared.iter().enumerate() {
        if let Some(&(_, first)) = index_of.iter().find(|(n, _)| *n == name.as_str()) {
            return Err(ParamRefError::DuplicateDeclaration {
                name: name.clone(),
                first,
                second: i,
            });
        }
        index_of.push((name.as_str(), i));
    }

    for reference in refs {
        let found = index_of
            .iter()
            .find(|(n, _)| *n == reference.name.as_str())
            .map(|&(_, i)| i);

        match found {
            None => {
                return Err(ParamRefError::Undeclared {
                    name: reference.name.clone(),
                    context: reference.context.clone(),
                    available: declared.to_vec(),
                    suggestion: suggest(&reference.name, declared),
                })
            }
            Some(declared_at) => {
                if let Some(used_at) = reference.must_precede {
                    if declared_at >= used_at {
                        return Err(ParamRefError::UsedBeforeDeclared {
                            name: reference.name.clone(),
                            context: reference.context.clone(),
                            declared_at,
                            used_at,
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

/// Closest declared name within a small edit distance, for "did you mean".
fn suggest(name: &str, candidates: &[String]) -> Option<String> {
    let budget = (name.chars().count() / 3).clamp(1, 3);
    candidates
        .iter()
        .map(|c| (levenshtein(name, c), c))
        .filter(|(d, _)| *d <= budget)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c.clone())
}

/// Plain Levenshtein edit distance (two-row dynamic program).
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn resolves_valid_references() {
        let declared = names(&["mu_pop", "sigma_pop", "theta", "sigma"]);
        let refs = vec![
            ParamReference::ordered("mu_pop", "prior 'theta' hyperparameter mu", 2),
            ParamReference::ordered("sigma_pop", "prior 'theta' hyperparameter sigma", 2),
            ParamReference::unordered("sigma", "likelihood 'y' sigma"),
        ];
        assert!(validate_param_references(&declared, &refs).is_ok());
    }

    #[test]
    fn missing_hyperparameter_is_named() {
        let declared = names(&["theta"]);
        let refs = vec![ParamReference::ordered(
            "sigma_pop",
            "prior 'theta' hyperparameter sigma",
            0,
        )];
        let err = validate_param_references(&declared, &refs).unwrap_err();
        assert_eq!(err.param_name(), "sigma_pop");
        let msg = err.to_string();
        assert!(msg.contains("sigma_pop"), "{}", msg);
        assert!(
            msg.contains("prior 'theta' hyperparameter sigma"),
            "{}",
            msg
        );
    }

    #[test]
    fn misspelled_name_gets_a_suggestion() {
        let declared = names(&["sigma_pop", "theta"]);
        let refs = vec![ParamReference::unordered(
            "sigma_pip",
            "likelihood 'y' sigma",
        )];
        let err = validate_param_references(&declared, &refs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("sigma_pip"), "{}", msg);
        assert!(msg.contains("Did you mean 'sigma_pop'?"), "{}", msg);
    }

    #[test]
    fn wildly_wrong_name_gets_no_suggestion_but_lists_available() {
        let declared = names(&["alpha", "beta"]);
        let refs = vec![ParamReference::unordered(
            "qqqqqqqq",
            "likelihood 'y' sigma",
        )];
        let err = validate_param_references(&declared, &refs).unwrap_err();
        let msg = err.to_string();
        assert!(!msg.contains("Did you mean"), "{}", msg);
        assert!(msg.contains("alpha, beta"), "{}", msg);
    }

    #[test]
    fn declaration_order_is_enforced_for_hyperparameters() {
        // theta declared at 0, references sigma_pop declared at 1.
        let declared = names(&["theta", "sigma_pop"]);
        let refs = vec![ParamReference::ordered(
            "sigma_pop",
            "prior 'theta' hyperparameter sigma",
            0,
        )];
        let err = validate_param_references(&declared, &refs).unwrap_err();
        assert!(matches!(err, ParamRefError::UsedBeforeDeclared { .. }));
        let msg = err.to_string();
        assert!(msg.contains("sigma_pop"), "{}", msg);
        assert!(msg.contains("declared later"), "{}", msg);
    }

    #[test]
    fn self_reference_is_rejected() {
        let declared = names(&["theta"]);
        let refs = vec![ParamReference::ordered(
            "theta",
            "prior 'theta' hyperparameter mu",
            0,
        )];
        assert!(matches!(
            validate_param_references(&declared, &refs).unwrap_err(),
            ParamRefError::UsedBeforeDeclared { .. }
        ));
    }

    #[test]
    fn unordered_references_ignore_position() {
        let declared = names(&["theta", "sigma"]);
        let refs = vec![ParamReference::unordered("sigma", "likelihood 'y' sigma")];
        assert!(validate_param_references(&declared, &refs).is_ok());
    }

    #[test]
    fn duplicate_declarations_are_rejected() {
        let declared = names(&["theta", "theta"]);
        let err = validate_param_references(&declared, &[]).unwrap_err();
        assert!(matches!(err, ParamRefError::DuplicateDeclaration { .. }));
        assert!(err.to_string().contains("theta"));
    }

    #[test]
    fn empty_model_reports_no_declared_parameters() {
        let refs = vec![ParamReference::unordered("theta", "likelihood 'y' sigma")];
        let err = validate_param_references(&[], &refs).unwrap_err();
        assert!(err.to_string().contains("declares no parameters"));
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("sigma", "sigmma"), 1);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }
}
