use crate::graph::Graph;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

const DENSE_BLOCK_MAX_DIM: usize = 512;
const REGULARIZATION_WEIGHT: f64 = 5.0;
const BASE_JITTER: f64 = 1e-3;

#[derive(Debug, Clone)]
pub struct MassMatrix {
    dim: usize,
    blocks: Vec<MassBlock>,
}

#[derive(Debug, Clone)]
struct MassBlock {
    start: usize,
    len: usize,
    kind: BlockKind,
}

#[derive(Debug, Clone)]
enum BlockKind {
    Scalar {
        variance: f64,
        inv_variance: f64,
    },
    Diagonal {
        variances: Vec<f64>,
        inv_variances: Vec<f64>,
    },
    Dense {
        chol: Vec<f64>,
    },
}

#[derive(Debug, Clone)]
pub struct MassMatrixAccumulator {
    blocks: Vec<AccumulatorBlock>,
}

#[derive(Debug, Clone)]
enum AccumulatorBlock {
    Scalar {
        start: usize,
        count: usize,
        mean: f64,
        m2: f64,
    },
    Diagonal {
        start: usize,
        count: usize,
        mean: Vec<f64>,
        m2: Vec<f64>,
    },
    Dense {
        start: usize,
        dim: usize,
        count: usize,
        mean: Vec<f64>,
        m2: Vec<f64>,
    },
}

impl MassMatrix {
    pub fn identity(graph: &Graph) -> Self {
        let mut blocks = Vec::with_capacity(graph.param_spans.len());
        for span in &graph.param_spans {
            blocks.push(MassBlock::identity(span.start, span.len));
        }
        Self {
            dim: graph.param_count,
            blocks,
        }
    }

    pub fn from_graph(graph: &Graph) -> Self {
        Self::identity(graph)
    }

    pub fn accumulator(graph: &Graph) -> MassMatrixAccumulator {
        MassMatrixAccumulator::from_graph(graph)
    }

    pub fn sample_momentum_into<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        momentum: &mut [f64],
        scratch: &mut [f64],
    ) {
        debug_assert_eq!(momentum.len(), self.dim);
        debug_assert!(scratch.len() >= self.dim);

        for block in &self.blocks {
            block.sample_momentum_into(rng, momentum, scratch);
        }
    }

    pub fn velocity_into(&self, momentum: &[f64], out: &mut [f64], scratch: &mut [f64]) {
        debug_assert_eq!(momentum.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        debug_assert!(scratch.len() >= self.dim);

        for block in &self.blocks {
            block.velocity_into(momentum, out, scratch);
        }
    }

    pub fn kinetic_energy(&self, momentum: &[f64], scratch: &mut [f64]) -> f64 {
        debug_assert_eq!(momentum.len(), self.dim);
        debug_assert!(scratch.len() >= self.dim);

        let mut ke = 0.0f64;
        for block in &self.blocks {
            ke += block.kinetic_energy(momentum, scratch);
        }
        ke
    }

    pub fn uturn(
        &self,
        left_q: &[f64],
        left_p: &[f64],
        right_q: &[f64],
        right_p: &[f64],
        scratch: &mut [f64],
    ) -> bool {
        debug_assert_eq!(left_q.len(), self.dim);
        debug_assert_eq!(right_q.len(), self.dim);
        debug_assert_eq!(left_p.len(), self.dim);
        debug_assert_eq!(right_p.len(), self.dim);
        debug_assert!(scratch.len() >= self.dim);

        let mut dot_left = 0.0f64;
        let mut dot_right = 0.0f64;
        for block in &self.blocks {
            let (dl, dr) = block.uturn_terms(left_q, left_p, right_q, right_p, scratch);
            dot_left += dl;
            dot_right += dr;
        }
        dot_left < 0.0 || dot_right < 0.0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl MassMatrixAccumulator {
    pub fn from_graph(graph: &Graph) -> Self {
        let mut blocks = Vec::with_capacity(graph.param_spans.len());
        for span in &graph.param_spans {
            blocks.push(AccumulatorBlock::new(span.start, span.len));
        }
        Self { blocks }
    }

    pub fn reset(&mut self) {
        for block in &mut self.blocks {
            block.reset();
        }
    }

    pub fn update(&mut self, q: &[f64]) {
        for block in &mut self.blocks {
            block.update(q);
        }
    }

    pub fn finalize(&self) -> MassMatrix {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            blocks.push(block.finalize());
        }
        let dim = blocks
            .iter()
            .map(|block| block.start + block.len)
            .max()
            .unwrap_or(0);
        MassMatrix { dim, blocks }
    }
}

impl MassBlock {
    fn identity(start: usize, len: usize) -> Self {
        let kind = if len == 1 {
            BlockKind::Scalar {
                variance: 1.0,
                inv_variance: 1.0,
            }
        } else if len <= DENSE_BLOCK_MAX_DIM {
            BlockKind::Dense {
                chol: identity_lower(len),
            }
        } else {
            BlockKind::Diagonal {
                variances: vec![1.0; len],
                inv_variances: vec![1.0; len],
            }
        };
        Self { start, len, kind }
    }

    fn sample_momentum_into<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        momentum: &mut [f64],
        scratch: &mut [f64],
    ) {
        let range = self.start..self.start + self.len;
        match &self.kind {
            BlockKind::Scalar { variance, .. } => {
                let z: f64 = StandardNormal.sample(rng);
                momentum[self.start] = z * variance.sqrt();
            }
            BlockKind::Diagonal { variances, .. } => {
                for (offset, &variance) in variances.iter().enumerate() {
                    let z: f64 = StandardNormal.sample(rng);
                    momentum[self.start + offset] = z * variance.sqrt();
                }
            }
            BlockKind::Dense { chol } => {
                let scratch_block = &mut scratch[range.clone()];
                for value in scratch_block.iter_mut() {
                    *value = StandardNormal.sample(rng);
                }
                dense_matvec_lower(chol, scratch_block, &mut momentum[range]);
            }
        }
    }

    fn velocity_into(&self, momentum: &[f64], out: &mut [f64], scratch: &mut [f64]) {
        let range = self.start..self.start + self.len;
        match &self.kind {
            BlockKind::Scalar { inv_variance, .. } => {
                out[self.start] = momentum[self.start] * inv_variance;
            }
            BlockKind::Diagonal { inv_variances, .. } => {
                for (offset, &inv_variance) in inv_variances.iter().enumerate() {
                    out[self.start + offset] = momentum[self.start + offset] * inv_variance;
                }
            }
            BlockKind::Dense { chol } => {
                let scratch_block = &mut scratch[range.clone()];
                scratch_block.copy_from_slice(&momentum[range.clone()]);
                solve_cholesky_in_place(chol, scratch_block);
                out[range].copy_from_slice(scratch_block);
            }
        }
    }

    fn kinetic_energy(&self, momentum: &[f64], scratch: &mut [f64]) -> f64 {
        let range = self.start..self.start + self.len;
        match &self.kind {
            BlockKind::Scalar { inv_variance, .. } => {
                0.5 * momentum[self.start] * momentum[self.start] * inv_variance
            }
            BlockKind::Diagonal { inv_variances, .. } => {
                let mut ke = 0.0f64;
                for (offset, &inv_variance) in inv_variances.iter().enumerate() {
                    let p = momentum[self.start + offset];
                    ke += 0.5 * p * p * inv_variance;
                }
                ke
            }
            BlockKind::Dense { chol } => {
                let scratch_block = &mut scratch[range.clone()];
                scratch_block.copy_from_slice(&momentum[range.clone()]);
                solve_cholesky_in_place(chol, scratch_block);
                let mut ke = 0.0f64;
                for (p, v) in momentum[range].iter().zip(scratch_block.iter()) {
                    ke += 0.5 * p * v;
                }
                ke
            }
        }
    }

    fn uturn_terms(
        &self,
        left_q: &[f64],
        left_p: &[f64],
        right_q: &[f64],
        right_p: &[f64],
        scratch: &mut [f64],
    ) -> (f64, f64) {
        let range = self.start..self.start + self.len;
        match &self.kind {
            BlockKind::Scalar { inv_variance, .. } => {
                let dq = right_q[self.start] - left_q[self.start];
                let v_left = left_p[self.start] * inv_variance;
                let v_right = right_p[self.start] * inv_variance;
                (dq * v_left, dq * v_right)
            }
            BlockKind::Diagonal { inv_variances, .. } => {
                let mut dot_left = 0.0f64;
                let mut dot_right = 0.0f64;
                for (offset, &inv_variance) in inv_variances.iter().enumerate() {
                    let delta = right_q[self.start + offset] - left_q[self.start + offset];
                    dot_left += delta * (left_p[self.start + offset] * inv_variance);
                    dot_right += delta * (right_p[self.start + offset] * inv_variance);
                }
                (dot_left, dot_right)
            }
            BlockKind::Dense { chol } => {
                let scratch_block = &mut scratch[range.clone()];
                scratch_block.copy_from_slice(&left_p[range.clone()]);
                solve_cholesky_in_place(chol, scratch_block);
                let mut dot_left = 0.0f64;
                for (offset, &v) in scratch_block.iter().enumerate() {
                    let dq = right_q[self.start + offset] - left_q[self.start + offset];
                    dot_left += dq * v;
                }

                scratch_block.copy_from_slice(&right_p[range.clone()]);
                solve_cholesky_in_place(chol, scratch_block);
                let mut dot_right = 0.0f64;
                for (offset, &v) in scratch_block.iter().enumerate() {
                    let dq = right_q[self.start + offset] - left_q[self.start + offset];
                    dot_right += dq * v;
                }

                (dot_left, dot_right)
            }
        }
    }
}

impl AccumulatorBlock {
    fn new(start: usize, len: usize) -> Self {
        if len == 1 {
            Self::Scalar {
                start,
                count: 0,
                mean: 0.0,
                m2: 0.0,
            }
        } else if len <= DENSE_BLOCK_MAX_DIM {
            Self::Dense {
                start,
                dim: len,
                count: 0,
                mean: vec![0.0; len],
                m2: vec![0.0; len * len],
            }
        } else {
            Self::Diagonal {
                start,
                count: 0,
                mean: vec![0.0; len],
                m2: vec![0.0; len],
            }
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Scalar {
                count, mean, m2, ..
            } => {
                *count = 0;
                *mean = 0.0;
                *m2 = 0.0;
            }
            Self::Diagonal {
                count, mean, m2, ..
            } => {
                *count = 0;
                mean.fill(0.0);
                m2.fill(0.0);
            }
            Self::Dense {
                count, mean, m2, ..
            } => {
                *count = 0;
                mean.fill(0.0);
                m2.fill(0.0);
            }
        }
    }

    fn update(&mut self, q: &[f64]) {
        match self {
            Self::Scalar {
                start,
                count,
                mean,
                m2,
            } => {
                let x = q[*start];
                *count += 1;
                let n = *count as f64;
                let delta = x - *mean;
                *mean += delta / n;
                let delta2 = x - *mean;
                *m2 += delta * delta2;
            }
            Self::Diagonal {
                start,
                count,
                mean,
                m2,
            } => {
                *count += 1;
                let n = *count as f64;
                for (offset, mean_i) in mean.iter_mut().enumerate() {
                    let x = q[*start + offset];
                    let delta = x - *mean_i;
                    *mean_i += delta / n;
                    let delta2 = x - *mean_i;
                    m2[offset] += delta * delta2;
                }
            }
            Self::Dense {
                start,
                dim,
                count,
                mean,
                m2,
            } => {
                *count += 1;
                let n = *count as f64;
                let mut delta = vec![0.0; *dim];
                let mut delta2 = vec![0.0; *dim];
                for i in 0..*dim {
                    let x = q[*start + i];
                    delta[i] = x - mean[i];
                }
                for i in 0..*dim {
                    mean[i] += delta[i] / n;
                    delta2[i] = q[*start + i] - mean[i];
                }
                for i in 0..*dim {
                    for j in 0..*dim {
                        m2[i * *dim + j] += delta[i] * delta2[j];
                    }
                }
            }
        }
    }

    fn finalize(&self) -> MassBlock {
        match self {
            Self::Scalar {
                start, count, m2, ..
            } => {
                let variance = regularize_variance(
                    if *count > 1 {
                        *m2 / (*count as f64 - 1.0)
                    } else {
                        1.0
                    },
                    *count,
                );
                MassBlock {
                    start: *start,
                    len: 1,
                    kind: BlockKind::Scalar {
                        variance,
                        inv_variance: 1.0 / variance,
                    },
                }
            }
            Self::Diagonal {
                start, count, m2, ..
            } => {
                let len = m2.len();
                let mut variances = vec![1.0; len];
                let mut inv_variances = vec![1.0; len];
                for i in 0..len {
                    let variance = regularize_variance(
                        if *count > 1 {
                            m2[i] / (*count as f64 - 1.0)
                        } else {
                            1.0
                        },
                        *count,
                    );
                    variances[i] = variance;
                    inv_variances[i] = 1.0 / variance;
                }
                MassBlock {
                    start: *start,
                    len,
                    kind: BlockKind::Diagonal {
                        variances,
                        inv_variances,
                    },
                }
            }
            Self::Dense {
                start,
                dim,
                count,
                mean,
                m2,
            } => {
                let mut cov = vec![0.0; dim * dim];
                if *count > 1 {
                    let scale = 1.0 / (*count as f64 - 1.0);
                    for i in 0..*dim {
                        for j in 0..*dim {
                            cov[i * *dim + j] = m2[i * *dim + j] * scale;
                        }
                    }
                } else {
                    for i in 0..*dim {
                        cov[i * *dim + i] = 1.0;
                    }
                }

                let shrink = if *count > 0 {
                    *count as f64 / (*count as f64 + REGULARIZATION_WEIGHT)
                } else {
                    0.0
                };
                let jitter =
                    BASE_JITTER * (REGULARIZATION_WEIGHT / (*count as f64 + REGULARIZATION_WEIGHT));
                for i in 0..*dim {
                    for j in 0..*dim {
                        cov[i * *dim + j] *= shrink;
                    }
                    cov[i * *dim + i] += jitter;
                }

                let chol = cholesky_with_jitter(cov, *dim);
                let _ = mean;
                MassBlock {
                    start: *start,
                    len: *dim,
                    kind: BlockKind::Dense { chol },
                }
            }
        }
    }
}

fn regularize_variance(variance: f64, count: usize) -> f64 {
    let n = count as f64;
    let shrunk = if count > 0 {
        (n / (n + REGULARIZATION_WEIGHT)) * variance
            + BASE_JITTER * (REGULARIZATION_WEIGHT / (n + REGULARIZATION_WEIGHT))
    } else {
        1.0
    };
    shrunk.max(1e-12)
}

fn identity_lower(dim: usize) -> Vec<f64> {
    let mut chol = vec![0.0; dim * dim];
    for i in 0..dim {
        chol[i * dim + i] = 1.0;
    }
    chol
}

fn dense_matvec_lower(lower: &[f64], x: &[f64], out: &mut [f64]) {
    let dim = x.len();
    for i in 0..dim {
        let mut sum = 0.0f64;
        for j in 0..=i {
            sum += lower[i * dim + j] * x[j];
        }
        out[i] = sum;
    }
}

fn solve_cholesky_in_place(lower: &[f64], rhs: &mut [f64]) {
    let dim = rhs.len();
    for i in 0..dim {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lower[i * dim + j] * rhs[j];
        }
        rhs[i] = sum / lower[i * dim + i];
    }
    for i in (0..dim).rev() {
        let mut sum = rhs[i];
        for j in i + 1..dim {
            sum -= lower[j * dim + i] * rhs[j];
        }
        rhs[i] = sum / lower[i * dim + i];
    }
}

fn cholesky_with_jitter(cov: Vec<f64>, dim: usize) -> Vec<f64> {
    let mut jitter = BASE_JITTER;
    for _ in 0..8 {
        let mut candidate = cov.clone();
        for i in 0..dim {
            candidate[i * dim + i] += jitter;
        }
        if cholesky_lower_in_place(&mut candidate, dim) {
            return candidate;
        }
        jitter *= 10.0;
    }

    identity_lower(dim)
}

fn cholesky_lower_in_place(a: &mut [f64], dim: usize) -> bool {
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = a[i * dim + j];
            for k in 0..j {
                sum -= a[i * dim + k] * a[j * dim + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return false;
                }
                a[i * dim + j] = sum.sqrt();
            } else {
                let diag = a[j * dim + j];
                if !diag.is_finite() || diag <= 0.0 {
                    return false;
                }
                a[i * dim + j] = sum / diag;
            }
        }
        for j in i + 1..dim {
            a[i * dim + j] = 0.0;
        }
    }
    true
}
