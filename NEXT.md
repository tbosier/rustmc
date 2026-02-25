What's needed next (ordered by impact)
# Tier 1 — Credibility
## NUTS sampler
     — Without this, any benchmark comparison is questioned. NUTS with dual averaging + mass matrix adaptation is the standard. This is the single most important missing piece.
## Diagnostic metrics 
    — R-hat, effective sample size (ESS), BFMO diagnostic. Users need to know if the chains converged. Without these, no statistician will trust the output.
## More distributions 
    — HalfNormal, StudentT, Uniform, Bernoulli, Poisson, Beta, Gamma. The distribution catalog is what determines what models people can express.
# Tier 2 — Production readiness
## Model serialization 
    — Compile a model graph once, serialize to disk, load in production without Python. This is a killer feature for deployment pipelines: data scientists define models in Python, ops deploys a Rust binary.
## Streaming/online posterior updates 
    — Start from an existing posterior and incorporate new data without re-sampling from scratch. Critical for real-time systems (fraud detection, dynamic pricing, sensor fusion).
## C API / FFI 
    — Expose rustmc_core via C headers so C++, Go, Julia, and other languages can use it. Broader adoption.
## Predictive sampling 
    — posterior_predictive() is how users validate models. Without it, the library is incomplete for the standard Bayesian workflow.
# Tier 3 — Competitive moat
## Batched multi-model execution 
    — Run 10K independent small models in a single call, sharing the thread pool. No Python equivalent can do this efficiently. Think: hierarchical marketing mix models per region, per-SKU demand forecasting, per-patient pharmacokinetic models.
## WASM compilation 
    — Rust compiles to WebAssembly. Bayesian inference in the browser, in Cloudflare Workers, in edge functions. PyMC will never do this.
## GPU-accelerated log-probability 
    — For very large observation counts (1M+), offload the likelihood sum to the GPU via wgpu/CUDA. The graph is already structured for this.
## Sparse/block mass matrices 
    — For large hierarchical models (10K+ parameters), a full diagonal mass matrix isn't enough. Block-diagonal structure that mirrors the model hierarchy would give Stan-level sampling quality.
## Automatic reparameterization 
    — Detect funnels and other pathologies in the posterior geometry, automatically apply non-centered parameterizations. This is research-level but would be genuinely novel.