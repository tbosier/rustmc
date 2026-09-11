# Site effects

A partial-pooling model estimates a shared population mean, a between-site scale,
and each site's deviation. Sites with fewer readings can borrow information from
the population. The example uses a noncentered parameterization and unequal counts.

Run `python examples/site_effects.py` from the repository root.
The [script](https://github.com/tbosier/rustmc/blob/main/examples/site_effects.py)
prints diagnostics, site intervals, and the posterior probability that one site's
mean exceeds another's.

Comparisons use paired posterior draws. Independently shuffling or fitting sites
would lose the joint dependence. The observation noise is known in this example;
a real model can assign it a positive prior.
