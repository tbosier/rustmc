# Repeated calibration

Fit independent instrument datasets with one compiled regression model. Each can have
a different number of readings. Stable IDs keep random streams tied to instruments
when the batch order changes.

Run `python examples/repeated_calibration.py` from the repository root.
The [script](https://github.com/tbosier/rustmc/blob/main/examples/repeated_calibration.py)
prints each fit's diagnostics and collects errors by instrument ID.

This does not pool instruments. Use [site effects](site-effects.md) when group estimates
should share information. Chunked batch dispatch currently retains inputs and fits;
see the roadmap for bounded streaming.
