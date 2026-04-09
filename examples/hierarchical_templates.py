"""Reusable hierarchical-model helpers.

This module captures the centered partial-pooling pattern that rustmc can
already express today. It gives users a stable template boundary now, while
the future non-centered API can slot in behind the same conceptual surface
once the DSL grows parameter-to-parameter transforms.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CenteredNormalHierarchy:
    mu_global: object
    sigma_group: object
    group_params: tuple[object, ...]


def build_centered_normal_partial_pooling(
    builder,
    *,
    observed_keys: Sequence[str],
    sigma_obs: float,
    mu_global_name: str = "mu_global",
    mu_global_mu: float = 0.0,
    mu_global_sigma: float = 10.0,
    sigma_group_name: str = "sigma_group",
    sigma_group_sigma: float = 5.0,
    group_prefix: str = "mu_",
    likelihood_prefix: str = "obs_",
) -> CenteredNormalHierarchy:
    """Build the centered 8-schools style hierarchy used by the examples.

    The current rustmc DSL can represent scalar hyperpriors and scalar
    hierarchical group means. This helper keeps that pattern reusable and
    isolates the eventual non-centered API change to one place.
    """

    mu_global = builder.normal_prior(mu_global_name, mu=mu_global_mu, sigma=mu_global_sigma)
    sigma_group = builder.half_normal_prior(sigma_group_name, sigma=sigma_group_sigma)

    group_params = []
    for idx, observed_key in enumerate(observed_keys):
        group_param = builder.normal_prior(
            f"{group_prefix}{idx}",
            mu=mu_global,
            sigma=sigma_group,
        )
        builder.normal_likelihood(
            f"{likelihood_prefix}{idx}",
            mu_expr=group_param,
            sigma=sigma_obs,
            observed_key=observed_key,
        )
        group_params.append(group_param)

    return CenteredNormalHierarchy(
        mu_global=mu_global,
        sigma_group=sigma_group,
        group_params=tuple(group_params),
    )
