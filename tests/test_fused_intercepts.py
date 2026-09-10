"""Fused regression intercepts must retain their expression type."""

import numpy as np
import pytest
import rustmc


def normal_log_density(values, mean, sigma):
    residual = (np.asarray(values) - mean) / sigma
    return np.sum(-0.5 * residual**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi))


@pytest.mark.parametrize('name', ['__const__5', '__const__not_a_number'])
def test_constant_like_parameter_name_keeps_intercept_value_and_gradient(name):
    x = np.array([-1., 0.25, 2.])
    y = np.array([0.1, -0.7, 1.4])
    sigma = 1.3
    builder = rustmc.ModelBuilder({'x': x, 'y': y})
    intercept = builder.normal_prior(name, 0., 1.)
    slope = builder.normal_prior('slope', 0., 1.)
    builder.normal_likelihood('obs', intercept + slope * 'x', sigma, 'y')
    compiled = builder.compile()

    for position in [np.array([0.4, -0.2]), np.array([-0.8, 0.3])]:
        a, b = position
        residual = y - (a + b*x)
        expected_density = normal_log_density(position, 0., 1.) + normal_log_density(y, a+b*x, sigma)
        expected_gradient = np.array([-a + residual.sum()/sigma**2,
                                      -b + x.dot(residual)/sigma**2])
        density, gradient = compiled.log_density({}, position)
        assert density == pytest.approx(expected_density, abs=1e-12)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_literal_constant_fused_intercept_keeps_density_and_gradient():
    x = np.array([-1., 0.25, 2.])
    y = np.array([0.1, -0.7, 1.4])
    intercept, sigma = 1.75, 1.3
    builder = rustmc.ModelBuilder({'x': x, 'y': y})
    slope = builder.normal_prior('slope', 0., 1.)
    builder.normal_likelihood('obs', intercept + slope * 'x', sigma, 'y')
    compiled = builder.compile()
    position = np.array([-0.2])
    mean = intercept + position[0]*x
    expected_density = normal_log_density(position, 0., 1.) + normal_log_density(y, mean, sigma)
    expected_gradient = -position[0] + x.dot(y-mean)/sigma**2
    density, gradient = compiled.log_density({}, position)
    assert density == pytest.approx(expected_density, abs=1e-12)
    assert gradient[0] == pytest.approx(expected_gradient, abs=1e-12)
