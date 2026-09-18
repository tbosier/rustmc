"""`summary()` must stay column-aligned for parameter names of any length.

The built-in models emit names such as ``observation_variance`` (20 chars),
so the rendered table is exercised here with real names rather than the short
synthetic ones used by the Rust unit tests.
"""

import numpy as np


def _fit(rmc):
    prior = rmc.InverseGammaPrior(shape=2.5, scale=0.3)
    model = rmc.BayesianLocalLevel(
        process_variance_prior=prior,
        observation_variance_prior=prior,
        initial_mean=0.0,
        initial_variance=4.0,
    )
    rng = np.random.default_rng(11)
    observations = np.cumsum(rng.normal(scale=0.3, size=40))
    return model.fit(observations, chains=2, draws=120, warmup=60, seed=7)


def _fields(line):
    """(start, end) character offsets of each whitespace-delimited field."""
    fields = []
    start = None
    for index, char in enumerate(line):
        if char.isspace():
            if start is not None:
                fields.append((start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        fields.append((start, len(line)))
    return fields


def _table_rows(summary):
    """Header row, body rows and rule width of the rendered summary table."""
    lines = summary.splitlines()
    rules = [index for index, line in enumerate(lines) if line.startswith("─")]
    assert len(rules) == 2, f"expected two horizontal rules, got {len(rules)}:\n{summary}"
    top, bottom = rules
    return lines[top - 1], lines[top + 1 : bottom], len(lines[top])


def test_summary_columns_line_up_for_long_parameter_names(rustmc_module):
    summary = _fit(rustmc_module).summary()
    header, body, rule_width = _table_rows(summary)

    assert body, f"no parameter rows rendered:\n{summary}"
    names = [row.split()[0] for row in body]
    assert any(len(name) > 12 for name in names), (
        f"this test only proves anything with names longer than 12 chars, got {names}"
    )

    expected = _fields(header)
    for row in [header] + body:
        assert len(row) == rule_width, (
            f"row width {len(row)} disagrees with rule width {rule_width}:\n{row}\n{summary}"
        )
        fields = _fields(row)
        assert len(fields) == len(expected), f"wrong number of columns:\n{row}\n{summary}"
        # First column is left aligned; the rest are right aligned, so their
        # fields must end at the same offsets as the header's.
        assert fields[0][0] == 0, f"first column is not flush left:\n{row}"
        for index, (field, reference) in enumerate(zip(fields[1:], expected[1:]), start=1):
            assert field[1] == reference[1], (
                f"column {index} ends at {field[1]} in\n{row}\n"
                f"but at {reference[1]} in\n{header}"
            )


# Historical fixed widths, which the renderer now treats as minimums.
_MIN_WIDTHS = [12, 8, 8, 10, 10, 10, 10, 8, 10]


def test_summary_columns_are_sized_to_their_contents(rustmc_module):
    """Independently re-derive the layout and compare it to what was rendered.

    This pins the column widths in both directions: a name column that is too
    narrow (the bug) and one padded wider than it needs to be both fail.
    """
    summary = _fit(rustmc_module).summary()
    header, body, rule_width = _table_rows(summary)

    cells = [row.split() for row in [header] + body]
    assert all(len(row) == len(_MIN_WIDTHS) for row in cells), (
        f"expected {len(_MIN_WIDTHS)} columns per row:\n{summary}"
    )

    widths = [
        max([minimum] + [len(row[index]) for row in cells])
        for index, minimum in enumerate(_MIN_WIDTHS)
    ]
    assert widths[0] > 12, f"long names should widen the name column:\n{summary}"
    assert rule_width == sum(widths) + len(widths) - 1, (
        f"rule width {rule_width} does not match the column widths {widths}:\n{summary}"
    )

    for row, source in zip(cells, [header] + body):
        rendered = " ".join(
            [row[0].ljust(widths[0])]
            + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])]
        )
        assert source == rendered, f"row is not laid out as expected:\n{source!r}\n{rendered!r}"
