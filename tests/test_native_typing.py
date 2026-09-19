"""Keep the maintained native stub aligned with the extension shipped in wheels."""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path



def _stub():
    package = importlib.import_module("rustmc")
    path = Path(package.__file__).resolve().parent / "_rustmc.pyi"
    return ast.parse(path.read_text())


def _native():
    return importlib.import_module("rustmc._rustmc")


def _public(name):
    return not name.startswith("_") or name in {"__enter__", "__exit__", "__getitem__", "__len__"}


def test_native_stub_covers_public_exports_and_members():
    tree, native = _stub(), _native()
    declarations = {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    aliases = {
        target.id: node.value.id
        for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
        for target in node.targets if isinstance(target, ast.Name)
    }
    missing = [name + " (stub only)" for name in declarations if _public(name) and not hasattr(native, name)]
    for name in dir(native):
        if not _public(name):
            continue
        definition = declarations.get(aliases.get(name, name))
        if definition is None:
            missing.append(name)
            continue
        value = getattr(native, name)
        if isinstance(definition, ast.ClassDef) and inspect.isclass(value):
            stub_members = {node.name for node in definition.body if isinstance(node, ast.FunctionDef) and _public(node.name)}
            runtime_members = {member for member in vars(value) if _public(member)}
            missing.extend(f"{name}.{member}" for member in runtime_members - stub_members)
            # Do not advertise unavailable methods/properties or proposed future APIs.
            missing.extend(f"{name}.{member} (stub only)" for member in stub_members - runtime_members
                           if member != "__init__")
    assert not missing, "Update _rustmc.pyi for native API changes: " + ", ".join(sorted(missing))


def test_native_stub_keyword_parameters_and_defaults_match_runtime():
    tree, native = _stub(), _native()
    mismatches = []
    for definition in tree.body:
        if isinstance(definition, ast.ClassDef) and hasattr(native, definition.name):
            owner = getattr(native, definition.name)
            # The final overload is the general public signature.
            methods = {node.name: node for node in definition.body if isinstance(node, ast.FunctionDef)}
            candidates = [(f"{definition.name}.{name}", owner if name == "__init__" else getattr(owner, name, None), node)
                          for name, node in methods.items() if name == "__init__" or not name.startswith("_")]
        elif isinstance(definition, ast.FunctionDef) and hasattr(native, definition.name):
            candidates = [(definition.name, getattr(native, definition.name), definition)]
        else:
            continue
        for label, value, node in candidates:
            if any(isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in node.decorator_list):
                continue
            try:
                runtime = inspect.signature(value)
            except (TypeError, ValueError):
                continue  # Some CPython method descriptors have no text signature.
            args = node.args.posonlyargs + node.args.args
            defaults = [None] * (len(args) - len(node.args.defaults)) + node.args.defaults
            signature = dict(zip((a.arg for a in args), defaults))
            signature.update(zip((a.arg for a in node.args.kwonlyargs), node.args.kw_defaults))
            signature.pop("self", None)
            expected = {name: parameter for name, parameter in runtime.parameters.items() if name not in {"self", "cls"}}
            if set(signature) != set(expected):
                mismatches.append(f"{label}: stub={sorted(signature)}, native={sorted(expected)}")
                continue
            keyword_only = {argument.arg for argument in node.args.kwonlyargs}
            for name, default in signature.items():
                parameter = expected[name]
                if (name in keyword_only) != (parameter.kind == inspect.Parameter.KEYWORD_ONLY):
                    mismatches.append(f"{label}.{name}: keyword-only status differs")
                if default is None:
                    if parameter.default is not inspect.Parameter.empty:
                        mismatches.append(f"{label}.{name}: missing native default")
                elif isinstance(default, ast.Constant) and default.value is not Ellipsis:
                    if parameter.default is inspect.Parameter.empty or default.value != parameter.default:
                        mismatches.append(f"{label}.{name}: default differs")
    assert not mismatches, "Update native stub signatures: " + "; ".join(mismatches)


# ── two things the checks above cannot see ────────────────────────────────
#
# `test_native_stub_covers_public_exports_and_members` compares member *names*,
# and the signature check skips anything it cannot call `inspect.signature` on
# -- which is every getset descriptor PyO3 emits for `#[getter]`. So a stub that
# declares a property as a plain method, or that promises the wrong return
# type, passes both. Those are exactly the shapes that drift when a binding is
# reworked rather than added to, which is what happened to the ArviZ export,
# the batch accessors and the prior predictive on this branch.


def _stub_classes():
    tree, native = _stub(), _native()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and hasattr(native, node.name):
            yield node, getattr(native, node.name)


def test_native_stub_declares_properties_and_static_methods_as_such():
    """A `#[getter]` must be a `@property` in the stub, and vice versa."""
    wrong = []
    for node, owner in _stub_classes():
        for member in node.body:
            if not isinstance(member, ast.FunctionDef) or member.name == "__init__":
                continue
            runtime = inspect.getattr_static(owner, member.name, None)
            if runtime is None:
                continue
            decorators = {
                decorator.id
                for decorator in member.decorator_list
                if isinstance(decorator, ast.Name)
            }
            kind = type(runtime).__name__
            declared = "property" in decorators
            actual = kind in {"getset_descriptor", "property"}
            if declared != actual:
                wrong.append(
                    f"{node.name}.{member.name}: stub "
                    f"{'declares' if declared else 'does not declare'} a property, "
                    f"runtime is a {kind}"
                )
                continue
            declared_static = "staticmethod" in decorators
            actual_static = kind in {"staticmethod", "builtin_function_or_method"}
            if not actual and declared_static != actual_static:
                wrong.append(
                    f"{node.name}.{member.name}: stub "
                    f"{'declares' if declared_static else 'does not declare'} a "
                    f"staticmethod, runtime is a {kind}"
                )
    assert not wrong, "Update _rustmc.pyi member kinds: " + "; ".join(sorted(wrong))


_ARRAY_DTYPES = {
    "_FloatArray": "float64",
    "_UIntArray": "uint64",
    "_BoolArray": "bool",
}


def _split_type_arguments(inner):
    """Split `dict[str, list[int]]`'s inner text on its top-level commas."""
    parts, depth, start = [], 0, 0
    for index, character in enumerate(inner):
        depth += (character == "[") - (character == "]")
        if character == "," and depth == 0:
            parts.append(inner[start:index])
            start = index + 1
    parts.append(inner[start:])
    return [part.strip() for part in parts]


def _annotation_is_known(annotation):
    """Whether `_value_matches` can decide this annotation *at all*.

    Checked separately from the value, because a container's element
    annotation is never reached when the runtime container is empty -- an
    empty `dict[str, DoesNotExist]` would otherwise pass as verified.
    """
    annotation = annotation.strip()
    if annotation in {"Any", "None", "str", "bool", "int", "float"}:
        return True
    if annotation in _ARRAY_DTYPES:
        return True
    if "|" in annotation:
        return all(_annotation_is_known(part) for part in annotation.split("|"))
    for prefix, arity in (("list[", 1), ("dict[", 2), ("tuple[", None)):
        if annotation.startswith(prefix) and annotation.endswith("]"):
            arguments = _split_type_arguments(annotation[len(prefix):-1])
            if arity is not None and len(arguments) != arity:
                return False
            return all(_annotation_is_known(argument) for argument in arguments)
    if annotation.startswith("_Diagnostic"):
        return True
    return isinstance(getattr(importlib.import_module("rustmc"), annotation, None), type)


def _value_matches(value, annotation):
    """Structural match of a runtime value against a stub return annotation.

    Assumes `_annotation_is_known(annotation)`; the caller checks that first,
    so an unrecognised form is reported as unchecked rather than passing.
    """
    import numpy as np

    annotation = annotation.strip()
    if annotation == "Any":
        return True
    if annotation == "None":
        return value is None
    if "|" in annotation:
        return any(_value_matches(value, part) for part in annotation.split("|"))
    if annotation in _ARRAY_DTYPES:
        # The dtype is half of what these aliases promise; a float64 array is
        # not an acceptable stand-in for a declared bool array.
        return (
            isinstance(value, np.ndarray)
            and value.dtype == np.dtype(_ARRAY_DTYPES[annotation])
        )
    if annotation == "str":
        return isinstance(value, str)
    if annotation == "bool":
        return isinstance(value, bool)
    if annotation == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation.startswith("list["):
        return isinstance(value, list) and all(
            _value_matches(item, annotation[5:-1]) for item in value
        )
    if annotation.startswith("dict["):
        key_type, value_type = _split_type_arguments(annotation[5:-1])
        return (
            isinstance(value, dict)
            and all(_value_matches(key, key_type) for key in value)
            and all(_value_matches(item, value_type) for item in value.values())
        )
    if annotation.startswith("tuple["):
        arguments = _split_type_arguments(annotation[6:-1])
        return (
            isinstance(value, tuple)
            and len(value) == len(arguments)
            and all(
                _value_matches(item, argument)
                for item, argument in zip(value, arguments)
            )
        )
    if annotation.startswith("_Diagnostic"):
        return isinstance(value, dict)
    return isinstance(value, getattr(importlib.import_module("rustmc"), annotation))


def _live_objects():
    """One live instance of each class whose accessors this check can reach.

    Scoping this to "the classes one branch reworked" is what let
    `KalmanFilterResult.log_likelihood` and `KalmanSmootherResult.log_likelihood`
    sit annotated as `dict[str, _FloatArray]` while returning a float: neither
    class was in the set, so nothing compared them. Add a class here whenever
    one instance of it can be built cheaply, rather than when it changes.
    """
    import numpy as np

    rustmc = importlib.import_module("rustmc")
    rng = np.random.default_rng(0)
    n = 24
    x = rng.normal(size=n)
    data = {"x": x, "y": 1.0 + 2.0 * x + rng.normal(scale=0.3, size=n)}

    builder = rustmc.ModelBuilder(data)
    alpha = builder.normal_prior("alpha", 0.0, 5.0)
    beta = builder.normal_prior("beta", 0.0, 5.0)
    sigma = builder.half_normal_prior("sigma", 1.0)
    builder.deterministic("mu", alpha + beta * "x")
    builder.normal_likelihood("obs", alpha + beta * "x", sigma, "y")
    compiled = builder.compile()

    options = dict(chains=2, draws=25, warmup=25, seed=1, show_progress=False)
    fit = rustmc.sample(builder.build(), **options)
    batch = compiled.sample_batch([data], ids=["cell"], **options)
    series = np.cumsum(rng.normal(size=n)) + rng.normal(scale=0.2, size=n)
    system = rustmc.LinearGaussianStateSpace.local_level(0.1, 0.3)

    return {
        "FitResult": fit,
        "BatchFit": batch,
        "BatchResult": batch[0],
        "KalmanFilterResult": system.filter(series),
        "KalmanSmootherResult": system.smooth(series),
    }


def test_native_stub_return_types_match_the_live_classes():
    """Every accessor reachable with no arguments returns what the stub promises.

    Covers the classes `_live_objects` can instantiate, and members callable
    with no arguments. `to_arviz` is excluded on purpose: it returns `Any`, so
    there is nothing to compare.
    """
    tree = _stub()
    objects = _live_objects()
    wrong, unchecked = [], []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in objects:
            continue
        instance = objects[node.name]
        for member in node.body:
            if not isinstance(member, ast.FunctionDef) or member.name.startswith("_"):
                continue
            if member.returns is None:
                continue
            positional = [a for a in member.args.args if a.arg != "self"]
            if len(positional) > len(member.args.defaults):
                continue  # needs arguments; out of scope here
            if any(default is None for default in member.args.kw_defaults):
                continue
            decorators = {
                decorator.id
                for decorator in member.decorator_list
                if isinstance(decorator, ast.Name)
            }
            annotation = ast.unparse(member.returns)
            if annotation == "Any":
                continue
            label = f"{node.name}.{member.name}"
            # Decided before the value is looked at, so an empty list or dict
            # cannot let an unresolvable element annotation through.
            if not _annotation_is_known(annotation):
                unchecked.append(f"{label} ({annotation})")
                continue
            value = getattr(instance, member.name)
            if "property" not in decorators:
                value = value()
            if not _value_matches(value, annotation):
                wrong.append(f"{label}: stub says {annotation}, got {type(value).__name__}")
    assert not wrong, "Update native stub return types: " + "; ".join(sorted(wrong))
    # A form this check cannot decide is a gap in the check, not a pass.
    assert not unchecked, (
        "_value_matches cannot decide these annotations, so they are unverified: "
        + ", ".join(sorted(unchecked))
    )


def test_prior_predictive_return_type_matches_the_stub():
    """`sample_prior_predictive` is a module function, not a class member."""
    import numpy as np

    rustmc = importlib.import_module("rustmc")
    builder = rustmc.ModelBuilder({"x": np.arange(4, dtype=float)})
    alpha = builder.normal_prior("alpha", 0.0, 1.0)
    builder.deterministic("signal", alpha * "x")
    draws = rustmc.sample_prior_predictive(builder.build(), n_samples=8, seed=3)

    declared = next(
        node for node in _stub().body
        if isinstance(node, ast.FunctionDef) and node.name == "sample_prior_predictive"
    )
    assert ast.unparse(declared.returns) == "dict[str, _FloatArray]"
    assert _value_matches(draws, "dict[str, _FloatArray]")
