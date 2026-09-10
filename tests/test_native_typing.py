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
