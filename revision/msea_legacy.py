"""Importable wrapper around the exact Kaggle notebook code (kaggle.com/code/absanim/msea-net).

The notebook's five cells are byte-identical to ``phase1_setup_and_data.py`` ...
``phase5_evaluation_xai.py`` (apart from the Kaggle dataset path).  Every number in the
manuscript was produced by that code, so the revision experiments reuse it unchanged.

This module executes only the *imports, function definitions and class definitions* of
phases 1-3 (data pipeline, model, training/evaluation utilities).  Kaggle-specific
top-level statements (pip installs, dataset discovery, figure generation, the demo
forward pass) are skipped.  Because the definitions live in a real importable module,
PyTorch DataLoader workers can unpickle them on Windows (spawn start method).

Usage::

    import msea_legacy as L
    L.Config.NUM_WORKERS = 8
    paths, labels, classes = L.load_dataset_paths(root)
    train_idx, val_idx, test_idx = L.create_data_splits(paths, labels, seed=42)
"""
import ast
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get("MSEA_LEGACY_ROOT", os.path.dirname(_HERE))
_FILES = ("phase1_setup_and_data.py", "phase2_model.py", "phase3_train_utils.py")


def _is_import_block(node):
    """True for the AMP-compatibility ``try: from torch.amp import ...`` block of phase 1."""
    return isinstance(node, ast.Try) and all(
        isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for n in node.body
    )


def _load(namespace):
    for fname in _FILES:
        path = os.path.join(_ROOT, fname)
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        keep = [
            n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
            or _is_import_block(n)
        ]
        exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), namespace)


_load(globals())

# Same non-interactive plotting backend as the notebook.
matplotlib.use("Agg")  # noqa: F821  (imported by phase 1)

LEGACY_SOURCES = tuple(os.path.join(_ROOT, f) for f in _FILES)
