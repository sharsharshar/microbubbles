from __future__ import annotations

import os
import pickle
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def thread_map(fn: Callable[[T], R], items: Iterable[T], max_workers: int | None = None) -> list[R]:
    """``list(map(fn, items))`` on a thread pool, for work that releases the GIL
    (scipy.ndimage filters, numpy reductions, h5py reads, ``os.pread``)."""
    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as pool:
        return list(pool.map(fn, items))


class NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy-2-authored pickles from NumPy-1 runtimes."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(path: str | Path):
    with Path(path).open("rb") as fp:
        return NumpyCompatUnpickler(fp).load()


def dump_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(obj, fp)
