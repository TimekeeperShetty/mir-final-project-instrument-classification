from __future__ import annotations

from typing import TYPE_CHECKING

# We keep this package import light on purpose.
#
# Why: commands like
#   python -m training_base.data.datasets
# first import the `training_base.data` package before they import the
# `datasets` submodule. If we eagerly import the datamodule here, that drags in
# Lightning immediately, even for utilities that only need dataset code.
#
# That makes simple dataset helpers depend on the full training stack and can
# fail in environments where Lightning extras are not installed yet.
#
# So we expose InstrumentDataModule lazily instead.
if TYPE_CHECKING:
    from .datamodule import InstrumentDataModule

__all__ = ["InstrumentDataModule"]


def __getattr__(name: str):
    if name == "InstrumentDataModule":
        from .datamodule import InstrumentDataModule

        return InstrumentDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
