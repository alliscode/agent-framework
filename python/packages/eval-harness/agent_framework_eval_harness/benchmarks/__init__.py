# Copyright (c) Microsoft. All rights reserved.

"""Built-in benchmark adapters."""

from ._gaia import GAIABenchmark, gaia_scorer
from ._tau2 import TauBenchmark

__all__ = [
    "GAIABenchmark",
    "TauBenchmark",
    "gaia_scorer",
]
