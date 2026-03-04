# Copyright (c) Microsoft. All rights reserved.

import importlib.metadata

from ._agent_provider import AzureAIAgentsProvider
from ._chat_client import AzureAIAgentClient, AzureAIAgentOptions
from ._client import AzureAIClient, AzureAIProjectAgentOptions, RawAzureAIClient
from ._foundry_evals import (
    AgentEvalConverter,
    EvalResults,
    Evaluators,
    evaluate_agent,
    evaluate_dataset,
    evaluate_foundry_target,
    evaluate_response,
    evaluate_traces,
    evaluate_workflow,
    setup_continuous_eval,
)
from ._foundry_memory_provider import FoundryMemoryProvider
from ._project_provider import AzureAIProjectAgentProvider
from ._shared import AzureAISettings

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode

__all__ = [
    "AgentEvalConverter",
    "AzureAIAgentClient",
    "AzureAIAgentOptions",
    "AzureAIAgentsProvider",
    "AzureAIClient",
    "AzureAIProjectAgentOptions",
    "AzureAIProjectAgentProvider",
    "AzureAISettings",
    "EvalResults",
    "Evaluators",
    "FoundryMemoryProvider",
    "RawAzureAIClient",
    "__version__",
    "evaluate_agent",
    "evaluate_dataset",
    "evaluate_foundry_target",
    "evaluate_response",
    "evaluate_traces",
    "evaluate_workflow",
    "setup_continuous_eval",
]
