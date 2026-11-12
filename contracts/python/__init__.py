"""Shared wire contract between the web-app, Research and character_swap repos.

Vendored byte-identical into all of them; see ../README.md before editing.
"""

from .capabilities import (
    ModelSpec,
    ParameterError,
    ParameterSpec,
    ProviderCapabilities,
    merge_specs,
    validate_override,
)
from .client import BrokerClient, ConsumedMessage
from .messages import (
    CompletedEvent,
    ErrorInfo,
    FailedEvent,
    JobRequest,
    Output,
    ProgressEvent,
    ProviderInfo,
    parse_event,
)
from .streams import CONTRACT_VERSION, DEFAULT_TASK, Group, Stream, StreamNames, Task, job_stream

__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_TASK",
    "BrokerClient",
    "CompletedEvent",
    "ConsumedMessage",
    "ErrorInfo",
    "FailedEvent",
    "Group",
    "JobRequest",
    "ModelSpec",
    "Output",
    "ParameterError",
    "ParameterSpec",
    "ProgressEvent",
    "ProviderCapabilities",
    "ProviderInfo",
    "Stream",
    "StreamNames",
    "Task",
    "job_stream",
    "merge_specs",
    "parse_event",
    "validate_override",
]
