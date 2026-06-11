"""Exception hierarchy for the remediation platform."""


class RemediationError(Exception):
    """Base class for all platform errors."""


class ToolError(RemediationError):
    """A stateless tool failed (git, docker, REST call, filesystem)."""


class LLMError(RemediationError):
    """The LLM returned unusable output after all repair attempts."""


class ValidationFailedError(RemediationError):
    """A generated artifact failed validation after max reflection rounds."""


class PipelineExecutionError(RemediationError):
    """A remote pipeline (AzDO or AML) failed and could not be repaired."""


class ConfigurationError(RemediationError):
    """Required configuration is missing or invalid."""
