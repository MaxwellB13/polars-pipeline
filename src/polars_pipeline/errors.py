"""Exception hierarchy. Everything the library raises derives from PipelineError."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error the library raises on purpose."""


class DefinitionError(PipelineError):
    """A pipeline class is mis-declared (unknown input, duplicate output, cycle...)."""


class MissingInputError(PipelineError):
    """A selected step needs a frame that neither a selected step nor staging provides."""


class StepError(PipelineError):
    """A step raised. The original exception is chained as __cause__."""

    def __init__(self, step: str, cause: BaseException) -> None:
        super().__init__(f"step {step!r} failed: {type(cause).__name__}: {cause}")
        self.step = step
        self.__cause__ = cause


class SourceError(PipelineError):
    """A reader or a source's prepare/conform stage failed."""


class StagingError(PipelineError):
    """The staging area is missing something a run needs, or is corrupt."""


class HermeticError(PipelineError):
    """A hermetic run cannot proceed (a source has no spec, FK cycle...)."""
