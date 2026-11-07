"""Exception hierarchy for the memory engine."""


class MemoryEngineError(Exception):
    """Base class for engine errors."""


class NotFoundError(MemoryEngineError):
    """A requested record does not exist."""


class AccessDeniedError(MemoryEngineError):
    """The agent is not authorized to perform the operation."""


class InvalidStateError(MemoryEngineError):
    """The operation is not valid for the record's current state."""
