"""Structured error types for consistent API error responses."""


class EngramError(Exception):
    """Raise from any route or dependency to produce a standard error response."""

    def __init__(
        self,
        code: str,
        message: str,
        status: int,
        *,
        existing_id: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status = status
        self.existing_id = existing_id
        self.retry_after = retry_after
        super().__init__(message)
