from fastapi import status

class LLMCrossCheckException(Exception):
    def __init__(self, message: str, error_code: str = "llm_error", status_code: int = status.HTTP_400_BAD_REQUEST, details: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}