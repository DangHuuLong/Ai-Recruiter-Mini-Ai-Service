from typing import Generic, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class ErrorItem(BaseModel):
    field: str | None = None
    message: str


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    message: str
    data: T | None = None


class ApiErrorResponse(BaseModel):
    success: bool = False
    message: str
    errors: list[ErrorItem] = Field(default_factory=list)