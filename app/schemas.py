"""Pydantic request/response models for the MIC prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sequence: str = Field(..., min_length=1, max_length=500)
    microbe_key: str


class PredictResponse(BaseModel):
    microbe_key: str
    microbe_display_name: str
    gram_status: str
    log10_mic: float
    mic_ug_per_ml: float
    warnings: list[str] = []


class Microbe(BaseModel):
    key: str
    display_name: str
    gram_status: str
