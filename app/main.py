"""FastAPI app serving MIC predictions for a peptide against a fixed microbe list."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.schemas import Microbe, PredictRequest, PredictResponse  # noqa: E402
from src.features.plm import PLMEncoder  # noqa: E402
from src.serving.inference import (  # noqa: E402
    load_mic_model,
    load_microbes,
    predict_mic,
    validate_sequence,
)

DEFAULT_MODEL_PATH = (
    ROOT / "results" / "models"
    / "mlp_mic_physchem_esm2_pca_context_regularized_model.joblib"
)
MODEL_PATH = Path(os.environ.get("MIC_MODEL_PATH", DEFAULT_MODEL_PATH))
MICROBES_PATH = Path(os.environ.get("MIC_MICROBES_PATH", ROOT / "app" / "microbes.json"))

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["loaded_model"] = load_mic_model(MODEL_PATH)
    STATE["microbes"] = load_microbes(MICROBES_PATH)
    plm_encoder = PLMEncoder(cache_dir=None)
    plm_encoder.encode(["ACDEFGHIKL"])  # force the HF model to load at startup
    STATE["plm_encoder"] = plm_encoder
    yield
    STATE.clear()


app = FastAPI(title="AMP MIC Predictor", lifespan=lifespan)


@app.get("/microbes", response_model=list[Microbe])
def list_microbes() -> list[Microbe]:
    return [
        Microbe(
            key=microbe.key,
            display_name=microbe.display_name,
            gram_status=microbe.gram_status,
        )
        for microbe in STATE["microbes"].values()
    ]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    microbe = STATE["microbes"].get(request.microbe_key)
    if microbe is None:
        raise HTTPException(400, f"Unknown microbe_key: {request.microbe_key!r}")

    try:
        warnings = validate_sequence(request.sequence)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    result = predict_mic(
        STATE["loaded_model"], STATE["plm_encoder"], request.sequence, microbe
    )
    return PredictResponse(
        microbe_key=microbe.key,
        microbe_display_name=microbe.display_name,
        gram_status=microbe.gram_status,
        warnings=warnings,
        **result,
    )


app.mount("/", StaticFiles(directory=str(ROOT / "app" / "static"), html=True), name="static")
