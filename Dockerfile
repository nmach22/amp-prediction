FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Only the modules actually imported by src/serving/inference.py's import chain
# (verified: no xgboost/catboost/wandb/gensim/ete3 needed at serving time).
COPY src/__init__.py src/__init__.py
COPY src/serving src/serving
COPY src/features/__init__.py src/features/__init__.py
COPY src/features/plm.py src/features/plm.py
COPY src/features/sequence_descriptors.py src/features/sequence_descriptors.py
COPY src/models/__init__.py src/models/__init__.py
COPY src/models/base.py src/models/base.py
COPY src/models/mic_baseline.py src/models/mic_baseline.py
COPY src/models/catboost_mic.py src/models/catboost_mic.py
COPY src/models/taxonomy_mic_baseline.py src/models/taxonomy_mic_baseline.py
COPY src/models/mlp_mic.py src/models/mlp_mic.py

COPY app app
COPY results/models/mlp_mic_physchem_esm2_pca_context_regularized_model.joblib \
     results/models/mlp_mic_physchem_esm2_pca_context_regularized_model.joblib

ENV HF_HOME=/app/.cache/huggingface
# Bake the ESM2 weights into the image at build time so the first production
# request doesn't pay a Hugging Face Hub download.
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D'); \
AutoModel.from_pretrained('facebook/esm2_t12_35M_UR50D')"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
