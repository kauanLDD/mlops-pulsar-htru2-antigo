"""API de inferência (FastAPI) servindo o modelo campeão de triagem de pulsares.

Local:   uvicorn mlops_demo.api:app --reload
Docker:  docker run -p 8000:8000 mlops-demo
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mlops_demo import __version__, config

CAMINHO_MODELO = Path(os.environ.get("MODEL_PATH", config.MODELO_CAMPEAO))
CAMINHO_METRICAS = CAMINHO_MODELO.with_name("metrics.json")

app = FastAPI(title="Triagem de candidatos a pulsar (demo CI/CD/CT)", version=__version__)
_modelo = joblib.load(CAMINHO_MODELO)


class Candidato(BaseModel):
    """Os 8 atributos de um candidato do HTRU2. Desvios precisam ser positivos."""

    perfil_media: float = Field(ge=0, le=255)
    perfil_desvio: float = Field(gt=0)
    perfil_curtose: float
    perfil_assimetria: float
    dmsnr_media: float = Field(ge=0)
    dmsnr_desvio: float = Field(gt=0)
    dmsnr_curtose: float
    dmsnr_assimetria: float


class Predicao(BaseModel):
    classe: str
    probabilidade_pulsar: float
    versao_codigo: str


@app.get("/health")
def saude() -> dict:
    metricas = {}
    if CAMINHO_METRICAS.exists():
        metricas = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))
    return {"status": "ok", "versao_codigo": __version__, "metricas_do_campeao": metricas}


@app.post("/predict", response_model=Predicao)
def prever(candidato: Candidato) -> Predicao:
    entrada = pd.DataFrame([candidato.model_dump()])[config.FEATURES]
    previsto = int(_modelo.predict(entrada)[0])
    # a coluna da classe pulsar (1) em predict_proba segue a ordem de classes_
    coluna = list(_modelo.classes_).index(config.CLASSE_POSITIVA)
    p_pulsar = float(_modelo.predict_proba(entrada)[0][coluna])
    return Predicao(
        classe="pulsar" if previsto == config.CLASSE_POSITIVA else "ruido",
        probabilidade_pulsar=round(p_pulsar, 4),
        versao_codigo=__version__,
    )
