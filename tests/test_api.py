"""Testes de contrato da API: formato de entrada e de saída do /predict."""

import pytest
from fastapi.testclient import TestClient

from mlops_demo import config
from mlops_demo.api import app

pytestmark = pytest.mark.api

cliente = TestClient(app)


def test_health_responde_ok():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def _candidato(linha) -> dict:
    return {nome: float(linha[nome]) for nome in config.FEATURES}


def test_predict_devolve_o_formato_combinado(base_holdout):
    resposta = cliente.post("/predict", json=_candidato(base_holdout.iloc[0]))
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"classe", "probabilidade_pulsar", "versao_codigo"}
    assert corpo["classe"] in {"pulsar", "ruido"}
    assert 0.0 <= corpo["probabilidade_pulsar"] <= 1.0


def test_predict_rejeita_atributo_faltando(base_holdout):
    candidato = _candidato(base_holdout.iloc[0])
    candidato.pop("dmsnr_curtose")
    resposta = cliente.post("/predict", json=candidato)
    assert resposta.status_code == 422


def test_predict_rejeita_desvio_zero_ou_media_fora_da_faixa(base_holdout):
    """Novo: a API aplica as mesmas regras físicas da validação de dados."""
    base = _candidato(base_holdout.iloc[0])
    for campo, valor in (("perfil_desvio", 0.0), ("dmsnr_desvio", -2.0), ("perfil_media", 300.0)):
        resposta = cliente.post("/predict", json={**base, campo: valor})
        assert resposta.status_code == 422, campo


def test_predict_acerta_um_pulsar_e_um_ruido_claros(base_holdout, campeao):
    """Escolhe os casos mais claros (maior confiança do campeão) de cada classe."""
    X = base_holdout[config.FEATURES]
    coluna = list(campeao.classes_).index(config.CLASSE_POSITIVA)
    p_pulsar = campeao.predict_proba(X)[:, coluna]
    i_pulsar = int(p_pulsar.argmax())
    i_ruido = int(p_pulsar.argmin())
    for indice, esperado in ((i_pulsar, "pulsar"), (i_ruido, "ruido")):
        resposta = cliente.post("/predict", json=_candidato(X.iloc[indice]))
        assert resposta.json()["classe"] == esperado


def test_health_mostra_as_metricas_do_campeao_dentro_do_contrato():
    """Novo: o /health expõe recall e precisão do campeão, e eles cumprem o contrato."""
    metricas = cliente.get("/health").json()["metricas_do_campeao"]
    assert metricas["recall_pulsar"] >= config.RECALL_MINIMO
    assert metricas["precisao_pulsar"] >= config.PRECISAO_MINIMA
