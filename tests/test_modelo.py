"""Testes de modelo: limiares, baseline e comportamento (ML Test Score, Breck et al., 2017)."""

import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from mlops_demo import config
from mlops_demo.evaluate import avaliar, calcular_metricas
from mlops_demo.features import separar_xy
from mlops_demo.train import treinar

pytestmark = pytest.mark.modelo


def test_campeao_versionado_foi_certificado_pelo_portao():
    """O metrics.json do campeão registra o que o portão mediu ao aprová-lo e cumpre o contrato."""
    with open(config.METRICAS_CAMPEAO, encoding="utf-8") as arquivo:
        declaradas = json.load(arquivo)
    assert declaradas["recall_pulsar"] >= config.RECALL_MINIMO, declaradas
    assert declaradas["precisao_pulsar"] >= config.PRECISAO_MINIMA, declaradas


def test_campeao_versionado_nao_colapsou_na_base_de_referencia(campeao, base_holdout):
    """Sanidade: o artefato carrega, prevê e não perdeu o básico no holdout fixo.

    O contrato (recall 0,85 e precisão 0,80) é medido pelo portão, no conjunto de avaliação
    da época da promoção.
    Depois que o modelo se adapta a dados novos, o holdout antigo pode cair um pouco.
    Por isso o piso aqui é menor.
    """
    metricas = avaliar(campeao, base_holdout)
    assert metricas["recall_pulsar"] >= config.PISO_SANIDADE, metricas
    assert metricas["precisao_pulsar"] >= config.PISO_SANIDADE, metricas


def test_treino_com_o_codigo_e_params_atuais_cumpre_o_contrato(modelo_atual, base_holdout):
    """Se alguém mexer no código ou no params.yaml, o CI treina de novo e confere os limiares."""
    metricas = avaliar(modelo_atual, base_holdout)
    assert metricas["recall_pulsar"] >= config.RECALL_MINIMO, metricas
    assert metricas["precisao_pulsar"] >= config.PRECISAO_MINIMA, metricas


def test_modelo_supera_o_baseline_ingenuo(modelo_atual, base_treino, base_holdout):
    X_tr, y_tr = separar_xy(base_treino)
    X_te, y_te = separar_xy(base_holdout)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    # acurácia não serve aqui: o baseline "sempre ruído" já acerta 91%. Comparamos o recall.
    nota_baseline = calcular_metricas(y_te, baseline.predict(X_te))["recall_pulsar"]
    nota_modelo = calcular_metricas(y_te, modelo_atual.predict(X_te))["recall_pulsar"]
    assert nota_modelo > nota_baseline + 0.5


def test_treino_e_deterministico(base_treino, params, xy_holdout):
    X_te, _ = xy_holdout
    a = treinar(base_treino, params).predict_proba(X_te)
    b = treinar(base_treino, params).predict_proba(X_te)
    assert np.allclose(a, b)


def test_modelo_consegue_memorizar_um_lote_pequeno(base_treino, params):
    """Teste clássico de sanidade: se nem um lote pequeno é aprendido, há um bug no pipeline."""
    pequeno = base_treino.sample(40, random_state=0)
    modelo = treinar(pequeno, params)
    X, y = separar_xy(pequeno)
    assert (modelo.predict(X) == y).mean() >= 0.95


def test_probabilidades_sao_validas(modelo_atual, xy_holdout):
    X_te, _ = xy_holdout
    probas = modelo_atual.predict_proba(X_te)
    assert np.all((probas >= 0) & (probas <= 1))
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_invariancia_a_ruido_minimo(modelo_atual, xy_holdout):
    """Teste de comportamento: ruído de 0,01% nos atributos quase nunca muda a previsão."""
    X_te, _ = xy_holdout
    rng = np.random.default_rng(0)
    ruidoso = X_te * (1 + rng.normal(0, 1e-4, X_te.shape))
    mudou = (modelo_atual.predict(X_te) != modelo_atual.predict(ruidoso)).mean()
    assert mudou <= 0.02


def test_direcional_curtose_maior_nao_reduz_a_chance_de_pulsar(modelo_atual, xy_holdout):
    """Teste direcional: pulso mais pontudo (curtose do perfil maior) não pode baixar a
    probabilidade média de pulsar. É a assinatura física de um pulsar."""
    X_te, _ = xy_holdout
    coluna = list(modelo_atual.classes_).index(config.CLASSE_POSITIVA)
    pontudo = X_te.copy()
    pontudo[config.COLUNA_DIRECIONAL] = pontudo[config.COLUNA_DIRECIONAL] + 0.5
    antes = modelo_atual.predict_proba(X_te)[:, coluna].mean()
    depois = modelo_atual.predict_proba(pontudo)[:, coluna].mean()
    assert depois >= antes


def test_modelo_nao_depende_da_ordem_das_linhas(modelo_atual, base_treino, params, xy_holdout):
    """Novo: embaralhar as linhas do treino não pode mudar o modelo (sem dependência de ordem)."""
    X_te, _ = xy_holdout
    embaralhado = base_treino.sample(frac=1.0, random_state=123).reset_index(drop=True)
    outro = treinar(embaralhado, params)
    concordancia = (modelo_atual.predict(X_te) == outro.predict(X_te)).mean()
    assert concordancia >= 0.995


def test_modelo_acha_mais_pulsares_que_o_baseline_aleatorio(modelo_atual, xy_holdout):
    """Novo: o modelo precisa ter recall muito acima de um sorteio com a taxa de 9%."""
    X_te, y_te = xy_holdout
    rng = np.random.default_rng(0)
    sorteio = (rng.random(len(y_te)) < 0.09).astype(int)
    recall_sorteio = calcular_metricas(y_te, sorteio)["recall_pulsar"]
    recall_modelo = calcular_metricas(y_te, modelo_atual.predict(X_te))["recall_pulsar"]
    assert recall_modelo > recall_sorteio + 0.5
