"""Testes de dados: esquema, valores e vazamento entre treino e teste."""

import pandas as pd
import pytest

from mlops_demo import config
from mlops_demo.data import listar_lotes, validar_dataframe

pytestmark = pytest.mark.dados


def _arquivos_de_dados():
    return [config.BASE_TREINO, config.BASE_HOLDOUT, *listar_lotes()]


@pytest.mark.parametrize("caminho", _arquivos_de_dados(), ids=lambda c: c.name)
def test_arquivo_respeita_esquema_e_valores(caminho):
    problemas = validar_dataframe(pd.read_csv(caminho))
    assert problemas == [], f"{caminho.name}: {problemas}"


def test_base_de_treino_mantem_a_proporcao_de_pulsares(base_treino):
    # no HTRU2 completo ~9,2% são pulsares; a divisão estratificada tem que manter isso
    proporcao = (base_treino[config.ALVO] == config.CLASSE_POSITIVA).mean()
    assert 0.07 <= proporcao <= 0.12


def test_holdout_tem_pulsares_suficientes_para_medir_recall(base_holdout):
    """Novo: com poucos pulsares no holdout, cada erro mexe demais no recall."""
    assert (base_holdout[config.ALVO] == config.CLASSE_POSITIVA).sum() >= 80


def test_nao_ha_vazamento_entre_treino_e_holdout(base_treino, base_holdout):
    chaves_treino = set(map(tuple, base_treino[config.FEATURES].round(6).to_numpy()))
    chaves_holdout = set(map(tuple, base_holdout[config.FEATURES].round(6).to_numpy()))
    assert chaves_treino.isdisjoint(chaves_holdout)


def test_nao_ha_linhas_duplicadas_na_base_de_treino(base_treino):
    assert not base_treino.duplicated().any()


def test_validador_detecta_nulos_e_desvio_invalido(base_treino):
    corrompido = base_treino.copy()
    corrompido.loc[0, "perfil_media"] = None
    corrompido.loc[1, "perfil_desvio"] = 0.0
    problemas = validar_dataframe(corrompido)
    assert any("nulos" in p for p in problemas)
    assert any("desvio padrão" in p for p in problemas)


def test_validador_detecta_valores_fora_da_faixa_do_dominio(base_treino):
    """Novo: média do perfil acima de 8 bits, média DM-SNR negativa e curtose impossível."""
    corrompido = base_treino.copy()
    corrompido.loc[0, "perfil_media"] = 300.0
    corrompido.loc[1, "dmsnr_media"] = -1.0
    corrompido.loc[2, "perfil_curtose"] = -10.0
    problemas = validar_dataframe(corrompido)
    assert any("8 bits" in p for p in problemas)
    assert any("média negativa" in p for p in problemas)
    assert any("curtose impossível" in p for p in problemas)


def test_validador_detecta_esquema_diferente(base_treino):
    assert validar_dataframe(base_treino.drop(columns=["perfil_media"]))
