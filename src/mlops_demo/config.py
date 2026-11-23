"""Configuração central do projeto: caminhos, nomes de colunas e limiares de qualidade."""

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]

# dados
DATASET_ORIGINAL = RAIZ / "data" / "fonte" / "htru2.csv"  # HTRU2 completo (UCI, CC BY 4.0)
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_LOTES = DADOS_BRUTOS / "lotes"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
BASE_TREINO = DADOS_BRUTOS / "base_treino.csv"
BASE_HOLDOUT = DADOS_BRUTOS / "base_holdout.csv"
TREINO_PROCESSADO = DADOS_PROCESSADOS / "treino.csv"
AVALIACAO_PROCESSADA = DADOS_PROCESSADOS / "avaliacao.csv"

# modelos e relatórios
MODELO_CAMPEAO = RAIZ / "models" / "champion" / "model.joblib"
METRICAS_CAMPEAO = RAIZ / "models" / "champion" / "metrics.json"
MODELO_CANDIDATO = RAIZ / "models" / "candidate" / "model.joblib"
METRICAS_CANDIDATO = RAIZ / "reports" / "candidate_metrics.json"
RELATORIO_GATE = RAIZ / "reports" / "gate.json"
RESUMO_GATE = RAIZ / "reports" / "gate.md"
RELATORIO_DRIFT = RAIZ / "reports" / "drift.md"
PARAMS = RAIZ / "params.yaml"

# esquema dos dados (HTRU2: candidatos a pulsar do levantamento High Time Resolution Universe)
# "perfil" = perfil de pulso integrado; "dmsnr" = curva DM-SNR (medida de dispersão x sinal/ruído)
FEATURES = [
    "perfil_media",
    "perfil_desvio",
    "perfil_curtose",
    "perfil_assimetria",
    "dmsnr_media",
    "dmsnr_desvio",
    "dmsnr_curtose",
    "dmsnr_assimetria",
]
ALVO = "pulsar"
# 1 = pulsar real (classe positiva, rara: ~9% da base), 0 = ruído ou interferência
CLASSE_POSITIVA = 1
# colunas que dependem da escala do sinal (ganho do receptor). Usadas para simular drift:
# se o receptor for recalibrado, média e desvio mudam, mas curtose e assimetria não.
COLUNAS_ESCALA = ["perfil_media", "perfil_desvio", "dmsnr_media", "dmsnr_desvio"]
# coluna do teste direcional: pulso estreito e pontudo (curtose alta) é a marca de um pulsar
COLUNA_DIRECIONAL = "perfil_curtose"

# limiares de qualidade (o "contrato" do modelo). Mudar aqui exige revisão em PR.
# Perder um pulsar (falso negativo) é perder uma descoberta: o candidato é descartado e não volta.
# Um falso positivo custa só alguns minutos de um astrônomo olhando o gráfico. Por isso o recall
# é a métrica principal, e a precisão tem um piso para não afogar a equipe em candidatos falsos.
RECALL_MINIMO = 0.85
PRECISAO_MINIMA = 0.80
TOLERANCIA_CAMPEAO = 0.0  # o candidato não pode ter recall menor que o campeão
PISO_SANIDADE = 0.75  # o campeão não pode colapsar na base de referência (holdout fixo)
LIMITE_DRIFT_P = 0.01  # p-valor do teste KS
MIN_FEATURES_DRIFT = 2  # gatilho do CT: 2 dos 8 atributos com drift já disparam o retreino


def carregar_params() -> dict:
    """Lê o params.yaml (hiperparâmetros e semente)."""
    with open(PARAMS, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)
