"""Cenários de prática: cada um aplica uma mudança para você ver o pipeline reagir.

Uso (a partir da raiz do projeto, sempre em uma branch nova):
    python demo/cenarios.py listar
    python demo/cenarios.py aplicar 1

Depois de aplicar: git add -A, git commit, git push e abra o Pull Request.
"""

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from mlops_demo.data import gerar_lote  # noqa: E402

DESCRICAO = {
    1: "Bug no pré-processamento: some a coluna dmsnr_assimetria. Esperado: CI falha em código.",
    2: "Base de treino corrompida (nulos e desvio zero). Esperado: CI falha na validação de dados.",
    3: "Regularização forte demais (C: 0.00005). Esperado: CI falha em modelo (recall ~0,31).",
    4: "Receptor recalibrado (ganho x0,75). Esperado: CI verde, CT retreina e aprova.",
    5: "Peso 50 na classe pulsar. Esperado: recall ~0,98 e precisão ~0,35, CI e portão reprovam.",
}


def _substituir(arquivo: Path, antigo: str, novo: str) -> None:
    texto = arquivo.read_text(encoding="utf-8")
    if antigo not in texto:
        raise SystemExit(f"Trecho não encontrado em {arquivo.name}. O cenário já foi aplicado?")
    arquivo.write_text(texto.replace(antigo, novo, 1), encoding="utf-8")
    print(f"alterado: {arquivo.relative_to(RAIZ)}")


def cenario_1() -> None:
    _substituir(
        RAIZ / "src" / "mlops_demo" / "features.py",
        "X = df[FEATURES].copy()",
        "X = df[FEATURES[:-1]].copy()",
    )


def cenario_2() -> None:
    caminho = RAIZ / "data" / "raw" / "base_treino.csv"
    df = pd.read_csv(caminho)
    df.loc[0:4, "perfil_media"] = None
    df.loc[5:7, "perfil_desvio"] = 0.0
    df.to_csv(caminho, index=False)
    print("alterado: data/raw/base_treino.csv (5 nulos em perfil_media, 3 desvios zerados)")


def cenario_3() -> None:
    _substituir(RAIZ / "params.yaml", "C: 1.0", "C: 0.00005")


def cenario_4() -> None:
    caminho = gerar_lote("lote_01", n=600, fator_drift=0.75, semente=1)
    print("criado:", Path(caminho).relative_to(RAIZ), "(média e desvio do sinal 25% menores)")


def cenario_5() -> None:
    _substituir(RAIZ / "params.yaml", "    1: 5", "    1: 50")


CENARIOS = {1: cenario_1, 2: cenario_2, 3: cenario_3, 4: cenario_4, 5: cenario_5}

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "listar":
        for numero, texto in DESCRICAO.items():
            print(f"{numero}. {texto}")
    elif len(sys.argv) == 3 and sys.argv[1] == "aplicar" and sys.argv[2].isdigit():
        numero = int(sys.argv[2])
        if numero not in CENARIOS:
            raise SystemExit("Cenário inexistente. Use: python demo/cenarios.py listar")
        print(f"Cenário {numero}: {DESCRICAO[numero]}")
        CENARIOS[numero]()
    else:
        print(__doc__)
        sys.exit(2)
