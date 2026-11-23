# Relatório da adaptação

Repositório: https://github.com/kauanLDD/mlops-pulsar-htru2

## 1. Problema e dados

O modelo faz triagem de candidatos a pulsar. Um radiotelescópio gera milhares de candidatos por levantamento e quase
todos são ruído ou interferência de rádio. O modelo diz quais valem a pena o astrônomo olhar.

- Dataset: **HTRU2** (Lyon et al., 2016), UCI Machine Learning Repository, id 372.
- Licença: CC BY 4.0. Não tem dado pessoal, são medidas de sinal de rádio.
- Tamanho: 17.898 linhas, 8 atributos numéricos, 1.639 pulsares (9,2%).
- Atributos: média, desvio, curtose e assimetria do perfil de pulso integrado (`perfil_*`) e da curva DM-SNR (`dmsnr_*`).
- Divisão (semente 7, estratificada): 3.000 para `base_treino.csv`, 1.000 para `base_holdout.csv` e o resto (13.898)
  fica como reserva, de onde saem os lotes novos. Não tem sobreposição entre treino e holdout (tem teste para isso).

## 2. Contrato de qualidade

- Classe positiva: **pulsar** (rótulo 1).
- Erro mais caro: **falso negativo**. Um pulsar descartado na triagem não volta, é uma descoberta perdida. Um falso
  positivo custa poucos minutos de alguém olhando o gráfico do candidato.
- Métrica principal: **recall da classe pulsar >= 0,85**.
- Piso de precisão: **>= 0,80**. Sem esse piso, um modelo que marca tudo como pulsar passa no recall e enche a fila de
  candidatos falsos. Com 0,80, de cada 10 candidatos que a equipe olha, pelo menos 8 são pulsares.
- O candidato não pode ter recall menor que o campeão no mesmo conjunto.
- Acurácia não entra no contrato: chutar "ruído" sempre já dá 91%.

Escolhemos 0,85 e não 0,90 porque a regressão logística no HTRU2 fica perto de 0,89 de recall. Com 0,90 o portão
ia reprovar quase tudo por ruído da amostra (o holdout tem 92 pulsares, cada erro vale ~1 ponto de recall).

## 3. O que mudou

| Etapa | Arquivo | Mudança |
|---|---|---|
| 1 | `data/raw/*.csv`, `data/fonte/htru2.csv` | dados do HTRU2, treino 3.000 e holdout 1.000, `lotes/` vazio |
| 2 | `data.py` | saiu `load_breast_cancer`, agora lê `data/fonte/htru2.csv` com divisão de semente fixa |
| 3 | `config.py` | `FEATURES`, `ALVO = "pulsar"`, `CLASSE_POSITIVA = 1`, `COLUNAS_ESCALA`, `COLUNA_DIRECIONAL`, limiares 0,85 e 0,80, `PISO_SANIDADE = 0,75` |
| 4 | `data.py` (`validar_dataframe`) | tipo numérico, infinitos, desvio <= 0, média negativa, `perfil_media` > 255 (8 bits), curtose < -3,5 |
| 5 | `params.yaml` | regressão logística mantida, `class_weight` 1:5 (peso 3 deixava o recall em 0,86, no limite) |
| 6 | `evaluate.py`, `gate.py` | `pos_label = 1`, métricas `recall_pulsar` e `precisao_pulsar` |
| 7 | `data.py` (`aplicar_drift`), `config.py` | drift = receptor recalibrado: média e desvio x fator, curtose e assimetria iguais. `MIN_FEATURES_DRIFT = 2` |
| 8 | `api.py`, `demo/smoke_test.py` | entrada com os 8 campos nomeados e validados (desvio > 0, média entre 0 e 255), resposta `pulsar`/`ruido` e `probabilidade_pulsar` |
| 9 | `tests/` | todos adaptados. Novos: acurácia engana (código), drift só nas colunas de escala (código), limiar exato (código), faixas do domínio (dados), holdout com 80+ pulsares (dados), ordem das linhas (modelo), recall acima do sorteio (modelo), API rejeita desvio zero e média fora da faixa (API), `/health` dentro do contrato (API). O direcional virou "curtose do perfil maior não baixa a chance de pulsar" |
| 10 | `demo/cenarios.py`, `demo/gerar_lote.py` | cenários com as colunas do HTRU2, lote padrão de 600 linhas |
| 11 | `models/champion/` | campeão inicial: recall 0,891 e precisão 0,882 no holdout |
| 12 | `README.md` | descrição, contrato e tabela de cenários |

## 4. Evidências

| | O que mostra | Execução |
|---|---|---|
| E1 | CI verde na main depois da adaptação (3 jobs) | https://github.com/kauanLDD/mlops-pulsar-htru2/actions/runs/45289333228 |
| E2 | CI vermelho no step **Testes de código** (cenário 1, PR #1) | https://github.com/kauanLDD/mlops-pulsar-htru2/actions/runs/45289602202 |
| E3 | CT com drift (4 de 8 atributos) e portão aprovando, 0,875 x 0,867 (cenário 4, PR #2) | https://github.com/kauanLDD/mlops-pulsar-htru2/actions/runs/45289620554 |
| E4 | CT vermelho no portão: recall 0,975 e precisão 0,325 (cenário 5, PR #4) | https://github.com/kauanLDD/mlops-pulsar-htru2/actions/runs/45290191947 |
| E5 | CD da tag v1.0.0: CI, imagem, teste de fumaça e publicação no ghcr.io | https://github.com/kauanLDD/mlops-pulsar-htru2/actions/runs/45290397401 |

Depois do merge do PR #2, o CT na main aprovou de novo e abriu o PR #3 com o novo campeão, que foi revisado e
aceito. No E4 o campeão não mudou (o PR foi fechado sem merge e o CT não promove em PR).

Imagem: `ghcr.io/kauanldd/mlops-pulsar-htru2:1.0.0`

## 5. Limites e riscos

O portão só olha números agregados em um conjunto de avaliação. Ele não garante um bom modelo porque:

- o holdout tem só 92 pulsares, então um ou dois acertos a mais já mudam o veredito (no E3 a diferença foi 0,008);
- ele não vê viés por tipo de pulsar (ex.: pulsares de milissegundo ou sinais fracos podem estar todos errados e o
  recall geral continuar alto);
- se o drift também chegar na avaliação, o portão compara candidato e campeão num conjunto que já mudou;
- ele não mede calibração da probabilidade, latência nem custo.

Em produção a gente monitoraria: drift por atributo (KS, como já faz), proporção de candidatos marcados como pulsar
(se pular de 9% para 30% tem algo errado), recall real com os rótulos que os astrônomos confirmam depois, distribuição
da probabilidade prevista, erros 422 na API (sinal de mudança no pipeline de sinal) e latência.

## 6. Contribuição e uso de IA

| Integrante | Contribuição |
|---|---|
| Kauan | criação do repositório, adaptação do código e testes, evidências |
| (preencher) | (preencher) |
| (preencher) | (preencher) |

Uso de IA: usamos o Claude Code (Anthropic) para ler o projeto original, sugerir a troca de dataset, adaptar o código
e os testes, rodar o CI local e escrever o rascunho deste relatório. Os limiares, o drift e os resultados foram
conferidos por nós rodando `python demo/local.py ci` e olhando os logs do Actions.
