# Resultados obtidos até 04/10/2026

Este resumo reúne as duas execuções registradas do experimento modular CNN + SVM em [`experimento_lmfcn`](../experimento_lmfcn/README.md). Os valores foram extraídos de `experimento_lmfcn/saida/smoke/metrics.json` e `experimento_lmfcn/saida/model/metrics.json`. Cópias completas desses arquivos estão em [`metricas/smoke.json`](metricas/smoke.json) e [`metricas/model.json`](metricas/model.json). Os pesos da CNN e o SVM permanecem em `experimento_lmfcn/saida/`, que é ignorada pelo Git.

## Dados e método

- Imagens usadas: 400 imagens originais de `Dataset/Nusrat Sultana/Original`, sendo 200 frescas (`0`) e 200 podres (`1`). As imagens da pasta `Augmented` não entram no experimento.
- Divisão local: 280 imagens para treino (140 por classe), 60 para validação (30 por classe) e 60 para teste (30 por classe). Os manifestos estão em `experimento_lmfcn/saida/manifests/` e são ignorados pelo Git porque contêm caminhos absolutos locais.
- Modelo: CNN que extrai características, seguida de SVM com kernel RBF. Entre épocas, exemplos escolhidos com auxílio do SVM orientam a atualização da CNN. Esta implementação adapta a ideia do LMFCN; não reproduz numericamente o código de Jonathan.
- Em cada execução, a época escolhida maximiza a acurácia balanceada na validação. O conjunto de teste é avaliado uma vez, com a época escolhida. A semente registrada é `42` e o dispositivo foi `mps` nas duas execuções.

## Acurácia

| Execução | Validação | Teste |
| --- | ---: | ---: |
| `smoke` | 80,00% (48/60) | 76,67% (46/60) |
| `model` | 96,67% (58/60) | 86,67% (52/60) |

Os valores são da época selecionada em cada execução. Como há o mesmo número de imagens por classe, a acurácia comum e a balanceada coincidem nesses conjuntos.

## Comparação das execuções

| Execução | Imagem | Características | Épocas | Melhor época | F1 macro no teste |
| --- | ---: | ---: | ---: | ---: | ---: |
| `smoke` (verificação rápida) | 64 × 64 | 8 | 2 | 2 | 76,56% |
| `model` (treino principal) | 128 × 128 | 16 | 5 | 1 | 86,53% |

As duas configurações também usam lote `32`, taxa de aprendizado `0,001`, `C=1,0` no SVM, margem `1,0` e peso de oposição `0,5`.

### Evolução por época

| Execução | Época | Treino (acurácia balanceada) | Validação (acurácia balanceada) | Vetores de suporte |
| --- | ---: | ---: | ---: | ---: |
| `smoke` | 1 | 82,50% | 76,67% | 223 |
| `smoke` | **2** | 82,50% | **80,00%** | 190 |
| `model` | **1** | 91,07% | **96,67%** | 156 |
| `model` | 2 | 88,21% | 93,33% | 170 |
| `model` | 3 | 86,07% | 91,67% | 176 |
| `model` | 4 | 87,50% | 93,33% | 178 |
| `model` | 5 | 90,00% | **96,67%** | 156 |

Na execução principal, as épocas 1 e 5 empataram na validação. O código mantém a primeira em caso de empate, por isso a época 1 foi usada no teste. Os números de treino são medidos sobre os dados usados para ajustar o SVM.

## Desempenho no teste

| Execução | Classe real | Precisão | Revocação | F1 | Acertos na classe |
| --- | --- | ---: | ---: | ---: | ---: |
| `smoke` | Fresco (`0`) | 73,53% | 83,33% | 78,13% | 25/30 |
| `smoke` | Podre (`1`) | 80,77% | 70,00% | 75,00% | 21/30 |
| `model` | Fresco (`0`) | 80,56% | 96,67% | 87,88% | 29/30 |
| `model` | Podre (`1`) | 95,83% | 76,67% | 85,19% | 23/30 |

Na execução principal, 7 dos 30 morangos podres foram classificados como frescos; 1 dos 30 frescos foi classificado como podre. A matriz de confusão reconstruída a partir do suporte e da revocação registrados é:

| Classe real \ Predição | Fresco | Podre |
| --- | ---: | ---: |
| Fresco | 29 | 1 |
| Podre | 7 | 23 |

## Leitura dos resultados e limites

O treino principal ficou 10 pontos percentuais acima do `smoke` no teste (86,67% contra 76,67%). As execuções usam tamanhos de imagem, números de características e durações diferentes; esses resultados, sozinhos, não isolam o efeito de cada escolha. Há uma semente e uma divisão de teste registradas, com apenas 60 imagens no teste. Ainda não há uma avaliação registrada do código original em `codigo_jonathan/` sobre esse mesmo conjunto, nem repetições com outras divisões ou sementes. Portanto, estes números descrevem as execuções atuais e não uma comparação conclusiva entre métodos.
