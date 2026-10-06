# Relatório — MobileNetV2, Cenário 1 (5 épocas)

Treino: 80% do `Cenario1`; validação interna estratificada: 20%. MobileNetV2 pré-treinada, sem DBSCAN, seed 42. O `Teste` foi usado somente após o treino (40 saudáveis e 40 danificados).

## Evolução por época — validação interna

| Época | Acurácia | F1 macro | Recall macro | Loss de treino |
|---:|---:|---:|---:|---:|
| 1 | 0.8750 | 0.8749 | 0.8750 | 0.6271 |
| 2 | 0.9844 | 0.9844 | 0.9844 | 0.4631 |
| 3 | 0.9844 | 0.9844 | 0.9844 | 0.3262 |
| 4 | 0.9844 | 0.9844 | 0.9844 | 0.2424 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.1599 |

## Métricas no teste final

| Acurácia | F1 macro | Recall macro |
|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 |

Arquivos: `metricas.json`, `metricas.csv`, `matriz_confusao.png` e `melhor.pt`.
