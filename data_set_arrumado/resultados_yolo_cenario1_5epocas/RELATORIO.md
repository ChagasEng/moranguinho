# Relatório — YOLO, Cenário 1 (5 épocas)

## Protocolo

- Treino: `Nusrat Sultana/Cenario1` (80% treino interno e 20% validação interna estratificada).
- Modelo: YOLO11n para classificação, sem DBSCAN, 5 épocas e seed 42.
- Teste final: `Nusrat Sultana/Teste`, usado somente após o treino (40 saudáveis e 40 danificados).

## Evolução por época — validação interna

| Época | Acurácia | Loss de treino | Loss de validação |
|---:|---:|---:|---:|
| 1 | 1.0000 | 0.4984 | 0.0728 |
| 2 | 1.0000 | 0.0994 | 0.0109 |
| 3 | 1.0000 | 0.0359 | 0.0067 |
| 4 | 1.0000 | 0.0481 | 0.0009 |
| 5 | 1.0000 | 0.0128 | 0.0010 |

## Métricas no teste final

| Acurácia | F1 macro | Recall macro |
|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 |

## Arquivos gerados

- `metricas.json` e `metricas.csv`: valores brutos das métricas;
- `matriz_confusao.png`: matriz de confusão do teste;
- `treinos/yolo_cenario1/weights/best.pt`: melhor modelo selecionado pela validação interna.
