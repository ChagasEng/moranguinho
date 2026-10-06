# Relatório — seleção por validação e teste final

`Teste` não participou de treino ou seleção. Cada cenário usa 80% treino e 20% validação estratificada.

## Validação

| Cenário | Método | Sadios treino | Danificados treino | Removidas DBSCAN | Acurácia | F1 macro | Recall macro |
|---|---|---:|---:|---:|---:|---:|---:|
| Cenario1 | YOLO | 510 | 510 | 0 | 1.0000 | 1.0000 | 1.0000 |
| Cenario1 | YOLO + DBSCAN | 480 | 508 | 32 | 1.0000 | 1.0000 | 1.0000 |
| Cenario2 | YOLO | 510 | 253 | 0 | 1.0000 | 1.0000 | 1.0000 |
| Cenario2 | YOLO + DBSCAN | 480 | 250 | 33 | 1.0000 | 1.0000 | 1.0000 |
| Cenario3 | YOLO | 253 | 510 | 0 | 1.0000 | 1.0000 | 1.0000 |
| Cenario3 | YOLO + DBSCAN | 237 | 508 | 18 | 1.0000 | 1.0000 | 1.0000 |
| Cenario4 | YOLO | 160 | 80 | 0 | 1.0000 | 1.0000 | 1.0000 |
| Cenario4 | YOLO + DBSCAN | 151 | 77 | 12 | 1.0000 | 1.0000 | 1.0000 |
| Cenario5 | YOLO | 80 | 160 | 0 | 1.0000 | 1.0000 | 1.0000 |
| Cenario5 | YOLO + DBSCAN | 75 | 156 | 9 | 1.0000 | 1.0000 | 1.0000 |

## Teste final (uso único)

Melhor configuração por F1 macro de validação: **Cenario1 — YOLO**.

| Acurácia | F1 macro | Recall macro |
|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 |
