# Resultados comparativos — datasetmorangonovo

## Protocolo único da comparação

Todas as linhas deste arquivo usam exclusivamente o `setting2`. Em cada arquitetura, a linha sem DBSCAN e a linha com DBSCAN usam a mesma divisão estratificada 80/20 (semente 42), o mesmo `setting2` e as mesmas 200 imagens externas de `test` (100 `fresh`, 100 `rotten`). A única diferença dentro de cada par é a remoção de ruídos pelo DBSCAN no conjunto de treino; validação e `test` nunca são filtrados.

Os valores abaixo são comparações controladas de DBSCAN, e não comparações entre settings diferentes. O `test` não participou do treino nem da divisão interna.

O tempo de inferência será preenchido diretamente no Raspberry Pi, pois uma medida feita na CPU de desenvolvimento não representa a latência do dispositivo-alvo. O protocolo previsto é: 20 imagens de aquecimento, seguida de inferência individual (`batch=1`) das 200 imagens de `test`; reportar a média em milissegundos por imagem, incluindo pré-processamento e predição, sem leitura de disco.

## YOLO11-cls

### Reprodutibilidade

| Item | Sem DBSCAN | Com DBSCAN |
|---|---|---|
| Dados de origem | `setting2`: 600 `fresh`, 300 `rotten` | mesmo `setting2` |
| Divisão | 480/120 `fresh`; 240/60 `rotten`; estratificada; semente 42 | mesma divisão |
| Modelo inicial | `yolo11n-cls.pt` | `yolo11n-cls.pt` |
| Treino | 3 épocas; batch 32; 224 px; CPU; `workers=0`; determinístico | mesmos parâmetros |
| DBSCAN | não aplicado | embeddings pré-treinados, `StandardScaler`, PCA 32 e DBSCAN por classe; 28 `fresh` + 12 `rotten` removidas |
| Checkpoint | `treinos/setting2/weights/best.pt` | `dbscan/setting2/treino/weights/best.pt` |
| SHA-256 DBSCAN | — | `4f50fbbcd1a375e065014e1de2d761237a057c42ace78928852211a1d7f5e90a` |

#### Parâmetros efetivos do treinamento YOLO

| Parâmetro | Valor |
|---|---|
| Versão do modelo | YOLO11 nano para classificação (`yolo11n-cls.pt`) |
| Épocas / batch / imagem | `3` / `32` / `224 × 224` px |
| Dispositivo / carregamento | CPU / `workers=0` |
| Semente / determinismo | `42` / `true` |
| Otimizador | `auto` (seleção do Ultralytics) |
| Taxa de aprendizado | `lr0=0.01`; `lrf=0.01` |
| Momentum / weight decay | `0.937` / `0.0005` |
| Warmup | `3.0` épocas; `warmup_momentum=0.8`; `warmup_bias_lr=0.1` |
| Precisão mista | `amp=true` |
| Pré-treinamento | `pretrained=true` |
| Aumentos | `auto_augment=randaugment`; `erasing=0.4`; `fliplr=0.5`; `translate=0.1`; `scale=0.5`; `mosaic=1.0` |
| Aumentos desativados | `degrees=0`; `shear=0`; `perspective=0`; `flipud=0`; `mixup=0`; `cutmix=0` |
| Avaliação | `split=val`; `iou=0.7`; `max_det=300` |

Os mesmos hiperparâmetros foram usados no YOLO com DBSCAN; somente as imagens de treino mudaram após a filtragem. O arquivo completo gerado pelo Ultralytics está em `resultados_yolo_settings/treinos/setting2/args.yaml`.

### Métricas finais no conjunto de teste

| Método / setting | Acurácia | F1 macro | Recall macro | Tempo de inferência no Rasp (ms/imagem) |
|---|---:|---:|---:|---:|
| YOLO11n-cls / `setting2` | 0.97 | 0.969973 | 0.97 | Pendente de medição no Rasp |
| YOLO11n-cls + DBSCAN / `setting2` | 0.99 | 0.989999 | 0.99 | Pendente de medição no Rasp |

Artefatos: `C:\Users\matheus\Documents\datasetmorangonovo\resultados_yolo_settings`. Para reproduzir somente a avaliação DBSCAN controlada:

```powershell
.\.venv-yolo\Scripts\python.exe ".\testes YOLO\comparar_yolo_dbscan_settings.py" --avaliar-setting setting2 --batch 32
```

## CNN — MobileNetV2

### Reprodutibilidade

| Item | Sem DBSCAN | Com DBSCAN |
|---|---|---|
| Dados e divisão | `setting2`; 480/120 `fresh`; 240/60 `rotten`; semente 42 | mesmo `setting2` e divisão |
| Modelo | MobileNetV2 com pesos ImageNet e classificador de duas classes | mesmo modelo |
| Treino | 3 épocas; AdamW (`lr=0,0001`, `weight_decay=0,0001`); batch 16; CPU; 4 threads | mesmos parâmetros |
| Transformações | treino: `Resize(256)`, `RandomResizedCrop(224)`, espelhamento; avaliação: `Resize(256)`, `CenterCrop(224)`, normalização ImageNet | mesmas transformações |
| DBSCAN | não aplicado | embeddings ImageNet, `StandardScaler`, PCA 32 e DBSCAN por classe; 36 `fresh` + 12 `rotten` removidas |
| Checkpoint | `treinos/setting2/best.pt` | `dbscan/setting2/treino/best.pt` |
| SHA-256 DBSCAN | — | `b422df37be1a4adfa099d313f3cc5a7e6f89f9470e4012ab9ba7c05d70a0f3f1` |

### Métricas finais no conjunto de teste

| Método / setting | Acurácia | F1 macro | Recall macro | Tempo de inferência no Rasp (ms/imagem) |
|---|---:|---:|---:|---:|
| MobileNetV2 / `setting2` | 0.98 | 0.979998 | 0.98 | Pendente de medição no Rasp |
| MobileNetV2 + DBSCAN / `setting2` | 0.98 | 0.979992 | 0.98 | Pendente de medição no Rasp |

Artefatos: `C:\Users\matheus\Documents\datasetmorangonovo\resultados_mobilenetv2_settings`. Para reproduzir somente a avaliação DBSCAN controlada:

```powershell
.\.venv-yolo\Scripts\python.exe ".\testes MobileNetV2\comparar_mobilenetv2_dbscan_settings.py" --avaliar-setting setting2 --batch 16 --threads 4
```

## LMFCN + SVM

### Reprodutibilidade

Este bloco usa uma CNN extratora de características seguida de SVM RBF pré-calculada, com atualização guiada por vetores de suporte. É uma adaptação modular reproduzível da técnica LMFCN; não é uma reprodução numérica literal do repositório original.

| Item | Sem DBSCAN | Com DBSCAN |
|---|---|---|
| Dados e divisão | `setting2`; 480/120 `fresh`; 240/60 `rotten`; semente 42 | mesmo `setting2` e divisão |
| Treino | 3 épocas; 128 px; 16 características; SVM RBF (`C=1,0`); batch 32; CPU | mesmos parâmetros |
| DBSCAN | não aplicado | aplicado somente ao treino sobre embeddings do extrator inicial; 32 `fresh` + 18 `rotten` removidas |
| Checkpoint | `sem_dbscan/setting2/treino/cnn.pt` | `dbscan/setting2/treino/cnn.pt` |

### Métricas finais no conjunto de teste

| Método / setting | Acurácia | F1 macro | Recall macro | Tempo de inferência no Rasp (ms/imagem) |
|---|---:|---:|---:|---:|
| LMFCN + SVM / `setting2` | 0.56 | 0.460189 | 0.56 | Pendente de medição no Rasp |
| LMFCN + SVM + DBSCAN / `setting2` | 0.565 | 0.469108 | 0.565 | Pendente de medição no Rasp |

Artefatos: `C:\Users\matheus\Documents\datasetmorangonovo\resultados_lmfcn_settings`.
