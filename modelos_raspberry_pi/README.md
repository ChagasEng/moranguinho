# Modelos para Raspberry Pi

Este pacote reúne o melhor modelo de cada arquitetura na comparação controlada do `setting2`:

| Arquitetura | Modelo incluído | Critério |
|---|---|---|
| YOLO11-cls | YOLO11n-cls + DBSCAN | maior F1 macro: `0.989999` |
| MobileNetV2 | MobileNetV2 sem DBSCAN | maior F1 macro: `0.979998` |
| LMFCN + SVM | LMFCN + SVM + DBSCAN | maior F1 macro: `0.469108` |

Os arquivos em `pesos/` são cópias dos checkpoints avaliados no conjunto externo. As métricas e os SHA-256 estão no `MANIFESTO_MODELOS.json`; consulte também `RESULTADOS_COMPARATIVOS.md`.

## Instalação no Raspberry Pi

Use Python 3.10 ou superior. Instale primeiro uma versão de PyTorch e Torchvision compatível com a arquitetura e o sistema operacional do Raspberry Pi; depois instale as demais dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

## Inferência de uma imagem

```bash
python inferir.py --modelo yolo --imagem /caminho/imagem.jpg
python inferir.py --modelo mobilenet --imagem /caminho/imagem.jpg
python inferir.py --modelo lmfcn --imagem /caminho/imagem.jpg
```

Classes: `fresh` = 0 e `rotten` = 1. O programa imprime a classe e o tempo de uma execução simples; esse tempo inclui o carregamento do modelo e não deve ser usado no artigo. Para o valor reportável, use o protocolo do relatório: carregue o modelo uma vez, faça 20 imagens de aquecimento e calcule a média de 200 imagens com `batch=1`, incluindo pré-processamento e predição, sem leitura de disco.

## Benchmark dos três modelos

Copie também a pasta `test` para o Raspberry Pi. Na raiz deste pacote, execute:

```bash
python benchmark_raspberry_pi.py --imagens /caminho/test --aquecimento 20 --threads 4 --saida benchmark_raspberry_pi.json
```

O arquivo JSON resultante traz média, mediana e percentil 95 em ms/imagem para YOLO11 + DBSCAN, MobileNetV2 e LMFCN + SVM + DBSCAN. A medição mantém os pesos carregados e as imagens em RAM, portanto não mede inicialização nem leitura de disco.
