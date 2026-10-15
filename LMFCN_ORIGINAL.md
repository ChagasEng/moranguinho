# Execucao do LMFCN original

O executor [`executar_lmfcn_original.py`](executar_lmfcn_original.py) utiliza diretamente os arquivos de
`codigo_jonathan/lmfcn`, sem alterar a arquitetura, o otimizador, as perdas ou o ciclo de treinamento.
Ele adapta somente o dispositivo (CPU ou CUDA), gera os manifestos e salva os artefatos da execucao.

## Protocolo executado

- treino com o `setting1` balanceado completo: 600 imagens `fresh` e 600 `rotten`;
- validacao exclusivamente na pasta externa `test`: 100 imagens por classe;
- nenhuma imagem do `setting1` foi retirada para validacao interna;
- parametros do exemplo para imagens fornecido pelo autor: 5 epocas, 350 x 230 px, 2 caracteristicas,
  `gamma=1`, `relax=1`, Adadelta com `lr=0.1`, `svclose=5`, `shclose=0`, `wrclose=1`, `norm=0`,
  `aug=1` e batch 256;
- semente do modelo: 0.

Comando usado:

```powershell
.\.venv-yolo\Scripts\python.exe .\executar_lmfcn_original.py `
  --setting setting1 `
  --data-protocol external `
  --output C:\Users\matheus\Documents\datasetmorangonovo\resultados_lmfcn_original\setting1_treino_completo `
  --threads 4
```

## Resultado

A melhor validacao na pasta `test` ocorreu na epoca 4:

- acuracia: **85,00%**;
- recall `fresh`: **79,00%**;
- recall `rotten`: **91,00%**;
- F1 macro derivado da matriz de confusao: **84,95%**.

Historico da validacao em `test`: 62,5%, 73,5%, 75,0%, 85,0% e 71,0%.

Os resultados e artefatos foram gravados em
`C:\Users\matheus\Documents\datasetmorangonovo\resultados_lmfcn_original\setting1_treino_completo`.

## Limite de reprodutibilidade

O codigo-fonte do LMFCN foi preservado, mas a execucao utilizou Python 3.12, PyTorch 2.14.1 CPU e
scikit-learn 1.9.1. O ambiente citado pelo autor era Python 3.9, PyTorch 1.11, CUDA 11.3,
torchvision 0.12 e scikit-learn 1.0.2. Portanto, esta e uma execucao literal do algoritmo em um
ambiente compativel atual, nao uma reproducao numerica bit a bit do ambiente antigo.
