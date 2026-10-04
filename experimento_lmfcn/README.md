# Experimento modular CNN + SVM

Esta implementação independente segue a ideia do [LMFCN de Jonathan Matos](https://github.com/jonathandematos/lmfcn): uma CNN produz vetores de características; um SVM com kernel RBF os classifica; os vetores de suporte e os erros do SVM escolhem exemplos para atualizar a CNN. O SVM não é diferenciado diretamente. A atualização usa pares da mesma classe (aproximação) e da classe oposta (margem). É uma adaptação modular da ideia, não uma reprodução numérica exata do código original.

## Dados

Por padrão, `prepare` lê `Dataset/Nusrat Sultana/Original`, com 200 imagens frescas e 200 podres. Usa 70% para treino, 15% para validação e 15% para teste em cada classe, com semente fixa. A pasta `Augmented` não entra na divisão para evitar vazamento de versões da mesma imagem. Os arquivos seguem o formato do Jonathan, `classe;caminho_absoluto_da_imagem`, com classe `0` = fresco e `1` = podre.

## Execução

O [README da raiz](../README.md) mostra como baixar o Dataset depois do clone e traz os comandos completos para Windows (PowerShell), Linux e macOS.

Os manifestos aparecem em `saida/manifests`; o modelo (`cnn.pt`, `svm.joblib`, `reference.npz`) e as métricas (`metrics.json`) aparecem em `saida/model`. Ambas as pastas são ignoradas pelo Git. O treino seleciona a época pela acurácia balanceada de validação e avalia o teste uma única vez, ao final. A execução escolhe CUDA, MPS (Mac) ou CPU automaticamente. O kernel do SVM exige memória proporcional ao quadrado do número de imagens de treino.

Para ajustar o experimento, consulte `python -m morango_lmfcn train --help`. Os manifestos gerados têm caminhos absolutos; ao mover o projeto, gere-os novamente. Carregue `svm.joblib` apenas de uma execução confiável, pois o formato usa serialização Python.
