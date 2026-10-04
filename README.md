# Classificação de morangos com CNN + SVM

Este repositório contém o experimento Python em [`experimento_lmfcn/`](experimento_lmfcn/) e usa a ideia do [LMFCN de Jonathan Matos](https://github.com/jonathandematos/lmfcn). O código de referência em `codigo_jonathan/` e as imagens em `Dataset/` não são incluídos no Git. 

## 1. Clonar e baixar o Dataset

Clone este repositório e entre na pasta criada pelo `git clone`. Depois, abra a [pasta do Dataset no Google Drive](https://drive.google.com/drive/folders/1d7YR44NljiyG3dp57XR8DcorJYzW8gAY) e use **Fazer download**. Extraia o arquivo ZIP baixado. Coloque o conteúdo extraído na raiz do repositório com o nome exato `Dataset` (D maiúsculo).

Antes de rodar os comandos, confira se os arquivos ficaram nestes caminhos:

```text
moranguinho/
├── Dataset/
│   └── Nusrat Sultana/
│       └── Original/
│           ├── FreshStrawberry/
│           │   └── ... imagens ...
│           └── RottenStrawberry/
│               └── ... imagens ...
└── experimento_lmfcn/
    └── requirements.txt
```

O programa usa as imagens dessas duas pastas `Original`. Se a extração criar uma pasta extra, como `Dataset/Dataset/`, mova o conteúdo da pasta interna para a externa. As outras pastas do download podem permanecer dentro de `Dataset`; elas não entram neste experimento. `Dataset/` está no `.gitignore`, então as imagens ficam apenas no computador de quem executa.

## 2. Instalar o Python e as dependências

Instale Python 3.9 ou superior, com `venv` e `pip`. É recomendável ter espaço livre para instalar o PyTorch. Todos os comandos abaixo devem começar na **raiz do repositório** após o clone.

### Windows (PowerShell)

```powershell
cd experimento_lmfcn
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Os exemplos usam o Python do ambiente virtual diretamente; não é necessário ativar o ambiente nem alterar a política de execução do PowerShell. Se `py` não estiver disponível, instale o Python pelo [site oficial](https://www.python.org/downloads/) com o Python Launcher.

### Linux

```bash
cd experimento_lmfcn
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Em distribuições que separam o módulo `venv`, instale o pacote correspondente à sua versão do Python (por exemplo, `python3-venv`) antes de criar o ambiente.

### macOS

```bash
cd experimento_lmfcn
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## 3. Preparar os dados, treinar e classificar

Execute os comandos **dentro de `experimento_lmfcn/`**, depois da instalação acima.

### Windows (PowerShell)

```powershell
.\.venv\Scripts\python.exe -m morango_lmfcn prepare --dataset ..\Dataset
.\.venv\Scripts\python.exe -m morango_lmfcn train
.\.venv\Scripts\python.exe -m morango_lmfcn predict "..\Dataset\Nusrat Sultana\Original\FreshStrawberry\FreshStrawberry_142.jpg"
```

### Linux e macOS

```bash
./.venv/bin/python -m morango_lmfcn prepare --dataset ../Dataset
./.venv/bin/python -m morango_lmfcn train
./.venv/bin/python -m morango_lmfcn predict "../Dataset/Nusrat Sultana/Original/FreshStrawberry/FreshStrawberry_142.jpg"
```

Substitua o caminho do último comando por qualquer imagem que queira classificar. O resultado é `fresco` ou `podre`.

`prepare` cria `saida/manifests/train.txt`, `val.txt` e `test.txt` a partir das 400 imagens originais: 280 para treino, 60 para validação e 60 para teste. Cada linha tem `classe;caminho_da_imagem`, onde `0` significa fresco e `1` significa podre. `train` salva a CNN, o SVM e as métricas em `saida/model/`. Esses arquivos também estão no `.gitignore`. O treino escolhe automaticamente CUDA, MPS (macOS compatível) ou CPU.

Os manifestos guardam caminhos absolutos. Se você mover o repositório ou o `Dataset`, apague `experimento_lmfcn/saida/manifests/` e execute `prepare` novamente antes de treinar. Para ajustar épocas, tamanho das imagens e outros parâmetros, use `train --help` com o mesmo executável Python mostrado acima. Mais detalhes sobre o modelo estão no [README do experimento](experimento_lmfcn/README.md).
