import os
from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict

# Diretório base usando pathlib (o mesmo esquema que funcionou no Win11)
BASE_DIR = Path.cwd()
PASTA_ORIGINAL = BASE_DIR / "Original"

def verificar_duplicatas_visuais():
    print("Iniciando a varredura visual das imagens originais...\n")
    
    if not PASTA_ORIGINAL.exists():
        print(f"Erro: A pasta {PASTA_ORIGINAL} não foi encontrada.")
        return

    extensoes_validas = ('.jpg', '.jpeg', '.png')
    hashes_encontrados = defaultdict(list)
    total_analisado = 0

    # Varre as subpastas (FreshStrawberry e RottenStrawberry)
    for root, _, files in os.walk(PASTA_ORIGINAL):
        pasta_atual = Path(root)
        
        for file in files:
            if file.lower().endswith(extensoes_validas):
                caminho_imagem = pasta_atual / file
                
                try:
                    # Abre a imagem e gera a "impressão digital" visual dela
                    img = Image.open(caminho_imagem)
                    # pHash é excelente para detectar imagens iguais mesmo com leve compressão
                    hash_visual = imagehash.phash(img) 
                    
                    # Armazena o caminho da imagem associado a esse hash
                    hashes_encontrados[hash_visual].append(caminho_imagem)
                    total_analisado += 1
                except Exception as e:
                    print(f"Erro ao ler a imagem {file}: {e}")

    print(f"Total de imagens analisadas: {total_analisado}")
    print("-" * 50)

    # Verifica se algum hash possui mais de uma imagem (ou seja, são duplicatas)
    duplicatas = {h: caminhos for h, caminhos in hashes_encontrados.items() if len(caminhos) > 1}

    if not duplicatas:
        print("✅ SUCESSO: Nenhuma imagem duplicada encontrada! As 400 imagens são visualmente únicas.")
    else:
        print(f"⚠️ ALERTA: Foram encontrados {len(duplicatas)} grupos de imagens duplicadas!\n")
        for i, (hash_val, caminhos) in enumerate(duplicatas.items(), 1):
            print(f"Grupo {i} (mesma imagem):")
            for caminho in caminhos:
                # Mostra o caminho relativo para ficar limpo no terminal
                print(f" -> {caminho.relative_to(BASE_DIR)}")
            print()

if __name__ == "__main__":
    verificar_duplicatas_visuais()