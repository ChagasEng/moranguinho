import os

# Diretório base
BASE_DIR = r"C:\Users\carlos\Drive\Documentos\Diversos\Educação\Mestrado\2026_2_Computação Aplicada\Artigo - Classificação e seleção de morangos danificados\Dataset\Nusrat Sultana"

def contar_imagens_nas_pastas():
    print(f"Contando imagens a partir de:\n{BASE_DIR}\n")
    print("-" * 50)
    
    if not os.path.exists(BASE_DIR):
        print("Erro: O diretório base não foi encontrado.")
        return

    extensoes_validas = ('.jpg', '.jpeg', '.png')
    
    # Percorrer toda a árvore de diretórios
    for root, dirs, files in os.walk(BASE_DIR):
        # Filtrar apenas imagens para a contagem
        imagens = [f for f in files if f.lower().endswith(extensoes_validas)]
        
        # Só exibe as pastas que contêm imagens
        if imagens:
            # Pega o caminho relativo para a exibição no console ficar mais limpa
            caminho_relativo = os.path.relpath(root, BASE_DIR)
            print(f"{caminho_relativo:<35} | {len(imagens):>4} imagens")
            
    print("-" * 50)
    print("Contagem finalizada.")

if __name__ == '__main__':
    contar_imagens_nas_pastas()