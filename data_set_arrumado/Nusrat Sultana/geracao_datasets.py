import os
import shutil
import random

# Diretório base original (mantido para rodar via acesso remoto)
BASE_DIR = r"C:\Users\carlos\Drive\Documentos\Diversos\Educação\Mestrado\2026_2_Computação Aplicada\Artigo - Classificação e seleção de morangos danificados\Dataset\Nusrat Sultana"

# Diretórios de origem
ORIGINAL_FRESH = os.path.join(BASE_DIR, "Original", "FreshStrawberry")
ORIGINAL_ROTTEN = os.path.join(BASE_DIR, "Original", "RottenStrawberry")

def organizar_dataset():
    print("Iniciando a amostragem e cópia do dataset...")

    # 1. Listar as imagens originais garantindo que sejam arquivos de imagem
    fresh_images = [f for f in os.listdir(ORIGINAL_FRESH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    rotten_images = [f for f in os.listdir(ORIGINAL_ROTTEN) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Total encontrado na origem: {len(fresh_images)} Fresh, {len(rotten_images)} Rotten.")

    # 2. Teste (20% do total = 40 imagens aleatórias de cada classe)
    test_fresh = random.sample(fresh_images, 40)
    test_rotten = random.sample(rotten_images, 40)

    # 3. Cenário 1 (Imagens restantes de treino: 160 de cada classe)
    c1_fresh = list(set(fresh_images) - set(test_fresh))
    c1_rotten = list(set(rotten_images) - set(test_rotten))

    # 4. Cenário 2 (Fresh = Todas do C1 [160]; Rotten = 50% do C1 [80])
    c2_fresh = c1_fresh.copy()
    c2_rotten = random.sample(c1_rotten, 80)

    # 5. Cenário 3 (Fresh = 50% do C1 [80]; Rotten = Todas do C1 [160])
    c3_fresh = random.sample(c1_fresh, 80)
    c3_rotten = c1_rotten.copy()

    # 6. Cenário 4 (Fresh = 80 aleatórias do C1; Rotten = 40 aleatórias do C1)
    c4_fresh = random.sample(c1_fresh, 80)
    c4_rotten = random.sample(c1_rotten, 40)

    # 7. Cenário 5 (Fresh = 40 aleatórias do C1; Rotten = 80 aleatórias do C1, exclusivas em relação ao C4)
    c5_fresh = random.sample(c1_fresh, 40)
    
    # Garantindo que as 80 de Rotten do Cenário 5 não contenham as 40 usadas no Cenário 4
    available_rotten_for_c5 = list(set(c1_rotten) - set(c4_rotten))
    c5_rotten = random.sample(available_rotten_for_c5, 80)

    # Função auxiliar para criar pasta e copiar os arquivos
    def copy_to_dest(file_list, src_folder, dest_folder_name, subfolder):
        dest_path = os.path.join(BASE_DIR, dest_folder_name, subfolder)
        os.makedirs(dest_path, exist_ok=True)
        for f in file_list:
            shutil.copy2(os.path.join(src_folder, f), os.path.join(dest_path, f))
        print(f" -> Copiadas {len(file_list)} imagens para {dest_folder_name}\\{subfolder}")

    # Executando as cópias físicas nos diretórios
    print("\n--- Montando Teste ---")
    copy_to_dest(test_fresh, ORIGINAL_FRESH, "Teste", "FreshStrawberry")
    copy_to_dest(test_rotten, ORIGINAL_ROTTEN, "Teste", "RottenStrawberry")

    print("\n--- Montando Cenário 1 ---")
    copy_to_dest(c1_fresh, ORIGINAL_FRESH, "Cenario1", "FreshStrawberry")
    copy_to_dest(c1_rotten, ORIGINAL_ROTTEN, "Cenario1", "RottenStrawberry")

    print("\n--- Montando Cenário 2 ---")
    copy_to_dest(c2_fresh, ORIGINAL_FRESH, "Cenario2", "FreshStrawberry")
    copy_to_dest(c2_rotten, ORIGINAL_ROTTEN, "Cenario2", "RottenStrawberry")

    print("\n--- Montando Cenário 3 ---")
    copy_to_dest(c3_fresh, ORIGINAL_FRESH, "Cenario3", "FreshStrawberry")
    copy_to_dest(c3_rotten, ORIGINAL_ROTTEN, "Cenario3", "RottenStrawberry")

    print("\n--- Montando Cenário 4 ---")
    copy_to_dest(c4_fresh, ORIGINAL_FRESH, "Cenario4", "FreshStrawberry")
    copy_to_dest(c4_rotten, ORIGINAL_ROTTEN, "Cenario4", "RottenStrawberry")

    print("\n--- Montando Cenário 5 ---")
    copy_to_dest(c5_fresh, ORIGINAL_FRESH, "Cenario5", "FreshStrawberry")
    copy_to_dest(c5_rotten, ORIGINAL_ROTTEN, "Cenario5", "RottenStrawberry")

    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    organizar_dataset()