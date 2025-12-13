import os
import pandas as pd
import yt_dlp
from config import CONFIG
import time

# ===========================================
# CONFIGURAÇÕES
# ===========================================
def configurar_ydl():
    """Configurações do yt-dlp para download de áudio"""
    return {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(CONFIG["PASTA_AUDIOS"], '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': CONFIG["FORMATO_AUDIO"],
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': False,
        'sleep_interval': 2  # Intervalo entre downloads para evitar bloqueio
    }

# ===========================================
# FUNÇÕES PRINCIPAIS
# ===========================================
def baixar_audio(video_id):
    """Tenta baixar o áudio de um vídeo pelo ID"""
    try:
        url = f'https://www.youtube.com/watch?v={video_id}'
        with yt_dlp.YoutubeDL(configurar_ydl()) as ydl:
            info = ydl.extract_info(url, download=True)
        return True
    except Exception as e:
        print(f"Erro no download {video_id}: {str(e)}")
        return False

def main():
    # Criar pasta para áudios
    os.makedirs(CONFIG["PASTA_AUDIOS"], exist_ok=True)
    
    # Carregar dados
    caminho_analise = os.path.join(CONFIG["PASTA_DADOS"], CONFIG["ARQUIVO_FINAL"])
    
    try:
        df = pd.read_csv(caminho_analise)
    except Exception as e:
        print(f"Erro ao ler arquivo CSV: {str(e)}")
        return

    # Verificar coluna obrigatória
    if 'id_video' not in df.columns:
        print("Erro: Coluna 'id_video' não encontrada no arquivo CSV!")
        return

    # Filtrar IDs únicos e válidos
    ids_unicos = df['id_video'].unique()
    total = len(ids_unicos)
    print(f"\nTotal de vídeos para processar: {total}")

    # Contadores
    success = 0
    errors = 0
    existentes = 0

    # Processar downloads
    for idx, video_id in enumerate(ids_unicos, 1):
        # Verificar ID válido
        if not isinstance(video_id, str) or len(video_id) != 11:
            print(f"\nID inválido na posição {idx}: {video_id}")
            errors += 1
            continue

        # Verificar se arquivo já existe
        caminho_arquivo = os.path.join(
            CONFIG["PASTA_AUDIOS"], 
            f"{video_id}.{CONFIG['FORMATO_AUDIO']}"
        )
        
        if os.path.exists(caminho_arquivo):
            print(f"\n[{idx}/{total}] {video_id} já existe. Pulando...")
            existentes += 1
            continue

        # Tentar download
        print(f"\n[{idx}/{total}] Baixando: {video_id}")
        if baixar_audio(video_id):
            success += 1
        else:
            errors += 1

        # Respeitar limite de downloads
        if success >= CONFIG["LIMITE_DOWNLOADS"]:
            print("\nLimite de downloads atingido!")
            break

    # Relatório final
    print("\n" + "="*50)
    print(" Relatório Final ".center(50, "="))
    print(f"Vídeos processados: {total}")
    print(f"Downloads bem-sucedidos: {success}")
    print(f"Arquivos já existentes: {existentes}")
    print(f"Erros: {errors}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()