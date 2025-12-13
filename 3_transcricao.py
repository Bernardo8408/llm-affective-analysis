# --- Bloco 1: Instalação e Montagem do Google Drive ---
print("Instalando o Whisper...")
!pip install git+https://github.com/openai/whisper.git

print("Importando bibliotecas...")
import whisper
import pandas as pd
import os
import glob
import torch
import csv
from google.colab import drive
from tqdm.auto import tqdm # Para uma barra de progresso

# Monta o seu Google Drive
# Uma janela de permissão aparecerá. Autorize.
print("Conectando ao Google Drive...")
drive.mount('/content/drive')

print("Drive montado com sucesso.")
# --- Bloco 2: Configuração e Carregamento do Modelo ---

# !!! MUDE ESTES CAMINHOS !!!
# 1. Defina o caminho para a pasta onde seus 352+ áudios .mp3 estão no Drive
AUDIO_FOLDER = "/content/drive/MyDrive/WAPOR_DISSERTACAO/dados/audios/audios" # Exemplo: Mude para o seu caminho real

# 2. Defina o caminho ONDE o CSV de resultados será salvo no seu Drive
# Este arquivo será o seu "checkpoint".
CHECKPOINT_CSV = "/content/drive/MyDrive/WAPOR_DISSERTACAO/transcricoes_checkpoint.csv"

# 3. Checar e mover para GPU (T4)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU detectada. Carregando modelo 'large-v3' na GPU...")
else:
    print("AVISO: GPU não detectada. O processo será INVIÁVEL. Ative a GPU.")

# 4. Carregar o modelo Whisper (large-v3 é o mais preciso)
model = whisper.load_model("large-v3", device=device)

print("Modelo Whisper carregado com sucesso.")
# --- Bloco 3: Lógica de Checkpointing (Adaptada para Paralelismo) ---
import random

print("Verificando arquivos...")

# 1. Garante que a pasta de checkpoint exista
checkpoint_directory = os.path.dirname(CHECKPOINT_CSV)
if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)
    print(f"Pasta '{checkpoint_directory}' criada.")

# 2. Cria o arquivo de checkpoint se ele não existir
if not os.path.exists(CHECKPOINT_CSV):
    with open(CHECKPOINT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id_video', 'transcricao']) # Escreve o cabeçalho
    print(f"Arquivo de checkpoint criado em: {CHECKPOINT_CSV}")

# 3. Pega a lista de TODOS os áudios .mp3 que existem na pasta do Drive
caminho_pesquisa_audios = os.path.join(AUDIO_FOLDER, "*.mp3")
todos_os_audios_mp3 = glob.glob(caminho_pesquisa_audios)
print(f"Encontrados {len(todos_os_audios_mp3)} arquivos .mp3 na pasta do Drive.")

# 4. !!! MUDANÇA CRUCIAL: EMBARALHAR A LISTA !!!
# Isso garante que a Conta A e a Conta B comecem em pontos diferentes
# e não tentem pegar o mesmo áudio ao mesmo tempo.
random.shuffle(todos_os_audios_mp3)
print("Lista de áudios embaralhada para processamento paralelo.")
# --- Bloco 4: O Loop Principal (Adaptado para Paralelismo) ---
# AVISO: Este bloco vai levar MUITAS HORAS.

print(f"Iniciando transcrição paralela de {len(todos_os_audios_mp3)} áudios...")

# Itera sobre a lista TOTAL e embaralhada de áudios
for audio_path in tqdm(todos_os_audios_mp3, desc="Transcrevendo áudios"):
    try:
        # --- LÓGICA DE CHECKPOINT (DENTRO DO LOOP) ---

        # 1. Extrai o ID do áudio
        filename = os.path.basename(audio_path)
        id_video = filename.replace(".mp3", "")

        # 2. Re-lê o CSV de checkpoint (a cada loop)
        # Isso garante que ele veja o trabalho que a *outra conta* acabou de salvar
        df_done = pd.read_csv(CHECKPOINT_CSV)
        videos_ja_processados = set(df_done['id_video'])

        # 3. Verifica se este áudio já foi feito (por esta ou outra conta)
        if id_video in videos_ja_processados:
            # print(f"PULANDO: {id_video} (já processado)")
            continue # Pula para o próximo áudio

        # --- Se não foi processado, transcreve ---

        # print(f"PROCESSANDO: {id_video}")
        result = model.transcribe(audio_path, language="pt", fp16=torch.cuda.is_available())
        transcricao_texto = result['text']

        # --- CHECKPOINTING (SALVAMENTO IMEDIATO) ---
        # 4. Salva este 1 resultado. Agora as outras contas verão que ele está pronto.
        with open(CHECKPOINT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([id_video, transcricao_texto])

    except Exception as e:
        print(f"--- FALHA AO PROCESSAR: {id_video} ---")
        print(f"Erro: {e}")

        # Salva a falha para não tentar de novo
        with open(CHECKPOINT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([id_video, f"FALHA_NA_TRANSCRICAO: {e}"])

print("--- Processamento Concluído! ---")