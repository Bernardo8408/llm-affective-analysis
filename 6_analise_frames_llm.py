# -*- coding: utf-8 -*-
"""
Script de Classificação de Frames - Dissertação Bernardo (IESP-UERJ)
Modelo: Gemini 2.5 Pro
Metodologia: LangChain Chunking + Chain of Thought + Few-Shot + Binário (0/1) + Trajetórias
"""

# ==========================================
# 1. INSTALAÇÃO E SETUP
# ==========================================
print("Instalando bibliotecas necessárias...")
# Removido o pandas para evitar conflito com o Colab. Usando o novo google-genai.
!pip install -q -U google-genai tqdm tenacity langchain-text-splitters

from google import genai
from google.genai import types
import pandas as pd
import json
import time
import os
import re
import signal
from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from json import JSONDecoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.colab import drive

drive.mount('/content/drive')

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_PATH = "/content/drive/MyDrive/WAPOR_DISSERTACAO/dados"
ARQUIVO_ENTRADA = os.path.join(BASE_PATH, "base_mestre_final_2026_HIGIENIZADA.csv")
ARQUIVO_SAIDA_CHUNKS = os.path.join(BASE_PATH, "analise_frames_chunks_raw.csv")
ARQUIVO_SAIDA_FINAL = os.path.join(BASE_PATH, "base_frames_agregada_trajetorias.csv")

# --- AUTENTICAÇÃO VERTEX AI ---
from google.colab import auth
print("Solicitando autenticação do Google Cloud...")
auth.authenticate_user()

PROJECT_ID = "XXXXXXX"

try:
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="us-central1"
    )
    print("✅ Conectado à Vertex AI com sucesso!")
except Exception as e:
    print(f"❌ Erro na conexão Vertex AI: {e}")
    raise e

# ==========================================
# 2. CONFIGURAÇÃO DO MODELO E PROMPT
# ==========================================
system_prompt = """Você é um assistente de pesquisa sociológica especializado em análise de discurso e extrema-direita no Brasil (debate do PL 2630).
Sua tarefa é ler um trecho (chunk) de uma transcrição de vídeo e classificar a presença de 5 frames políticos.

INSTRUÇÕES DE ANÁLISE (Szwako, 2023 - Liberdade Reacionária):
1. frame_ameaca_a_liberdade: O Estado/STF é retratado como inimigo; regulação é associada a censura ou ditadura.
2. frame_vitimismo: Construção de identidade perseguida, silenciamento do grupo ('nos calaram', 'sistema contra nós').
3. frame_desconfianca_epistemica: Ataque à mídia tradicional ou instituições de checagem.
4. frame_religioso_moral: Pânico moral, associação da regulação à perseguição cristã ou destruição da família.
5. frame_resistencia_e_mobilizacao: Chamado explícito para ação, resistência ou mobilização (ruas, redes).

METODOLOGIA OBRIGATÓRIA:
Passo 1: Faça uma breve análise interna (raciocínio) sobre o trecho.
Passo 2: Atribua o valor INTEIRO 1 (presença clara e inequívoca do frame) ou 0 (ausência ou menção muito fraca/ambígua).

=== EXEMPLOS DE CLASSIFICAÇÃO (FEW-SHOT) ===

EXEMPLO 1:
Texto: "Eles aprovaram urgência nesse PL da Censura. O consórcio da velha imprensa, a Rede Globo, já comemora. Mas nós não vamos aceitar que um ministro do STF dite o que a família cristã pode ler na internet. Se preparem, domingo todo mundo na Paulista!"
Saída:
{
  "raciocinio": {
    "analise_trecho": "O locutor chama o PL 2630 de 'PL da Censura' (ameaça à liberdade), ataca a 'velha imprensa' (desconfiança epistêmica), evoca a 'família cristã' como alvo (religioso/moral) e convoca manifestação na Paulista (mobilização)."
  },
  "classificacao": {
    "frame_ameaca_a_liberdade": 1,
    "frame_vitimismo": 0,
    "frame_desconfianca_epistemica": 1,
    "frame_religioso_moral": 1,
    "frame_resistencia_e_mobilizacao": 1
  }
}

EXEMPLO 2:
Texto: "Nós estamos aqui há anos lutando e a verdade é que o sistema opera contra a direita. Eles derrubam nossos canais, cortam nossa monetização. É um massacre o que fazem conosco todos os dias, mas continuaremos falando."
Saída:
{
  "raciocinio": {
    "analise_trecho": "Discurso fortemente ancorado na narrativa de perseguição do grupo ('sistema opera contra a direita', 'derrubam nossos canais', 'massacre'), caracterizando vitimismo estrutural. Não há pânico moral religioso explícito ou convocação para ação nas ruas."
  },
  "classificacao": {
    "frame_ameaca_a_liberdade": 0,
    "frame_vitimismo": 1,
    "frame_desconfianca_epistemica": 0,
    "frame_religioso_moral": 0,
    "frame_resistencia_e_mobilizacao": 0
  }
}

EXEMPLO 3:
Texto: "O projeto de lei 2630 que está tramitando na Câmara dos Deputados cria regras para as redes sociais e plataformas de busca. A votação foi adiada após pedido dos relatores."
Saída:
{
  "raciocinio": {
    "analise_trecho": "Linguagem puramente técnica e institucional. Não apresenta nenhum dos frames reacionários ou afetivos."
  },
  "classificacao": {
    "frame_ameaca_a_liberdade": 0,
    "frame_vitimismo": 0,
    "frame_desconfianca_epistemica": 0,
    "frame_religioso_moral": 0,
    "frame_resistencia_e_mobilizacao": 0
  }
}
============================================

SAÍDA OBRIGATÓRIA (Formato JSON estrito baseado no trecho fornecido):
"""

configuracao_geracao = types.GenerateContentConfig(
    system_instruction=system_prompt,
    temperature=0.0,
    response_mime_type="application/json",
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]
)

# ==========================================
# 3. DIVISÃO DE TEXTO (CHUNKING)
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=400,
    length_function=len,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
)

# ==========================================
# 4. FUNÇÕES DE PROCESSAMENTO
# ==========================================
EXPECTED_FRAMES = [
    "frame_ameaca_a_liberdade", "frame_vitimismo",
    "frame_desconfianca_epistemica", "frame_religioso_moral",
    "frame_resistencia_e_mobilizacao"
]

def extract_json_robust(text):
    try: return json.loads(text)
    except: pass
    start = text.find('{')
    if start == -1: return None
    try:
        decoder = JSONDecoder()
        obj, end = decoder.raw_decode(text[start:])
        return obj
    except:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except: pass
        return None

# Usamos Exception genérica para o retry, pois os erros específicos da API mudaram no novo SDK
@retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(10), retry=retry_if_exception_type(Exception))
def analisar_chunk(texto_chunk):
    prompt = f"Analise o seguinte trecho da transcrição:\n---\n{texto_chunk}\n---\nSiga a estrutura JSON exigida."
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=configuracao_geracao
        )
        resultado = extract_json_robust(response.text)

        if resultado and "classificacao" in resultado:
            classificacao = {k: int(resultado["classificacao"].get(k, 0)) for k in EXPECTED_FRAMES}
            raciocinio = resultado.get("raciocinio", {}).get("analise_trecho", "")
            return classificacao, raciocinio
        else:
            return {k: 0 for k in EXPECTED_FRAMES}, "Falha na extração do JSON."

    except Exception as e:
        if "400" in str(e): raise ValueError("ERRO_400") # Trata violações de API isoladamente
        raise e

# ==========================================
# 5. LOOP PRINCIPAL (PROCESSAMENTO DE CHUNKS)
# ==========================================
print("Carregando base de dados...")
df = pd.read_csv(ARQUIVO_ENTRADA, sep=',') # Ajuste o separador se necessário
df_validos = df[df['transcricao'].notna() & (df['transcricao'].str.strip() != "")].copy()

lista_tarefas = []
for idx, row in df_validos.iterrows():
    id_video = row['id_video']
    transcricao = row['transcricao']
    chunks = text_splitter.split_text(transcricao)

    for i, chunk_text in enumerate(chunks):
        lista_tarefas.append({
            'id_video': id_video,
            'chunk_id': i + 1,
            'total_chunks': len(chunks),
            'texto': chunk_text
        })

print(f"Total de vídeos válidos: {len(df_validos)}")
print(f"Total de chunks gerados para análise: {len(lista_tarefas)}")

resultados_chunks = []
start_pos = 0

if os.path.exists(ARQUIVO_SAIDA_CHUNKS):
    try:
        df_check = pd.read_csv(ARQUIVO_SAIDA_CHUNKS, sep=';')
        if not df_check.empty:
            resultados_chunks = df_check.to_dict('records')
            start_pos = len(resultados_chunks)
            print(f"✅ Retomando do chunk {start_pos}...")
        else:
            print("⚠️ Checkpoint encontrado, mas sem dados. Iniciando do zero...")
    except pd.errors.EmptyDataError:
        print("⚠️ Checkpoint vazio/corrompido detectado. Ignorando e iniciando do zero...")

def salvar_checkpoint_chunks(rows):
    pd.DataFrame(rows).to_csv(ARQUIVO_SAIDA_CHUNKS, sep=';', index=False)

def handle_sigint(signum, frame):
    print("\nSalvando progresso parcial...")
    salvar_checkpoint_chunks(resultados_chunks)
    raise KeyboardInterrupt()

signal.signal(signal.SIGINT, handle_sigint)

print("Iniciando requisições à API...")
for pos in tqdm(range(start_pos, len(lista_tarefas)), desc="Analisando Chunks"):
    tarefa = lista_tarefas[pos]

    try:
        classificacao, raciocinio = analisar_chunk(tarefa['texto'])
    except ValueError:
        print(f"⚠️ Erro 400 no vídeo {tarefa['id_video']} chunk {tarefa['chunk_id']}. Pulando.")
        classificacao, raciocinio = {k: 0 for k in EXPECTED_FRAMES}, "Erro 400"

    resultado_linha = {
        'id_video': tarefa['id_video'],
        'chunk_id': tarefa['chunk_id'],
        'raciocinio_llm': raciocinio
    }
    resultado_linha.update(classificacao)
    resultados_chunks.append(resultado_linha)

    if (pos + 1) % 10 == 0:
        salvar_checkpoint_chunks(resultados_chunks)

    # time.sleep(3) # Delay para respeitar cota da API

salvar_checkpoint_chunks(resultados_chunks)

# ==========================================
# 6. AGREGAÇÃO E CRIAÇÃO DAS TRAJETÓRIAS
# ==========================================
print("\nGerando base final com densidade (médias) e trajetórias (listas)...")
df_chunks = pd.DataFrame(resultados_chunks)
df_chunks = df_chunks.sort_values(by=['id_video', 'chunk_id'])

agg_dict = {}
for frame in EXPECTED_FRAMES:
    agg_dict[f"{frame}_densidade"] = pd.NamedAgg(column=frame, aggfunc='mean')
    agg_dict[f"{frame}_trajetoria"] = pd.NamedAgg(column=frame, aggfunc=list)

df_agregado = df_chunks.groupby('id_video').agg(**agg_dict).reset_index()

for frame in EXPECTED_FRAMES:
    df_agregado[f"{frame}_trajetoria"] = df_agregado[f"{frame}_trajetoria"].apply(json.dumps)

# Merge com metadados
colunas_meta = [c for c in df_validos.columns if c != 'transcricao']
df_final = df_validos[colunas_meta].merge(df_agregado, on='id_video', how='inner')

df_final.to_csv(ARQUIVO_SAIDA_FINAL, sep=';', index=False, encoding='utf-8')
print(f"✅ Sucesso! Base final consolidada salva em: {ARQUIVO_SAIDA_FINAL}")
