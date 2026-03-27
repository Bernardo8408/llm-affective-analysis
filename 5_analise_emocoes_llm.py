# -*- coding: utf-8 -*-
"""
Script de Classificação de Emoções - Dissertação Bernardo (IESP-UERJ)
Modelo: Gemini 2.5 Pro (Vertex AI)
Metodologia: LangChain Chunking + Chain of Thought + Few-Shot + Binário (0/1)
"""
!pip install -q -U google-genai tqdm tenacity langchain-text-splitters
import os
import json
import re
import signal
import pandas as pd
from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from json import JSONDecoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
from google.colab import drive, auth

drive.mount('/content/drive')

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_PATH = "/content/drive/MyDrive/WAPOR_DISSERTACAO/dados"
ARQUIVO_ENTRADA = os.path.join(BASE_PATH, "base_mestre_final_2026_HIGIENIZADA.csv")
ARQUIVO_SAIDA_CHUNKS = os.path.join(BASE_PATH, "analise_emocoes_chunks_raw.csv")
ARQUIVO_SAIDA_FINAL = os.path.join(BASE_PATH, "base_emocoes_agregada_trajetorias.csv")

# --- AUTENTICAÇÃO VERTEX AI ---
print("Solicitando autenticação do Google Cloud...")
auth.authenticate_user()
PROJECT_ID = "XXXXXXXX"

try:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")
    print("✅ Conectado à Vertex AI com sucesso!")
except Exception as e:
    print(f"❌ Erro na conexão Vertex AI: {e}")
    raise e

# --- PROMPT E MODELO ---
system_prompt = """Você é um assistente de pesquisa em sociologia das emoções analisando o debate político digital (PL 2630).
Sua tarefa é ler um trecho (chunk) de transcrição de vídeo e identificar a manifestação de 5 categorias emocionais (Lazarus, 1991; Marcus et al., 2000).

INSTRUÇÕES DE AVALIAÇÃO (APPRAISAL):
1. emocao_raiva: Indignação moral, atribuição de culpa a um agente intencional, desejo de retaliação ou confronto.
2. emocao_medo: Ansiedade, percepção de vulnerabilidade ou ameaça iminente (vigilância) sem um agente culpável claro no momento.
3. emocao_alegria: Celebração, entusiasmo, ironia comemorativa ou validação do próprio grupo.
4. emocao_nojo: Aversão moral extrema, repulsa, desumanização do adversário.
5. emocao_neutro: Relato factual, leitura técnica de documentos, ausência de valência emocional.

METODOLOGIA:
Passo 1: Faça uma breve análise interna (raciocínio) sobre o tom afetivo do trecho.
Passo 2: Atribua o valor INTEIRO 1 (emoção claramente manifesta) ou 0 (emoção ausente). Múltiplas emoções podem receber 1 simultaneamente.

=== EXEMPLOS DE CLASSIFICAÇÃO ===
EXEMPLO 1:
Texto: "Esses ministros canalhas estão destruindo o país de propósito! É um absurdo o que estão fazendo com a nossa constituição, não podemos aceitar calados!"
Saída:
{
  "raciocinio": {
    "analise_trecho": "O locutor demonstra forte indignação moral, atribui culpa direta a agentes ('ministros canalhas') e incita confronto, caracterizando Raiva. Há uso de linguagem repulsiva que tangencia o Nojo."
  },
  "classificacao": {
    "emocao_raiva": 1,
    "emocao_medo": 0,
    "emocao_alegria": 0,
    "emocao_nojo": 1,
    "emocao_neutro": 0
  }
}

EXEMPLO 2:
Texto: "O PL 2630 foi colocado em pauta nesta terça-feira. O relator apresentou um novo substitutivo."
Saída:
{
  "raciocinio": {
    "analise_trecho": "Linguagem puramente descritiva e técnica, sem valência afetiva ou julgamento de valor."
  },
  "classificacao": {
    "emocao_raiva": 0,
    "emocao_medo": 0,
    "emocao_alegria": 0,
    "emocao_nojo": 0,
    "emocao_neutro": 1
  }
}
============================================

SAÍDA OBRIGATÓRIA (Formato JSON estrito):
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

# --- CHUNKING E EXTRAÇÃO ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400, length_function=len)
EXPECTED_EMOTIONS = ["emocao_raiva", "emocao_medo", "emocao_alegria", "emocao_nojo", "emocao_neutro"]

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

@retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(10), retry=retry_if_exception_type(Exception))
def analisar_chunk(texto_chunk):
    prompt = f"Analise o seguinte trecho:\n---\n{texto_chunk}\n---\nSiga a estrutura JSON exigida."
    try:
        response = client.models.generate_content(model='gemini-2.5-pro', contents=prompt, config=configuracao_geracao)
        resultado = extract_json_robust(response.text)
        if resultado and "classificacao" in resultado:
            classificacao = {k: int(resultado["classificacao"].get(k, 0)) for k in EXPECTED_EMOTIONS}
            raciocinio = resultado.get("raciocinio", {}).get("analise_trecho", "")
            return classificacao, raciocinio
        return {k: 0 for k in EXPECTED_EMOTIONS}, "Falha na extração do JSON."
    except Exception as e:
        if "400" in str(e): raise ValueError("ERRO_400")
        raise e

# --- EXECUÇÃO ---
print("Carregando base de dados...")
df = pd.read_csv(ARQUIVO_ENTRADA, sep=',')
df_validos = df[df['transcricao'].notna() & (df['transcricao'].str.strip() != "")].copy()

lista_tarefas = []
for idx, row in df_validos.iterrows():
    id_video = row['id_video']
    chunks = text_splitter.split_text(row['transcricao'])
    for i, chunk_text in enumerate(chunks):
        lista_tarefas.append({'id_video': id_video, 'chunk_id': i + 1, 'texto': chunk_text})

resultados_chunks = []
start_pos = 0

if os.path.exists(ARQUIVO_SAIDA_CHUNKS):
    try:
        df_check = pd.read_csv(ARQUIVO_SAIDA_CHUNKS, sep=';')
        if not df_check.empty:
            resultados_chunks = df_check.to_dict('records')
            start_pos = len(resultados_chunks)
            print(f"✅ Retomando do chunk {start_pos}...")
    except pd.errors.EmptyDataError:
        pass

def salvar_checkpoint_chunks(rows):
    pd.DataFrame(rows).to_csv(ARQUIVO_SAIDA_CHUNKS, sep=';', index=False)

def handle_sigint(signum, frame):
    print("\nSalvando progresso...")
    salvar_checkpoint_chunks(resultados_chunks)
    raise KeyboardInterrupt()

signal.signal(signal.SIGINT, handle_sigint)

print("Iniciando classificação de emoções...")
for pos in tqdm(range(start_pos, len(lista_tarefas)), desc="Analisando Emoções"):
    tarefa = lista_tarefas[pos]
    try:
        classificacao, raciocinio = analisar_chunk(tarefa['texto'])
    except ValueError:
        classificacao, raciocinio = {k: 0 for k in EXPECTED_EMOTIONS}, "Erro 400"

    linha = {'id_video': tarefa['id_video'], 'chunk_id': tarefa['chunk_id'], 'raciocinio_llm': raciocinio}
    linha.update(classificacao)
    resultados_chunks.append(linha)

    if (pos + 1) % 10 == 0: salvar_checkpoint_chunks(resultados_chunks)

salvar_checkpoint_chunks(resultados_chunks)

# --- AGREGAÇÃO ---
print("\nGerando base final de emoções...")
df_chunks = pd.DataFrame(resultados_chunks).sort_values(by=['id_video', 'chunk_id'])
agg_dict = {f"{emocao}_densidade": pd.NamedAgg(column=emocao, aggfunc='mean') for emocao in EXPECTED_EMOTIONS}
df_agregado = df_chunks.groupby('id_video').agg(**agg_dict).reset_index()

colunas_meta = [c for c in df_validos.columns if c != 'transcricao']
df_final = df_validos[colunas_meta].merge(df_agregado, on='id_video', how='inner')
df_final.to_csv(ARQUIVO_SAIDA_FINAL, sep=';', index=False, encoding='utf-8')
print(f"✅ Sucesso! Base de emoções salva em: {ARQUIVO_SAIDA_FINAL}")
