# --- Bloco 1: Instalação e Imports ---
print("Instalando bibliotecas...")
!pip install transformers torch pandas tqdm sentencepiece

import pandas as pd
import torch
from transformers import pipeline
from tqdm.auto import tqdm
import os
import gc

# Monta o Drive para acessar o modelo salvo e os dados
from google.colab import drive
drive.mount('/content/drive')

print("Ambiente pronto.")
# --- Bloco 2: Carregar Dados e Modelo ---

# 1. Caminhos (Configurados conforme sua pasta)
CAMINHO_MODELO = "/content/drive/MyDrive/WAPOR_DISSERTACAO/Modelos/bertimbau-emocao-finetuned"
CAMINHO_DADOS = "/content/drive/MyDrive/WAPOR_DISSERTACAO/dados/transcricoes.csv"

# 2. Carregar Transcrições
print(f"Carregando dados de: {CAMINHO_DADOS}")
try:
    # Importante: sep=';' conforme combinamos
    df = pd.read_csv(CAMINHO_DADOS, sep=';')

    # Filtra apenas linhas com transcrição válida (remove os NAs se houver)
    df_validos = df[df['transcricao'].notna() & (df['transcricao'].str.strip() != "")].copy()

    print(f"Total de linhas no arquivo: {len(df)}")
    print(f"Vídeos válidos para análise: {len(df_validos)}")

except Exception as e:
    print(f"Erro ao carregar CSV: {e}")

# 3. Carregar o SEU Modelo Fine-Tuned
print(f"Carregando modelo Fine-Tuned de: {CAMINHO_MODELO}")

device = 0 if torch.cuda.is_available() else -1

try:
    # AQUI ESTÁ O SEGREDO: function_to_apply="sigmoid"
    # Isso garante que os scores sejam independentes (0 a 1) e não somem 1 (softmax).
    emotion_classifier = pipeline(
        "text-classification",
        model=CAMINHO_MODELO,
        tokenizer=CAMINHO_MODELO,
        device=device,
        top_k=None,
        function_to_apply="sigmoid"
    )
    print("✅ Modelo carregado com sucesso na GPU!")

except Exception as e:
    print(f"❌ Erro ao carregar o modelo. Verifique se o caminho da pasta está correto. Erro: {e}")

# --- Bloco 3: Função de Processamento Ponderada ---
import numpy as np

# O BERTimbau tem limite de 512 tokens. Usamos 500 para segurança.
CHUNK_SIZE = 500

def get_emotion_scores(text):
    """
    Recebe a transcrição, quebra em chunks, classifica
    e retorna a média PONDERADA das probabilidades.
    """
    # 1. Tokeniza o texto inteiro
    tokens = emotion_classifier.tokenizer(text, return_tensors="pt", truncation=False)['input_ids'][0]
    chunk_data = []
    
    # 2. Loop pelos chunks
    for i in range(0, len(tokens), CHUNK_SIZE):
        chunk_tokens = tokens[i:i+CHUNK_SIZE]
        # Reconverte tokens em texto
        chunk_text = emotion_classifier.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # O PESO é o número de tokens neste pedaço
        weight = len(chunk_tokens)
        
        if chunk_text.strip():
            try:
                # Roda o modelo
                # Retorna lista de dicts: [{'label': 'LABEL_0', 'score': 0.8}, ...]
                # Nota: O modelo pode retornar 'LABEL_0' ou 'raiva' dependendo do config salvo.
                # Vamos tratar isso no próximo bloco.
                result = emotion_classifier(chunk_text)[0]
                
                # Cria dict: {'LABEL_0': 0.8, 'LABEL_1': 0.1}
                scores_dict = {item['label']: item['score'] for item in result}
                
                # Adiciona o peso
                scores_dict['_weight'] = weight
                chunk_data.append(scores_dict)
            except Exception as e:
                continue
    
    if not chunk_data:
        return None

    # 3. Cálculo da Média Ponderada
    df_chunks = pd.DataFrame(chunk_data)
    total_weight = df_chunks['_weight'].sum()
    final_scores = {}
    
    for col in df_chunks.columns:
        if col != '_weight':
            # Fórmula: Soma(Score * Peso) / Soma(Pesos)
            weighted_mean = (df_chunks[col] * df_chunks['_weight']).sum() / total_weight
            final_scores[col] = weighted_mean
            
    return final_scores

# --- Bloco 4: Executar Análise ---
# Isso deve ser relativamente rápido (BERT Base é eficiente)

print(f"Iniciando análise de {len(df_validos)} vídeos...")
tqdm.pandas(desc="Analisando Emoções")

# Aplica a função
emotion_results = df_validos['transcricao'].progress_apply(get_emotion_scores)

# Transforma os resultados (lista de dicts) em colunas do DataFrame
df_scores = pd.json_normalize(emotion_results)

# Garante que o índice bate para o merge
df_scores.index = df_validos.index

# Junta os scores com os dados originais (mantendo ID e Transcrição)
df_com_emocoes = pd.concat([df_validos, df_scores], axis=1)

print("Análise concluída!")
print(df_com_emocoes.head(3))
# --- Bloco 4.5: Renomear Colunas (Correção dos Labels) ---

# Dicionário de mapeamento baseado na ordem do nosso treino
mapa_labels = {
    'emo_LABEL_0': 'emo_raiva',
    'emo_LABEL_1': 'emo_medo',
    'emo_LABEL_2': 'emo_alegria',
    'emo_LABEL_3': 'emo_tristeza',
    'emo_LABEL_4': 'emo_surpresa',
    'emo_LABEL_5': 'emo_nojo',
    'emo_LABEL_6': 'emo_neutro'
}

# Renomeia as colunas no DataFrame
df_com_emocoes = df_com_emocoes.rename(columns=mapa_labels)

print("Colunas renomeadas com sucesso!")
print(df_com_emocoes.head(3))
# --- Bloco 5: Salvar Arquivo Parcial ---

CAMINHO_SAIDA = "/content/drive/MyDrive/WAPOR_DISSERTACAO/dados/base_dados_COM_EMOCOES.csv"

# Salva com separador ';' para manter consistência
df_com_emocoes.to_csv(CAMINHO_SAIDA, index=False, sep=';')

print(f"Arquivo salvo com sucesso em: {CAMINHO_SAIDA}")
print("Agora você tem as emoções calculadas pelo seu próprio modelo!")
print("Próximo passo: Análise de Frames (ZSC).")