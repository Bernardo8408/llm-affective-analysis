# --- Bloco 1: Instalação, Login e Configuração ---
print("Instalando bibliotecas...")
!pip install transformers datasets scikit-learn accelerate -U

import pandas as pd
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from google.colab import drive
from huggingface_hub import login
import os

# 1. Monta o Drive
print("Conectando ao Google Drive...")
drive.mount('/content/drive')

# 2. Login no Hugging Face usando seu arquivo de token
TOKEN_PATH = "/content/drive/MyDrive/WAPOR_DISSERTACAO/dados/token_roberta.txt"

print(f"Lendo token de: {TOKEN_PATH}")
try:
    with open(TOKEN_PATH, 'r') as f:
        my_token = f.read().strip()
    
    if my_token:
        login(token=my_token, add_to_git_credential=False)
        print("✅ Login no Hugging Face realizado com sucesso!")
    else:
        print("❌ O arquivo de token está vazio.")
except FileNotFoundError:
    print(f"❌ Arquivo de token não encontrado em: {TOKEN_PATH}")
    print("Verifique se o caminho está correto no seu Google Drive.")

# 3. Define o caminho de salvamento do Modelo
OUTPUT_DIR = "/content/drive/MyDrive/WAPOR_DISSERTACAO/Modelos/bertimbau-emocao-finetuned"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"O modelo final será salvo em: {OUTPUT_DIR}")
# --- Bloco 2: Carregar e Mapear Manualmente (CORRIGIDO E DUPLICATAS REMOVIDAS) ---

print("Baixando dataset GoEmotions PT-BR via Pandas...")

# 1. Carregar direto da URL
url_csv = "https://huggingface.co/datasets/antoniomenezes/go_emotions_ptbr/resolve/main/goemotions_1_pt.csv"

try:
    df = pd.read_csv(url_csv)
    print(f"Dataset carregado com sucesso! {len(df)} linhas.")
except Exception as e:
    print(f"Erro ao baixar CSV: {e}")

# 2. Limpeza das Colunas
def limpar_colunas(cols):
    mapa_colunas = {}
    emoções_esperadas = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    for col in cols:
        for emo in emoções_esperadas:
            # Verifica se o nome da emoção está na coluna (ex: 'grid_3x3admiration')
            if emo in col.lower(): 
                mapa_colunas[col] = emo
                break
    return mapa_colunas

# Renomeia as colunas
df = df.rename(columns=limpar_colunas(df.columns))

# !!! CORREÇÃO CRÍTICA AQUI !!!
# Remove colunas duplicadas (se houver duas colunas 'joy', mantém só a primeira)
df = df.loc[:, ~df.columns.duplicated()]
print("Colunas duplicadas removidas.")

# 3. Mapeamento para Ekman (7 Emoções)
ekman_mapping = {
    'anger': 0, 'annoyance': 0, 'disapproval': 0,
    'fear': 1, 'nervousness': 1,
    'joy': 2, 'amusement': 2, 'approval': 2, 'excitement': 2, 'gratitude': 2, 'love': 2, 'optimism': 2, 'relief': 2, 'pride': 2, 'admiration': 2, 'desire': 2, 'caring': 2,
    'sadness': 3, 'disappointment': 3, 'embarrassment': 3, 'grief': 3, 'remorse': 3,
    'surprise': 4, 'realization': 4, 'confusion': 4, 'curiosity': 4,
    'disgust': 5,
    'neutral': 6
}

NUM_LABELS = 7

def criar_labels_ekman(row):
    labels_vec = [0.0] * NUM_LABELS
    has_label = False
    
    # Itera apenas sobre as chaves do mapeamento que existem como colunas no DF
    for emo_name, target_idx in ekman_mapping.items():
        if emo_name in row: # Verifica se a coluna existe
            # O valor pode ser 0/1 ou True/False, forçamos int para garantir
            if int(row[emo_name]) == 1:
                labels_vec[target_idx] = 1.0
                has_label = True
            
    if not has_label:
        labels_vec[6] = 1.0
        
    return labels_vec

print("Aplicando mapeamento Ekman...")
from tqdm.auto import tqdm
tqdm.pandas()
df['labels'] = df.progress_apply(criar_labels_ekman, axis=1)

# 4. Converter para Dataset
from datasets import Dataset

# Filtra só o que precisamos. Note que o nome da coluna de texto original pode variar.
# Vamos tentar achar a coluna de texto correta.
coluna_texto = 'text' # Padrão
if 'texto' in df.columns: coluna_texto = 'texto'
elif 'text_formattextsort' in df.columns: coluna_texto = 'text_formattextsort' # Nome estranho comum no Kaggle

print(f"Usando coluna de texto: {coluna_texto}")
df_final = df[[coluna_texto, 'labels']].rename(columns={coluna_texto: 'text'})

# Garante que o texto é string (remove nulos)
df_final = df_final.dropna(subset=['text'])
df_final['text'] = df_final['text'].astype(str)

full_dataset = Dataset.from_pandas(df_final)

# Divide em Treino (90%) e Validação (10%)
dataset_split = full_dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

print(f"Dataset Pronto! Treino: {len(train_dataset)}, Validação: {len(eval_dataset)}")
# --- Bloco 3: Tokenização (BERTimbau) - CORRIGIDO ---

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
print(f"Carregando tokenizador do BERTimbau ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # Tokeniza o texto em português, truncando para 128 tokens (padrão eficiente)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizando os datasets de treino e validação...")

# Aplicamos a tokenização separadamente no treino e na validação
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Define o formato para PyTorch (inputs numéricos que o modelo entende)
print("Formatando para PyTorch...")
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print("✅ Tokenização concluída.")
print(f"Treino pronto: {len(tokenized_train)} amostras.")
print(f"Validação pronta: {len(tokenized_eval)} amostras.")
# --- Bloco 4: Treino Otimizado (Versão F1 Tuning) ---
from transformers import EarlyStoppingCallback
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction
import numpy as np

# 1. Função de Métricas OTIMIZADA
# Agora ela testa um threshold mais baixo (0.35) que costuma ser melhor para emoções
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    y_true = p.label_ids

    # --- TRUQUE DO THRESHOLD ---
    # Em vez de fixar em 0.5, vamos usar 0.35. 
    # Isso captura emoções mais sutis e geralmente AUMENTA o F1.
    threshold = 0.35 
    
    y_pred = np.zeros(probs.shape)
    y_pred[probs >= threshold] = 1
    
    # Métricas Detalhadas
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro') # Importante para classes raras
    roc_auc = roc_auc_score(y_true, probs, average = 'micro')
    
    return {
        'f1_micro': f1_micro, 
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
        'threshold_used': threshold
    }

print("Carregando BERTimbau...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

# 2. Hiperparâmetros Ajustados
training_args = TrainingArguments(
    output_dir="./results",
    
    num_train_epochs=4,              # 4 épocas é suficiente se bem ajustado
    
    # Ajuste de Velocidade/Memória
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,   # Padrão
    
    # Otimização do Aprendizado
    learning_rate=3e-5,              # Levemente maior que o padrão (2e-5)
    warmup_ratio=0.1,                # 10% do tempo aquecendo
    weight_decay=0.01,
    
    # Logs e Avaliação
    logging_dir='./logs',
    logging_steps=100,               # Log mais frequente
    eval_strategy="steps", 
    eval_steps=500,                  # Avalia a cada 500 passos (mais rápido que 200)
    save_strategy="steps",
    save_steps=500,
    
    # Melhor Modelo
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro", # Focamos no F1 Micro
    greater_is_better=True,
    save_total_limit=1,              # Salva só o melhor
    fp16=True,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)] # Mais paciência
)

print("--- INICIANDO O TREINAMENTO OTIMIZADO ---")
# Inicia o TensorBoard para você ver o gráfico subindo
%load_ext tensorboard
%tensorboard --logdir ./logs

trainer.train()
print("Treino concluído.")
# --- Bloco 5: Salvar no Google Drive ---

print(f"Salvando o modelo final em: {OUTPUT_DIR}")

# 1. Salva o Modelo
model.save_pretrained(OUTPUT_DIR)

# 2. Salva o Tokenizer (MUITO IMPORTANTE para usar depois)
tokenizer.save_pretrained(OUTPUT_DIR)

print("--- SUCESSO TOTAL ---")
print("Seu modelo BERTimbau Fine-Tuned (Ekman) está salvo no seu Google Drive.")
print("Você pode usá-lo no Script 4 carregando o caminho da pasta.")