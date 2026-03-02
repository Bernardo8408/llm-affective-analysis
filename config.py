import os

CONFIG = {
    "PASTA_DADOS": os.path.join(os.path.dirname(__file__), "dados"),
    "ARQUIVO_VIDEOS": "videos_coletados.csv",
    "ARQUIVO_CLASSIFICACAO": "classificacao_canais.csv",
    "ARQUIVO_FINAL": "analise_final.csv",
    "TERMOS_BUSCA": [
        "PL das Fake News",
        "liberdade de expressão",
        "regulação das redes",
        "censura",
        "fake news"
    ],
    "PASTA_AUDIOS": os.path.join(os.path.dirname(__file__), "dados", "audios"),
    "FORMATO_AUDIO": "mp3",
    "LIMITE_DOWNLOADS": 1000
}

# Criar pastas se não existirem
os.makedirs(CONFIG["PASTA_DADOS"], exist_ok=True)
os.makedirs(CONFIG["PASTA_AUDIOS"], exist_ok=True)