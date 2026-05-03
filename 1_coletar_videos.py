# 1. Instalação de dependências ausentes no Colab
!pip install isodate

# 2. Importações
from googleapiclient.discovery import build
import pandas as pd
import os
from isodate import parse_duration
from google.colab import drive

# ===========================================
# 3. CONECTAR AO GOOGLE DRIVE (Blindagem de Dados)
# ===========================================
print("Conectando ao Google Drive...")
drive.mount('/content/drive')

# ===========================================
# 4. COLE SUA CHAVE DE API AQUI DENTRO DAS ASPAS
# ===========================================
CHAVE_YOUTUBE = "Youtube_API.txt"

# ===========================================
# 5. CONFIGURAÇÕES DIRETAS NO DRIVE
# ===========================================
BASE_DIR = "/content/drive/MyDrive/WAPOR_DISSERTACAO"

CONFIG = {
    "PASTA_DADOS": os.path.join(BASE_DIR, "dados"),
    "ARQUIVO_VIDEOS": "videos_coletados.csv",
    "TERMOS_BUSCA": [
        "PL das Fake News",
        "liberdade de expressão",
        "regulação das redes",
        "censura",
        "fake news"
    ],
    "PASTA_AUDIOS": os.path.join(BASE_DIR, "dados", "audios"),
    "FORMATO_AUDIO": "mp3",
    "LIMITE_DOWNLOADS": 1000
}

os.makedirs(CONFIG["PASTA_DADOS"], exist_ok=True)
os.makedirs(CONFIG["PASTA_AUDIOS"], exist_ok=True)

def configurar_api():
    try:
        return build('youtube', 'v3', developerKey=CHAVE_YOUTUBE)
    except Exception as e:
        print(f"Erro ao configurar API: {e}")
        exit()

class GerenciadorQuota:
    def __init__(self, limite=10000):
        self.quota_usada = 0
        self.limite = limite

    def usar(self, custo):
        if self.quota_usada + custo > self.limite:
            return False
        self.quota_usada += custo
        return True

def coletar_dados():
    youtube = configurar_api()
    gerenciador_quota = GerenciadorQuota()

    videos_data = []
    titulos_unicos = set()

    published_after = "2023-04-25T00:00:00Z"
    published_before = "2023-05-31T23:59:59Z"

    print("\nIniciando varredura na API do YouTube...")

    for term in CONFIG["TERMOS_BUSCA"]:
        print(f"\n>>> Buscando termo: '{term}'")
        for duration_filter in ["medium", "long"]:
            print(f"  -> Filtro de duração: {duration_filter}")
            next_page_token = None
            pagina_atual = 1

            while True:
                if not gerenciador_quota.usar(100):
                    print(f"     [!] Quota esgotada para: {term}")
                    break

                try:
                    resp = youtube.search().list(
                        q=term,
                        part="snippet",
                        type="video",
                        maxResults=50,
                        publishedAfter=published_after,
                        publishedBefore=published_before,
                        order="viewCount",
                        relevanceLanguage="pt",
                        pageToken=next_page_token,
                        videoDuration=duration_filter
                    ).execute()
                except Exception as e:
                    print(f"     [ERRO] Falha na API: {e}")
                    break

                candidatos = []
                for item in resp.get('items', []):
                    vid = item['id']['videoId']
                    sn = item['snippet']
                    title = sn['title']
                    pub = sn['publishedAt']
                    canal = sn['channelTitle']

                    if 'cortes' in canal.lower():
                        continue
                    if (title, pub) in titulos_unicos:
                        continue

                    titulos_unicos.add((title, pub))
                    candidatos.append((vid, sn))

                print(f"     Página {pagina_atual}: {len(candidatos)} vídeos potenciais encontrados.")

                for i in range(0, len(candidatos), 50):
                    if not gerenciador_quota.usar(1):
                        print("     [!] Quota esgotada durante estatísticas.")
                        break

                    batch = candidatos[i:i+50]
                    ids_batch = [vid for vid, _ in batch]

                    try:
                        stats_resp = youtube.videos().list(
                            part="statistics,contentDetails",
                            id=",".join(ids_batch)
                        ).execute()
                    except Exception as e:
                        print(f"     [ERRO] Falha ao puxar metadados: {e}")
                        continue

                    for vid_item in stats_resp.get('items', []):
                        vid_id = vid_item['id']
                        stats = vid_item.get('statistics', {})
                        content = vid_item.get('contentDetails', {})

                        views = int(stats.get('viewCount', 0))
                        comments = int(stats.get('commentCount', 0))
                        likes = int(stats.get('likeCount', 0))

                        # O filtro de views e comentários original do seu script
                        if views < 1000 or comments < 10:
                            continue

                        try:
                            duration_sec = parse_duration(
                                content.get('duration', 'PT0S')
                            ).total_seconds()
                        except:
                            duration_sec = 0

                        if duration_filter == "medium" and not (240 <= duration_sec <= 1200):
                            continue
                        elif duration_filter == "long" and duration_sec <= 1200:
                            continue

                        sn = next(sn for vid, sn in batch if vid == vid_id)
                        videos_data.append({
                            'id_video': vid_id,
                            'termo_busca': term,
                            'filtro_duracao': duration_filter,
                            'titulo': sn['title'],
                            'descricao': sn['description'],
                            'data_publicacao': sn['publishedAt'],
                            'canal': sn['channelTitle'],
                            'duracao_segundos': duration_sec,
                            'visualizacoes': views,
                            'likes': likes,
                            'comentarios': comments
                        })

                next_page_token = resp.get('nextPageToken')
                if not next_page_token:
                    break

                pagina_atual += 1

    df = pd.DataFrame(videos_data)

    caminho_saida = os.path.join(CONFIG["PASTA_DADOS"], "videos_abril_pl2630.csv")
    df.to_csv(caminho_saida, index=False, encoding='utf-8', sep=';')

    print(f"\n{'='*40}")
    print(f"Coleta concluída com sucesso!")
    print(f"Quota utilizada: {gerenciador_quota.quota_usada}/{gerenciador_quota.limite}")
    print(f"Total de vídeos aprovados pelos filtros: {len(df)}")
    print(f"Arquivo salvo em: {caminho_saida}")

if __name__ == "__main__":
    coletar_dados()
