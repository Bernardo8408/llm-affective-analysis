from googleapiclient.discovery import build
import pandas as pd
import os
from isodate import parse_duration
from config import CONFIG

# ===========================================
# CONFIGURAÇÃO DA API
# ===========================================
def configurar_api():
    """Carrega a chave da API e cria o cliente do YouTube"""
    try:
        caminho_chave = os.path.join(os.path.dirname(__file__), "Chave API.txt")
        with open(caminho_chave, "r") as arquivo:
            api_key = arquivo.read().strip()
        return build('youtube', 'v3', developerKey=api_key)
    except Exception as e:
        print(f"Erro ao configurar API: {e}")
        exit()

# ===========================================
# FUNÇÕES DE CONTROLE DE QUOTA
# ===========================================
class GerenciadorQuota:
    def __init__(self, limite=10000):
        self.quota_usada = 0
        self.limite = limite
    
    def usar(self, custo):
        if self.quota_usada + custo > self.limite:
            return False
        self.quota_usada += custo
        return True

# ===========================================
# FUNÇÃO PRINCIPAL DE COLETA
# ===========================================
def coletar_dados():
    youtube = configurar_api()
    gerenciador_quota = GerenciadorQuota()
    
    videos_data = []
    titulos_unicos = set()

    # Parâmetros da busca
    published_after = "2023-04-25T00:00:00Z"
    published_before = "2023-05-31T23:59:59Z"

    for term in CONFIG["TERMOS_BUSCA"]:
        for duration_filter in ["medium", "long"]:
            next_page_token = None
            
            while True:
                # Controle de quota
                if not gerenciador_quota.usar(100):
                    print(f"Quota esgotada para: {term} ({duration_filter})")
                    break
                
                try:
                    # Busca paginada
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
                    print(f"Erro na busca: {term} - {e}")
                    break
                
                # Processar resultados
                candidatos = []
                for item in resp['items']:
                    vid = item['id']['videoId']
                    sn = item['snippet']
                    title = sn['title']
                    pub = sn['publishedAt']
                    canal = sn['channelTitle']
                    
                    # Filtros iniciais
                    if 'cortes' in canal.lower():
                        continue
                    
                    if (title, pub) in titulos_unicos:
                        continue
                    
                    titulos_unicos.add((title, pub))
                    candidatos.append((vid, sn))
                
                # Obter estatísticas em batch
                for i in range(0, len(candidatos), 50):
                    if not gerenciador_quota.usar(1):
                        print("Quota esgotada durante obtenção de estatísticas")
                        break
                    
                    batch = candidatos[i:i+50]
                    ids_batch = [vid for vid, _ in batch]
                    
                    try:
                        stats_resp = youtube.videos().list(
                            part="statistics,contentDetails",
                            id=",".join(ids_batch)
                        ).execute()
                    except Exception as e:
                        print(f"Erro nas estatísticas: {e}")
                        continue
                    
                    # Processar estatísticas
                    for vid_item in stats_resp.get('items', []):
                        vid_id = vid_item['id']
                        stats = vid_item.get('statistics', {})
                        content = vid_item.get('contentDetails', {})
                        
                        # Conversão de valores
                        views = int(stats.get('viewCount', 0))
                        comments = int(stats.get('commentCount', 0))
                        likes = int(stats.get('likeCount', 0))
                        
                        # Filtro de engajamento
                        if views < 1000 or comments < 10:
                            continue
                        
                        # Filtro de duração
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
                        
                        # Adicionar aos dados
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
                
                # Próxima página
                next_page_token = resp.get('nextPageToken')
                if not next_page_token:
                    break

    # Salvar resultados
    df = pd.DataFrame(videos_data)
    caminho_saida = os.path.join(CONFIG["PASTA_DADOS"], CONFIG["ARQUIVO_VIDEOS"])
    df.to_csv(caminho_saida, index=False)
    
    print(f"\n{'='*40}")
    print(f"Coleta concluída! Quota utilizada: {gerenciador_quota.quota_usada}/{gerenciador_quota.limite}")
    print(f"Vídeos coletados: {len(df)}")
    print(f"Arquivo salvo em: {caminho_saida}")

if __name__ == "__main__":

    coletar_dados()
