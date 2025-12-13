# Engenharia do Afeto: Análise Computacional com LLMs

Este repositório contém o código e os notebooks desenvolvidos durante a pesquisa de Mestrado em Sociologia (IESP-UERJ).

O projeto investiga a "engenharia afetiva" em ecossistemas políticos digitais, utilizando métodos computacionais para identificar como emoções e estruturas narrativas são mobilizadas em vídeos do YouTube.

## Objetivo
Automatizar a leitura hermenêutica de grandes volumes de texto (transcrições de vídeo) para identificar padrões de polarização que escapam à análise de sentimento tradicional.

## Tech Stack & Metodologia
O pipeline foi construído inteiramente em **Python** e opera nas seguintes etapas:

* **Coleta de Dados:** Extração de transcrições via API do YouTube.
* **Processamento (NLP):** Limpeza e estruturação de dados textuais com `Pandas`.
* **Classificação com IA:** Uso de **Large Language Models (LLMs)** para anotação semântica de narrativas e afetos.
* **Análise:** Mensuração de engajamento e correlação com *features* emocionais.

## Estrutura do Repositório
* `/notebooks`: Jupyter Notebooks com os experimentos de classificação e análise exploratória.
* `/scripts`: Scripts Python para coleta e processamento em lote.
* `/data`: Amostras dos datasets utilizados (anonimizados).

## Contexto Acadêmico
* **Autor:** Bernardo Cruz
* **Instituição:** IESP-UERJ
* **Conferências:** Resultados preliminares apresentados na WAPOR Latam.

---
*Este código é parte de uma pesquisa acadêmica em andamento e focado em reprodutibilidade científica.*
