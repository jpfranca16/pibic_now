# --- 1. BIBLIOTECAS ---
import streamlit as st
import geopandas as gpd
import pandas as pd
import geemap
import ee
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from plotly.subplots import make_subplots




#papel de parede com css:
#===============================================================
# usando css para boniteza:
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1743046813915-94cf6d5e6942?q=80&w=1528&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-attachment: fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- ELEMENTOS DA SIDEBAR ---
st.sidebar.markdown("## 🌦️ **DASHBOARD Clima**")
st.sidebar.caption("Painel de Análise e Predição Climática")
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Navegação & Filtros")
st.sidebar.info("Utilize as páginas acima para alternar entre as séries históricas e o modelo preditivo.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 0.8em; opacity: 0.8;">
        Desenvolvido com Python & Earth Engine<br>
        <b>Paraíba, Brasil</b>
    </div>
    """, 
    unsafe_allow_html=True
)


#================================================================
#================================================================




                            #TUDO COMEÇA AQUI:


# Ativa a opção para restaurar o arquivo .shx automaticamente
os.environ["SHAPE_RESTORE_SHX"] = "YES"

# Inicialização do Earth Engine
# ee.Authenticate() # Descomente se for a primeira execução nesta máquina

#======================================================================================

def inicializar_earth_engine():
    try:
        # Se estiver no Streamlit Cloud com os Secrets configurados
        if "gcp_service_account" in st.secrets:
            key_dict = dict(st.secrets["gcp_service_account"])
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], 
                key_data=key_dict["private_key"]
            )
            ee.Initialize(credentials=credentials, project='infinite-unity-500221-h5')
        else:
            # Se estiver rodando localmente no seu computador
            ee.Initialize(project='infinite-unity-500221-h5')
    except Exception as e:
        st.error(f"Erro ao inicializar o Earth Engine: {e}")

# Chame a função no início do seu script
inicializar_earth_engine()


#======================================================================================


PASTA_ATUAL = os.path.dirname(__file__)

# Aponta dinamicamente para a pasta do arquivo SHP dentro do seu projeto
CAMINHO_SHP = os.path.join(PASTA_ATUAL, "PB_Municipios_2025", "PB_Municipios_2025.shp")


# --- 2. LEITURA E FILTRAGEM DO SHAPEFILE ---
@st.cache_data
def carregar_municipios():
    municipios_all = gpd.read_file(CAMINHO_SHP)
# Filtra apenas os 8 municípios selecionados do PIBIC
    nomes_pibic = ["Patos", "Água Branca", "Imaculada", "Juru", "Manaíra", "Princesa Isabel", "Tavares", "Teixeira"]
    return municipios_all[municipios_all["NM_MUN"].isin(nomes_pibic)].sort_values("NM_MUN")

municipios_filtrados = carregar_municipios()

# --- 3. INTERFACE STREAMLIT ---
st.title("DESENVOLVIMENTO DE DASHBOARD ATRAVÉS DO PYTHON® PARA VISUALIZAÇÃO DOS DADOS E PREDIÇÃO DA PLUVIOSIDADE UTILIZANDO COLEÇÕES CHIRPS E ERA-5 LAND")

# Seletor interativo que exibe apenas as cidades do projeto
municipio_selecionado = municipios_filtrados["NM_MUN"]

# Filtra o shapefile local do município escolhido e converte para Earth Engine dinamicamente
municipio_gdf = municipios_filtrados[municipios_filtrados["NM_MUN"] == municipio_selecionado]
municipio_ee = geemap.geopandas_to_ee(municipio_gdf)


# --- MAPA COLORIDO INTEGRADO COM A PARAIBA COMPLETA ---
st.subheader("Mapa da Área de Estudo")

fig_mapa, ax = plt.subplots(figsize=(12, 6), facecolor='none')
ax.set_facecolor('none')

# Carrega a Paraíba completa direto da variável original para o fundo neutro
municipios_all = gpd.read_file(CAMINHO_SHP)
municipios_all.plot(
   ax=ax, 
   color="#AFAFAF", 
   edgecolor='#555555', 
   linewidth=0.4
)

# Plota por cima APENAS os 8 municípios do projeto
municipios_filtrados.plot(
    ax=ax, 
    column='NM_MUN', 
    cmap='tab10', 
    edgecolor='#444444', 
    linewidth=0.8,
    legend=True, 
    legend_kwds={'loc': 'upper left', 'bbox_to_anchor': (1.02, 1)}
)

# Borda para destacar qual deles está selecionado no selectbox
municipio_gdf.plot(
    ax=ax, 
    facecolor='none', 
    edgecolor='black', 
    linewidth=2.0, 
    linestyle='-'
)

ax.set_axis_off()
st.pyplot(fig_mapa)
st.markdown("---")

#botoes:
if st.button("Séries Temporais & Exportação",type="primary",use_container_width=True):
    st.switch_page("pages/baixar_visualizar.py")

if st.button("Predição Climática de Longo Prazo",type="primary",use_container_width=True):
    st.switch_page("pages/predicao.py")    


