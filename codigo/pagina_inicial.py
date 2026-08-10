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

#================================================================
#================================================================




                            #TUDO COMEÇA AQUI:


# Ativa a opção para restaurar o arquivo .shx automaticamente
os.environ["SHAPE_RESTORE_SHX"] = "YES"

# Inicialização do Earth Engine
# ee.Authenticate() # Descomente se for a primeira execução nesta máquina
ee.Initialize(project='infinite-unity-500221-h5')

# --- 2. LEITURA E FILTRAGEM DO SHAPEFILE ---
@st.cache_data
def carregar_municipios():
    municipios_all = gpd.read_file(
        "C:/Users/joaop/OneDrive/Documentos/GitHub/NOVO_PIBIC/pibic_now/codigo/PB_Municipios_2025/PB_Municipios_2025.shp"
    )
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
municipios_all = gpd.read_file(
    "C:/Users/joaop/OneDrive/Documentos/GitHub/NOVO_PIBIC/pibic_now/codigo/PB_Municipios_2025/PB_Municipios_2025.shp"
)
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


