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
    /* Papel de parede de fundo */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1743046813915-94cf6d5e6942?q=80&w=1528&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-attachment: fixed;
        background-size: cover;
    }

    /* Fundo da Sidebar em Vidro Fosco */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.70) !important;
        backdrop-filter: blur(12px) !important;
    }

    /* Aumenta o tamanho e deixa em negrito os nomes das páginas */
    [data-testid="stSidebarNav"] span {
        font-size: 18px !important;
        font-weight: 700 !important;
    }

    /* Estilo em cartão para cada aba da navegação */
    [data-testid="stSidebarNav"] a {
        background-color: rgba(255, 255, 255, 0.5) !important;
        border-radius: 10px !important;
        margin-bottom: 6px !important;
        padding: 6px 12px !important;
        transition: all 0.3s ease !important;
    }

    /* Cores individuais para cada página na lista */
    [data-testid="stSidebarNav"] ul li:nth-child(1) span {
        color: #0284C7 !important; /* Azul para a 1ª página */
    }
    [data-testid="stSidebarNav"] ul li:nth-child(2) span {
        color: #059669 !important; /* Verde para a 2ª página */
    }
    [data-testid="stSidebarNav"] ul li:nth-child(3) span {
        color: #D97706 !important; /* Laranja para a 3ª página */
    }

    /* Destaque para a página selecionada atualmente */
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-left: 5px solid #0284C7 !important;
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
# Substitua a linha do ee.Initialize por esta função:
def inicializar_earth_engine():
    try:
        if "gcp_service_account" in st.secrets:
            key_dict = dict(st.secrets["gcp_service_account"])
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], 
                key_data=key_dict["private_key"]
            )
            ee.Initialize(credentials=credentials, project='infinite-unity-500221-h5')
        else:
            ee.Initialize(project='infinite-unity-500221-h5')
    except Exception as e:
        st.error(f"Erro ao inicializar o Earth Engine: {e}")

inicializar_earth_engine()

# --- 2. LEITURA E FILTRAGEM DO SHAPEFILE ---

PASTA_PAGES = os.path.dirname(__file__)
PASTA_CODIGO = os.path.dirname(PASTA_PAGES)

# Aponta para a pasta do shapefile que está dentro de 'codigo'
CAMINHO_SHP = os.path.join(PASTA_CODIGO, "PB_Municipios_2025", "PB_Municipios_2025.shp")


@st.cache_data
def carregar_municipios():
    municipios_all = gpd.read_file(CAMINHO_SHP)
# Filtra apenas os 8 municípios selecionados do PIBIC
    nomes_pibic = ["Patos", "Água Branca", "Imaculada", "Juru", "Manaíra", "Princesa Isabel", "Tavares", "Teixeira"]
    return municipios_all[municipios_all["NM_MUN"].isin(nomes_pibic)].sort_values("NM_MUN")

municipios_filtrados = carregar_municipios()

# --- 3. INTERFACE STREAMLIT ---
st.title("Visualização das Séries Temporais e Exportação da Base de Dados Climáticos (CSV)")

# Seletor interativo que exibe apenas as cidades do projeto
st.subheader("Selecione um Município: ")
municipio_selecionado = st.selectbox("", municipios_filtrados["NM_MUN"])

# Filtra o shapefile local do município escolhido e converte para Earth Engine dinamicamente
municipio_gdf = municipios_filtrados[municipios_filtrados["NM_MUN"] == municipio_selecionado]
municipio_ee = geemap.geopandas_to_ee(municipio_gdf)

st.markdown("---")

#==============================================================================================================
#==============================================================================================================
#==============================================================================================================
#==============================================================================================================

# --- 4. CONFIGURAÇÃO DA SÉRIE TEMPORAL (CHIRPS) ---
st.subheader("Dados de Precipitação (CHIRPS):")

st.write(f"**Buscando dados históricos do CHIRPS para {municipio_selecionado}...**")


chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
ano_inicio = 1982
ano_fim = 2025
anos = ee.List.sequence(ano_inicio, ano_fim)
meses = ee.List.sequence(1, 12)

def calcular_mensal(ano):
    def por_mes(mes):
        data_inicio = ee.Date.fromYMD(ano, mes, 1)
        data_fim = data_inicio.advance(1, 'month')
        colecao_mes = chirps.filterDate(data_inicio, data_fim)

        chuva_mes = ee.Image(
            ee.Algorithms.If(
            colecao_mes.size().gt(0),
            colecao_mes.sum().rename(['precipitation']),
            ee.Image.constant(0).rename(['precipitation'])
            )
        )
 
        media_municipio = chuva_mes.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=municipio_ee.geometry(), 
            scale=5000,
            maxPixels=1e13
        )
 
        valor_chuva = ee.List([media_municipio.get('precipitation'), 0.0]).reduce(ee.Reducer.firstNonNull())
 
        return ee.Feature(None, {
            'ano': ano,
            'mes': mes,
            'precipitacao': valor_chuva
        })
    return meses.map(por_mes)

# Mapeamento em nuvem
recursos_mensais = anos.map(calcular_mensal).flatten()
colecao_final = ee.FeatureCollection(recursos_mensais)

# Conversão para DataFrame
features = colecao_final.getInfo()['features']
dados_lista = [f['properties'] for f in features]

df_precipitacao = pd.DataFrame(dados_lista)
df_precipitacao['Data'] = pd.to_datetime(df_precipitacao['ano'].astype(str) + '-' + df_precipitacao['mes'].astype(str) + '-01')
df_precipitacao = df_precipitacao.sort_values('Data').set_index('Data')
df_precipitacao = df_precipitacao.dropna()

st.success("✅ **Sucesso:** Dados históricos extraídos com sucesso!")

st.table(df_precipitacao.tail(6))

#convertendo para csv:
csv_df_precipitacao = df_precipitacao.to_csv(index=False).encode('utf-8')

#botao para download:
st.download_button(
    label='Baixar todos os dados em CSV',
    type="primary",
    use_container_width=True,
    data=csv_df_precipitacao,
    file_name='dados_precipitacao.csv',
    mime='text/csv',
)

#------------------------------------------------------
st.subheader("Séries Temporais da Precipitação (CHIRPS):")

st.info("💡 **Informação:** Para melhor visualizar, ao invés de usar os dados mensais foi calculado a média e a soma por ano.")


# Criação do gráfico com Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_precipitacao.index, df_precipitacao["precipitacao"], color="blue", linewidth=1.5)
ax.set_title("Série Histórica da Precipitação", fontsize=14, fontweight="bold")
ax.set_xlabel("Data", fontsize=12)
ax.set_ylabel("Precipitação (mm)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Exibe o gráfico na tela do Streamlit

#para melhor visualizar, fazendo a serie temporal por ano:
df_precipitacao_anual = df_precipitacao.resample('YE').sum() 


# 1. Cria a figura do Plotly
fig = go.Figure()

# 2. Adiciona a linha do tempo
fig.add_trace(
    go.Scatter(
        x=df_precipitacao_anual.index,
        y=df_precipitacao_anual["precipitacao"],
        mode="lines",
        name="Precipitação (CHIRPS)",
        line=dict(color="#1f77b4", width=2),
    )
)

# 3. Personaliza o layout e títulos
fig.update_layout(
    title="Série Histórica da Precipitação",
    xaxis_title="Data",
    yaxis_title="Precipitação (mm)",
    hovermode="x unified",  # Destaca os valores exatos ao passar o mouse
    template="plotly_white",
    height=650, 
    margin=dict(l=10, r=10, t=50, b=10),
)

# 4. Renderiza no Streamlit
st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.markdown("---")

#==============================================================================================================
#==============================================================================================================
#==============================================================================================================
#==============================================================================================================

st.subheader("Dados de Temperatura Média (ERA5-Land):")

# --- CONFIGURAÇÃO DA SÉRIE TEMPORAL (ERA5-LAND) ---
st.write(f"**Buscando dados históricos de temperatura do ERA5-Land para {municipio_selecionado}...**")

# Coleção agregada diária do ERA5-Land
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

ano_inicio = 1982
ano_fim = 2025
anos = ee.List.sequence(ano_inicio, ano_fim)
meses = ee.List.sequence(1, 12)

def calcular_temperatura_mensal(ano):
    def por_mes(mes):
        data_inicio = ee.Date.fromYMD(ano, mes, 1)
        data_fim = data_inicio.advance(1, 'month')
        colecao_mes = era5.filterDate(data_inicio, data_fim)

        # Média mensal da banda 'temperature_2m' convertida de Kelvin para Celsius (- 273.15)
        temp_mes = ee.Image(
            ee.Algorithms.If(
                colecao_mes.size().gt(0),
                colecao_mes.select('temperature_2m').mean().subtract(273.15).rename(['temperatura']),
                ee.Image.constant(0).rename(['temperatura'])
            )
        )
 
        media_municipio = temp_mes.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=municipio_ee.geometry(), 
            scale=11132,  # Resolução espacial do ERA5-Land (aprox. 11 km)
            maxPixels=1e13
        )
 
        valor_temp = ee.List([media_municipio.get('temperatura'), 0.0]).reduce(ee.Reducer.firstNonNull())
 
        return ee.Feature(None, {
            'ano': ano,
            'mes': mes,
            'temperatura': valor_temp
        })
    return meses.map(por_mes)

# Processamento e DataFrame
recursos_mensais = anos.map(calcular_temperatura_mensal).flatten()
colecao_final = ee.FeatureCollection(recursos_mensais)

features = colecao_final.getInfo()['features']
dados_lista = [f['properties'] for f in features]

df_temp = pd.DataFrame(dados_lista)
df_temp['Data'] = pd.to_datetime(df_temp['ano'].astype(str) + '-' + df_temp['mes'].astype(str) + '-01')
df_temp = df_temp.sort_values('Data').set_index('Data')
df_temp = df_temp.dropna()

st.success("✅ **Sucesso:** Dados históricos extraídos com sucesso!")

st.table(df_temp.tail(6))

#convertendo para csv:
csv_df_temp = df_temp.to_csv(index=False).encode('utf-8')

#botao para download:
st.download_button(
    label='Baixar todos os dados em CSV',
    type="primary",
    use_container_width=True,
    data=csv_df_temp,
    file_name='dados_temperatura.csv',
    mime='text/csv',
)

# --- GRÁFICO DA TEMPERATURA ---
st.subheader("Séries Temporais da Temperatura Média (ERA5-Land):")

st.info("💡 **Informação:** Para melhor visualizar, ao invés de usar os dados mensais foi calculado a média e a soma por ano.")


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_temp.index, df_temp["temperatura"], color="red", linewidth=1.5)
ax.set_title("Série Histórica da Temperatura Média Média 2m", fontsize=14, fontweight="bold")
ax.set_xlabel("Data", fontsize=12)
ax.set_ylabel("Temperatura (°C)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

#para melhor visualizar, fazendo a serie temporal por ano:
df_temp_anual = df_temp.resample('YE').mean()

# Exibe o gráfico na tela do Streamlit

# 1. Cria a figura do Plotly
fig = go.Figure()

# 2. Adiciona a linha do tempo
fig.add_trace(
    go.Scatter(
        x=df_temp_anual.index,
        y=df_temp_anual["temperatura"],
        mode="lines",
        name="Temperatura (ERA5-Land)",
        line=dict(color="#e41414", width=2),
    )
)

# 3. Personaliza o layout e títulos
fig.update_layout(
    title="Série Histórica da Precipitação",
    xaxis_title="Data",
    yaxis_title="Temperatura",
    hovermode="x unified",  # Destaca os valores exatos ao passar o mouse
    template="plotly_white",
    height=650, 
    margin=dict(l=10, r=10, t=50, b=10),
)

# 4. Renderiza no Streamlit
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("---")

#==============================================================================================================
#==============================================================================================================
#==============================================================================================================
#==============================================================================================================

# --- GRÁFICO COMBINADO (PRECIPITAÇÃO + TEMPERATURA) ---
st.subheader("Série Temporal Combinada: Precipitação e Temperatura")

st.info("💡 **Informação:** Para melhor visualizar, ao invés de usar os dados mensais foi calculado a média e a soma por ano.")


# 1. Cria a figura com suporte a 2 eixos Y
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 2. Adiciona a Precipitação (Eixo Y Esquerdo)
fig.add_trace(
    go.Scatter(
        x=df_precipitacao_anual.index,
        y=df_precipitacao_anual["precipitacao"],
        name="Precipitação (CHIRPS)",
        line=dict(color="#1f77b4", width=2),
    ),
    secondary_y=False,
)

# 3. Adiciona a Temperatura (Eixo Y Direito)
fig.add_trace(
    go.Scatter(
        x=df_temp_anual.index,
        y=df_temp_anual["temperatura"],
        name="Temperatura (ERA5-Land)",
        line=dict(color="#d62728", width=2.5),
    ),
    secondary_y=True,
)

# 4. Configurações dos eixos e títulos
fig.update_layout(
    title_text=f"Tendência Climática - {municipio_selecionado}",
    hovermode="x unified",
    template="plotly_white",
)

fig.update_yaxes(
    title_text="Precipitação Anual (mm)",
    title_font_color="#1f77b4",
    secondary_y=False,
)
fig.update_yaxes(
    title_text="Temperatura Média Anual (°C)",
    title_font_color="#d62728",
    secondary_y=True,
)

# 5. Renderiza no Streamlit
st.plotly_chart(fig, use_container_width=True)





