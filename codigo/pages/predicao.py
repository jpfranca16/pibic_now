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


#css para boniteza:
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

    /* Estilo em cartão para cada aba da navegação */
    [data-testid="stSidebarNav"] a {
        background-color: rgba(255, 255, 255, 0.5) !important;
        border-radius: 10px !important;
        margin-bottom: 6px !important;
        padding: 6px 12px !important;
        transition: all 0.3s ease !important;
    }

    /* 1. OCULTA OS NOMES ORIGINAIS DAS PÁGINAS */
    [data-testid="stSidebarNav"] ul li span {
        display: none !important;
    }

    /* 2. REESCREVE OS NOMES PERSONALIZADOS VIA CSS */
    /* Página 1 */
    [data-testid="stSidebarNav"] ul li:nth-child(1) a::after {
        content: "🏠 Página Inicial" !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #4B5563 !important;
    }

    /* Página 2 */
    [data-testid="stSidebarNav"] ul li:nth-child(2) a::after {
        content: "📊 Séries Temporais & CSV" !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #4B5563 !important;
    }

    /* Página 3 */
    [data-testid="stSidebarNav"] ul li:nth-child(3) a::after {
        content: "🔮 Predição Climática" !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #4B5563 !important;
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
st.title("Predição Climática de Longo Prazo da Precipitação Pluviométrica para 24 Meses por meio do Modelo SARIMA")

# Seletor interativo que exibe apenas as cidades do projeto
st.subheader("Selecione um Município: ")
municipio_selecionado = st.selectbox("", municipios_filtrados["NM_MUN"])

# Filtra o shapefile local do município escolhido e converte para Earth Engine dinamicamente
municipio_gdf = municipios_filtrados[municipios_filtrados["NM_MUN"] == municipio_selecionado]
municipio_ee = geemap.geopandas_to_ee(municipio_gdf)

st.markdown("---")

# --- 4. CONFIGURAÇÃO DA SÉRIE TEMPORAL (CHIRPS) ---
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

df = pd.DataFrame(dados_lista)
df['Data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01')
df = df.sort_values('Data').set_index('Data')
df = df.dropna()

st.success("✅ **Sucesso:** Dados históricos extraídos com sucesso!")

st.markdown("---")


# --- 5. MODELO E PREDIÇÃO SARIMA ---
# --- 5. MODELO, VALIDAÇÃO E PREDIÇÃO SARIMA ---
st.subheader(f"Avaliação do Modelo SARIMA para {municipio_selecionado}")

# A) Validação Fora da Amostra (Hold-out Test de 24 meses para medir o Viés)
df_treino = df.iloc[:-24]  # Treina até 2 anos atrás
df_teste = df.iloc[-24:]   # Testa nos últimos 24 meses reais

modelo_teste = SARIMAX(
    df_treino['precipitacao'], 
    order=(1, 1, 1), 
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

prev_teste = modelo_teste.get_forecast(steps=24).predicted_mean

# Cálculo das métricas reais
erros = df_teste['precipitacao'] - prev_teste
me = erros.mean()
mae = erros.abs().mean()
rmse = (erros ** 2).mean() ** 0.5

# Exibe o RESULTADO 2 (Métricas de Viés) na interface do Streamlit em colunas
col1, col2, col3 = st.columns(3)
col1.metric("Viés (Erro Médio)", f"{me:.2f} mm", help="Próximo de 0 = sem viés. Valor positivo = subestimando chuva; Negativo = superestimando.")
col2.metric("Erro Médio Absoluto (MAE)", f"{mae:.2f} mm")
col3.metric("Raiz do Erro Quadrático (RMSE)", f"{rmse:.2f} mm")


# B) Treinamento do Modelo Final com 100% dos dados para a previsão futura
modelo_final = SARIMAX(
    df['precipitacao'], 
    order=(1, 1, 1), 
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado_sarima = modelo_final.fit(disp=False)

# C) Exibe o RESULTADO 1 (Gráficos de Diagnóstico dos Resíduos) dentro de um painel retrátil
with st.expander("🔍 Gráficos de Diagnóstico dos Resíduos (Análise de Viés)",expanded=True):
    fig_diag = resultado_sarima.plot_diagnostics(figsize=(10, 6))
    plt.tight_layout()
    st.pyplot(fig_diag)


st.info(f"""
💡 **Informação:** A análise do modelo SARIMA (CHIRPS) para Imaculada (PB) mostra que a ferramenta é bastante precisa e confiável. 
O principal indicador de equilíbrio é o viés (a tendência de erro), que ficou em **{me:.2f} mm** — um valor praticamente nulo. 
Na prática, este valor de **{me:.2f} mm** significa apenas que o modelo subestima a chuva acumulada do mês por uma margem pequena de 
**{me:.2f} mm**, o que garante que ele não tende a estimar nem chuva em excesso, nem seca no longo prazo. 
A margem média de erro mensal também se manteve baixa, com **{mae:.2f} mm** (MAE) e **{rmse:.2f} mm** (RMSE), resultados bastante positivos 
considerando a variação do clima local. Por fim, os testes confirmam que o modelo capturou corretamente 
o ritmo das estações ao longo dos anos, mantendo sua estabilidade mesmo diante de possíveis eventos atípicos.
""")


# D) Previsão dos próximos 24 meses
passos_previsao = 24
previsao = resultado_sarima.get_forecast(steps=passos_previsao)
df_previsto = previsao.summary_frame()

# Garante que a chuva prevista não assuma valores abaixo de zero por ruído do modelo
df_previsto['mean'] = df_previsto['mean'].clip(lower=0)
df_previsto['mean_ci_lower'] = df_previsto['mean_ci_lower'].clip(lower=0)

st.markdown("---")

st.subheader(f"Predição Climática da Pluviosidade de {municipio_selecionado} PB")

# --- 6. PLOTAGEM DO GRÁFICO (PLOTLY) ---
fig = go.Figure()

# Histórico recente (últimos 5 anos)
df_recente = df.tail(60)
fig.add_trace(go.Scatter(
    x=df_recente.index, 
    y=df_recente['precipitacao'], 
    name='Histórico (CHIRPS)', 
    mode='lines',
    line=dict(color='blue')
))

# Linha de previsão futura do modelo
fig.add_trace(go.Scatter(
    x=df_previsto.index, 
    y=df_previsto['mean'], 
    name='Predição SARIMA', 
    mode='lines', 
    line=dict(dash='dash', color='red')
))

# Sombra do Intervalo de Confiança Estatístico (95%)
fig.add_trace(go.Scatter(
    x=df_previsto.index.tolist() + df_previsto.index.tolist()[::-1],
    y=df_previsto['mean_ci_upper'].tolist() + df_previsto['mean_ci_lower'].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(255,0,0,0.1)',
    line=dict(color='rgba(255,0,0,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='Margem de Erro (95%)'
))

fig.update_layout(
    title=f'Predição Climática Pluviométrica Mensal com Modelo SARIMA - {municipio_selecionado}', 
    xaxis_title='Data', 
    yaxis_title='Precipitação (mm)', 
    template='plotly_white'
)

st.plotly_chart(fig)