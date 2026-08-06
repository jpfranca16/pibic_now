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

# Ativa a opção para restaurar o arquivo .shx automaticamente
os.environ["SHAPE_RESTORE_SHX"] = "YES"

# Inicialização do Earth Engine
# ee.Authenticate() # Descomente se for a primeira execução nesta máquina
ee.Initialize(project='infinite-unity-500221-h5')

# --- 2. LEITURA E FILTRAGEM DO SHAPEFILE ---
@st.cache_data
def carregar_municipios():
    municipios_all = gpd.read_file(
        "C:/Users/joaop/OneDrive/Documentos/GitHub/NOVO_PIBIC/PB_Municipios_2025/PB_Municipios_2025.shp"
    )
# Filtra apenas os 8 municípios selecionados do PIBIC
    nomes_pibic = ["Patos", "Água Branca", "Imaculada", "Juru", "Manaíra", "Princesa Isabel", "Tavares", "Teixeira"]
    return municipios_all[municipios_all["NM_MUN"].isin(nomes_pibic)].sort_values("NM_MUN")

municipios_filtrados = carregar_municipios()

# --- 3. INTERFACE STREAMLIT ---
st.title("Previsão de Precipitação com SARIMA (CHIRPS)")

# Seletor interativo que exibe apenas as cidades do projeto
municipio_selecionado = st.selectbox("Selecione o Município para Análise:", municipios_filtrados["NM_MUN"])

# Filtra o shapefile local do município escolhido e converte para Earth Engine dinamicamente
municipio_gdf = municipios_filtrados[municipios_filtrados["NM_MUN"] == municipio_selecionado]
municipio_ee = geemap.geopandas_to_ee(municipio_gdf)

# --- MAPA COLORIDO INTEGRADO COM A PARAIBA COMPLETA ---
st.subheader("Mapa da Área de Estudo (Localização no Estado da Paraíba)")

fig_mapa, ax = plt.subplots(figsize=(12, 6), facecolor='white')
ax.set_facecolor('white')

# Carrega a Paraíba completa direto da variável original para o fundo neutro
municipios_all = gpd.read_file(
    "C:/Users/joaop/OneDrive/Documentos/GitHub/NOVO_PIBIC/PB_Municipios_2025/PB_Municipios_2025.shp"
)
municipios_all.plot(
   ax=ax, 
   color='#f5f5f5', 
   edgecolor='#cccccc', 
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

# --- 4. CONFIGURAÇÃO DA SÉRIE TEMPORAL (CHIRPS) ---
st.write(f"Buscando dados históricos do CHIRPS para {municipio_selecionado}...")

chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
ano_inicio = 1980
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

st.write("Dados históricos extraídos com sucesso!")
st.dataframe(df.tail(12))

# --- 5. MODELO E PREDIÇÃO SARIMA ---
# --- 5. MODELO, VALIDAÇÃO E PREDIÇÃO SARIMA ---
st.subheader(f"Avaliação do Modelo e Predição para {municipio_selecionado}")

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
with st.expander("🔍 Clique para ver os Gráficos de Diagnóstico dos Resíduos (Análise de Viés)"):
    fig_diag = resultado_sarima.plot_diagnostics(figsize=(10, 6))
    plt.tight_layout()
    st.pyplot(fig_diag)

# D) Previsão dos próximos 24 meses
passos_previsao = 24
previsao = resultado_sarima.get_forecast(steps=passos_previsao)
df_previsto = previsao.summary_frame()

# Garante que a chuva prevista não assuma valores abaixo de zero por ruído do modelo
df_previsto['mean'] = df_previsto['mean'].clip(lower=0)
df_previsto['mean_ci_lower'] = df_previsto['mean_ci_lower'].clip(lower=0)



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
    name='Previsão SARIMA', 
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
    title=f'Previsão de Chuva Mensal com Modelo SARIMA - {municipio_selecionado}', 
    xaxis_title='Data', 
    yaxis_title='Precipitação (mm)', 
    template='plotly_white'
)

st.plotly_chart(fig)