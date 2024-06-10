import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide", page_title="Dashboard")

# Carregar os dados
tabela = pd.read_csv("dataset.csv", sep=";", header=0, encoding="iso-8859-1")
col_1, col_2 = st.columns(2)
col_3, col_4 = st.columns(2)
# st.title("Carrossel de Gráficos de Infectados e Óbitos")

csv = pd.read_csv('dados_tratados.csv', encoding='latin1', sep=',')
df = pd.read_csv("dataset.csv", sep=";", header=0, encoding="iso-8859-1")
csv['Data_Obito'] = pd.to_datetime(csv['Data_Obito'], format="%d/%m/%Y")
csv['Data_Inicio_Sintoma'] = pd.to_datetime(csv['Data_Inicio_Sintoma'], format="%d/%m/%Y")

obitosSim = csv[csv['Obito'] == 'SIM']
obitosSim['Ano_Obito'] = obitosSim['Data_Obito'].dt.year

#infectados por ano
infectadoPorAno = csv.groupby('Ano_Inicio_Sintoma').size().reset_index(name='Infectados')
infectPorAno = px.bar(infectadoPorAno, x='Ano_Inicio_Sintoma', y='Infectados', title='Infectados por ano')
col_1.plotly_chart(infectPorAno)

#mortes por região
mortesPorRegiaoEPorAno = obitosSim.groupby(['Regiao', 'Ano_Obito']).size().reset_index(name='Quantidade')
fig_mortes_regiao_ano = px.bar(mortesPorRegiaoEPorAno, x='Ano_Obito', y='Quantidade', color='Regiao', barmode='group', title='Mortes por Região e Ano')
col_2.plotly_chart(fig_mortes_regiao_ano)

# Gráficos de mortes por região
mortesPorRegiao = obitosSim.groupby('Regiao').size().reset_index(name='Quantidade')
fig_mortes_regiao = px.bar(mortesPorRegiao, x='Regiao', y='Quantidade', title='Mortes por Região')
col_3.plotly_chart(fig_mortes_regiao)

#Identificando as regiões epidemicas no surto de 2017 - 2018
registros_2017_2018 = df[df['ANO_IS'].isin([2017, 2018])]
total_2017_2018 = registros_2017_2018.groupby('UF_LPI').size().reset_index(name='Total_Registros')
total17E18 = px.bar(total_2017_2018, x='UF_LPI', y='Total_Registros', title='Quantidade de Registros Por Estado (2017 e 2018)')
col_4.plotly_chart(total17E18)

# Função para criar histograma de idade dos registros
def create_histogram(dados, titulo):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dados['IDADE'], nbinsx=10, marker_color='skyblue', opacity=0.75))
    fig.update_layout(
        title=titulo,
        xaxis_title='Idade',
        yaxis_title='Número de Registros',
        bargap=0.2
    )
    return fig

# Função para criar histograma de idade dos óbitos
def create_histogram_obitos(dados, titulo):
    obitos = dados[dados['OBITO'] == 'SIM']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=obitos['IDADE'], nbinsx=10, marker_color='skyblue', opacity=0.75))
    fig.update_layout(
        title=titulo,
        xaxis_title='Idade',
        yaxis_title='Número de Óbitos',
        bargap=0.2
    )
    return fig

# Função para criar gráfico de evolução de casos e óbitos
def create_evolution_graph(dados_2017, dados_2018, dados_2019, uf):
    casos_2017 = dados_2017.shape[0]
    obitos_2017 = dados_2017[dados_2017['OBITO'] == 'SIM'].shape[0]
    
    casos_2018 = dados_2018.shape[0]
    obitos_2018 = dados_2018[dados_2018['OBITO'] == 'SIM'].shape[0]
    
    casos_2019 = dados_2019.shape[0]
    obitos_2019 = dados_2019[dados_2019['OBITO'] == 'SIM'].shape[0]
    
    anos = [2017, 2018, 2019]
    casos = [casos_2017, casos_2018, casos_2019]
    obitos = [obitos_2017, obitos_2018, obitos_2019]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anos, y=casos, mode='lines+markers', name='Casos'))
    fig.add_trace(go.Scatter(x=anos, y=obitos, mode='lines+markers', name='Óbitos'))
    
    fig.update_layout(
        title=f'Evolução de Casos e Óbitos em {uf} (2017-2019)',
        xaxis_title='Ano',
        yaxis_title='Quantidade',
        legend_title='Legenda',
        xaxis=dict(tickmode='linear')
    )
    
    return fig

# Filtrar dados por estado e ano
def get_data_by_uf_and_year(tabela, uf, year):
    return tabela[(tabela['UF_LPI'] == uf) & (tabela['ANO_IS'] == year)]

# Criar todos os gráficos
ufs = ['SP', 'RJ', 'MG']
anos = [2017, 2018, 2019]

graficos = []

for uf in ufs:
    for ano in anos:
        dados = get_data_by_uf_and_year(tabela, uf, ano)
        graficos.append(create_histogram(dados, f'Histograma de Idade dos Registros em {uf} ({ano})'))
        graficos.append(create_histogram_obitos(dados, f'Histograma de Idade dos Óbitos em {uf} ({ano})'))
    
    dados_2017 = get_data_by_uf_and_year(tabela, uf, 2017)
    dados_2018 = get_data_by_uf_and_year(tabela, uf, 2018)
    dados_2019 = get_data_by_uf_and_year(tabela, uf, 2019)
    graficos.append(create_evolution_graph(dados_2017, dados_2018, dados_2019, uf))

# Exibir carrossel

current_graph_index = st.session_state.get('current_graph_index', 0)

if st.button('Próximo'):
    current_graph_index = (current_graph_index + 1) % len(graficos)
if st.button('Anterior'):
    current_graph_index = (current_graph_index - 1) % len(graficos)

st.session_state['current_graph_index'] = current_graph_index

st.plotly_chart(graficos[current_graph_index])

df_demografia = pd.read_excel("CIDADES-SURTO.xlsx")

# Filtrando os dados para os anos de surto (2017 e 2018)
surto = df[df['ANO_IS'].isin([2017, 2018])]

# Lista de cidades para analisar
cidades = ['MAIRIPORÃ', 'ANGRA DOS REIS', 'ATIBAIA', 'LADAINHA', 'JUIZ DE FORA',
           'MARIANA', "SANTA LEOPOLDINA", "NOVA LIMA", "VALENÇA", "NOVO CRUZEIRO",
           "SÃO PAULO", "NAZARÉ PAULISTA", "CARATINGA", "GUARULHOS", "BARÃO DE COCAIS",
           "POTÉ", "DOMINGOS MARTINS", "TERESÓPOLIS", "COLATINA", "NOVA FRIBURGO"]

# Processamento dos dados
resultados = []
resultados_60 = []

for cidade in cidades:
    cidade_dados = surto[surto['MUN_LPI'] == cidade]
    total_infectados = len(cidade_dados)
    total_obitos = len(cidade_dados[cidade_dados['OBITO'] == "SIM"])
    taxa_obitos_infectados = (total_obitos / total_infectados) * 100 if total_infectados > 0 else 0

    resultados.append({
        'Cidade': cidade,
        'Total Infectados': total_infectados,
        'Total Óbitos': total_obitos,
        'Taxa Óbitos/Infectados (%)': taxa_obitos_infectados
    })

    maiores_60 = cidade_dados[cidade_dados['IDADE'] > 60]
    total_infectados_maiores_60 = len(maiores_60)
    total_obitos_maiores_60 = len(maiores_60[maiores_60['OBITO'] == "SIM"])
    taxa_obitos_infectados_maiores_60 = (total_obitos_maiores_60 / total_infectados_maiores_60) * 100 if total_infectados_maiores_60 > 0 else 0

    resultados_60.append({
        'Cidade': cidade,
        'Total Infectados (60+)': total_infectados_maiores_60,
        'Total Óbitos (60+)': total_obitos_maiores_60,
        'Taxa Óbitos(60+)(%)': taxa_obitos_infectados_maiores_60
    })

resultados_df = pd.DataFrame(resultados)
resultados_df_60 = pd.DataFrame(resultados_60)

# Removendo colunas duplicadas de df2
resultados_df_60_unique = resultados_df_60[resultados_df_60.columns.difference(resultados_df.columns)]
df_final = pd.concat([resultados_df, resultados_df_60_unique], axis=1)

# Selecionando colunas relevantes da demografia e concatenando com os resultados
df_demografia = df_demografia[["HABITANTES", "DENSIDADE"]]
df_final = pd.concat([df_final, df_demografia], axis=1)

# Gráficos de dispersão e barras
fig_dispersao = px.scatter(df_final, x='HABITANTES', y='Total Infectados', title='Dispersão entre Total de Infectados e Habitantes')
fig_dispersao.update_layout(xaxis_title='Habitantes', yaxis_title='Total Infectados')

cidades_para_remover = ['MAIRIPORÃ', 'JUIZ DE FORA', 'SANTA LEOPOLDINA',
                        'SÃO PAULO' , 'GUARULHOS', 'TERESÓPOLIS', 'NOVA FRIBURGO',
                        'COLATINA', "LADAINHA"]

dados_filtrados = df_final[~df_final['Cidade'].isin(cidades_para_remover)]

fig_barras = px.bar(dados_filtrados, y='Cidade', x='Total Infectados', orientation='h', title='Total de Infectados por Cidade com População Informativa')
fig_barras.update_layout(xaxis_title='Total Infectados', yaxis_title='Cidades')

fig_dispersao_filtrados = px.scatter(dados_filtrados, x='HABITANTES', y='Total Infectados', title='Dispersão entre Total de Infectados e Habitantes com Dados Filtrados')
fig_dispersao_filtrados.update_layout(xaxis_title='Habitantes', yaxis_title='Total Infectados')

# Previsão

novas_cidades = pd.read_excel("CIDADES-RS.xlsx")
novas_cidades = novas_cidades.drop(columns=["FONTE", "PER_ATINGIDOS"])

X = df_final[['HABITANTES', 'DENSIDADE']]
y = df_final['Total Infectados']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

X_novas = novas_cidades[['HABITANTES', 'DENSIDADE']]
novas_cidades['CASOS_PREVISTOS'] = modelo.predict(X_novas)

valoresPrevistos = [28, 253, 30, 24, 103, 78]
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=novas_cidades['CIDADE'],
    y=valoresPrevistos,
    marker_color='skyblue',
    text=valoresPrevistos,
    textposition='outside'
))

fig_bar.update_layout(
    title='Casos Previstos por Cidade em um Período de Dois Anos',
    xaxis_title='Cidades',
    yaxis_title='Casos Previstos',
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)

# Criar carrossel de gráficos
graficos = [
    fig_dispersao,
    fig_barras,
    fig_dispersao_filtrados,
    fig_bar
]

# Exibir carrossel no Streamlit

current_graph_index = st.session_state.get('current_graph_index', 0)

if st.button('Next'):
    current_graph_index = (current_graph_index + 1) % len(graficos)
if st.button('Prev'):
    current_graph_index = (current_graph_index - 1) % len(graficos)

st.session_state['current_graph_index'] = current_graph_index

st.plotly_chart(graficos[current_graph_index])