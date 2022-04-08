import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import networkx as nx
import plotly.express as px
from feature_properties import name_columns

def processing_features(data):
    '''
    This function processing some features will use in dash
    '''
    data['YEAR'] = data['TIMESTAMP'].apply(lambda x: x.split('-')[0])
    data['LINK_SENTIMENT'] = data['LINK_SENTIMENT'].apply(lambda x: 'Neutral or Positive' if x==1 else 'Negative')
    return data

def filter_data(data):
    '''
    This function will filter cases will show in dashboard because the data is so big and we can't process all them
    '''
    data_2013 = data.loc[(data['YEAR'] == '2013')][:10]
    data_2014 = data.loc[(data['YEAR'] == '2014')][:10]
    data_filter = pd.concat([data_2013, data_2014], axis=0).reset_index()
    return data_filter

def expand_data(data):
    '''
    This function expand data in more columns
    '''
    data_expand = data.PROPERTIES.str.split(",",expand=True)
    data_expand.columns = name_columns
    data_features = pd.concat([data.drop('PROPERTIES', axis=1), data_expand], axis=1)
    
    return data_features

def filter_cases_show(network_df, YEAR=False, SUBREDDIT=False, SENTIMENT=False):
    '''
    This function prepare data will use in dash, input is the filter to apply in the dash and 
    the output is data filtered and options of filter.
    '''
    if YEAR:
        network_df = network_df.loc[(network_df['YEAR'] == YEAR[0])]
    if SUBREDDIT:
        network_df = network_df.loc[(network_df['SOURCE_SUBREDDIT'] == SUBREDDIT[0])]
    if SENTIMENT:
        network_df = network_df.loc[(network_df['LINK_SENTIMENT'] == SENTIMENT[0])]
        
    return network_df


#### Read the dataset and processing features will show in dash
df_title = pd.read_csv('data_title.tsv',sep='\t')
df_body = pd.read_csv('data_body.tsv',sep='\t')
network_df = pd.concat([df_title, df_body], axis=0).reset_index()
network_df = processing_features(network_df)
network_df = filter_data(network_df)
network_df = expand_data(network_df)

year_options = [dict(label=year, value=year) for year in network_df['YEAR'].unique()]
dropdown_year = dcc.Dropdown(
        id='year_drop',
        options=year_options,
        value=['2013'],
        multi=True
)

reddit_options = [dict(label=reddit, value=reddit) for reddit in network_df['SOURCE_SUBREDDIT'].unique()]
dropdown_reddit = dcc.Dropdown(
        id='reddit_drop',
        options=reddit_options,
        value=[],
        multi=True
)

sentiment_options = [dict(label=sentim, value=sentim) for sentim in network_df['LINK_SENTIMENT'].unique()]
dropdown_sentim = dcc.Dropdown(
        id='sentim_drop',
        options=sentiment_options,
        value=[],
        multi=True
)

def networkGraph(YEAR, SUBREDDIT, SENTIMENT):
    network_df_filter = filter_cases_show(network_df, YEAR, SUBREDDIT, SENTIMENT)
    source = list(network_df_filter['SOURCE_SUBREDDIT'].unique())
    target = list(network_df_filter['TARGET_SUBREDDIT'].unique())
    node_list = set(source+target)

    # Add nodes
    G = nx.Graph()
    for i in node_list:
        G.add_node(i)
    for i,j in network_df_filter.iterrows():
        G.add_edges_from([(j["SOURCE_SUBREDDIT"],j["TARGET_SUBREDDIT"])])
    
    #Simulate the position nodes, that will be close if there are conections each others
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    for n, p in pos.items():
        G.node[n]['pos'] = p

    #Scatter plot edge_trace
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    #Scatter plot nodes
    node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='RdBu',
                reversescale=True,
                color=[],
                size=15,
                colorbar=dict(
                    thickness=10,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=0)))
    #Put information axis x and y
    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    #Others information in graph
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])

    #Generate graph
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Reddit Hyperlink Social Network',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    ############ Bar chart #####################################
    fig2 = px.histogram(network_df, x="YEAR", y="LINK_SENTIMENT",
             color='LINK_SENTIMENT', barmode='group',
             histfunc='count',
             height=400)

    ############ Table Information post #####################################
    table_cols = network_df_filter if YEAR or SUBREDDIT or SENTIMENT else network_df
    
    table_info = table_cols[['Number of words', 'Number of unique works', 'Number of unique stopwords',
                            'Fraction of stopwords', 'Number of sentences', 'Positive sentiment calculated by VADER',
                            'Negative sentiment calculated by VADER','Compound sentiment calculated by VADER']].astype(float)

    table_info_mean = table_info.describe().loc[['mean']].T.reset_index().rename(columns={"index": "Information about post", "mean": "Mean the Reddit filter"})
    dash_table_info = table_info_mean.to_dict('records')
    columns_table = [{"name": i, "id": i} for i in table_info_mean.columns]


    ############ Table sentiment #####################################

    table_sent = pd.DataFrame(table_cols[['LINK_SENTIMENT']].value_counts()).reset_index().rename(columns={"LINK_SENTIMENT": "Sentiment about link", 0: "Quantity post"})
    dash_table_sent = table_sent.to_dict('records')
    columns_sent = [{"name": i, "id": i} for i in table_sent.columns]

    ############ Bar chart LIWC #####################################
    cols_liwc = list(network_df_filter.columns[network_df_filter.columns.str.contains('LIWC')])
    

    table_liwc = network_df_filter[cols_liwc].astype(float)
    table_liwc = table_liwc.describe().loc[['mean']].T.reset_index().rename(columns={"index": "category", "mean": "Mean percentual"})

    table_liwc["category"] = table_liwc["category"].replace("LIWC_", "", regex=True)

    fig3 = px.bar(table_liwc, x='category', y='Mean percentual')

    return fig, dash_table_info, columns_table, dash_table_sent, columns_sent, fig3, fig2


# Dash app
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__)
app.title = 'Dash Networkx'


app.layout = html.Div([

    html.Div([
        html.H1('Reddit Hyperlink Social Network'), 
        ], id='1st row', className='pretty_box'),
    html.Div([
        html.Div([
            html.Label('Choose your YEAR'),
            dropdown_year,
            html.Br(),
    
            html.Label('Choose your SUBREDDIT'),
            dropdown_reddit,
            html.Br(),

            html.Label('Choose SENTIMENT'),
            dropdown_sentim,
            html.Br(),

            html.Button('Submit', id='button')
            ],  id='Iteraction', style={'width': '30%'}, className='pretty_box'),
            html.Div([
                html.Br(),
                dcc.Graph(id='my-graph'),
                ], id='graph', style={'width': '70%'}, className='pretty_box')
        ], id='2nd row', style={'display': 'flex'}),

    html.Div([
        html.Div([
            html.Div([
                html.Br(),
                dash_table.DataTable(id='table_info'),
            ]),
            html.Div([
                html.Br(),
                dash_table.DataTable(id='table_sent'),
            ]),
        ], id='tables', style={'width': '30%'}, className='pretty_box'),
        html.Div([
            html.Br(),
            dcc.Graph(id='bar_graph'),
        ], id='bar', style={'width': '70%'}, className='pretty_box'),
    ], id='3th row', style={'display': 'flex'}),
        html.Br(),
        dcc.Graph(id='hist_graph'),
    ]
)


@app.callback(
    Output('my-graph', 'figure'),
    Output("table_info", "data"),
    Output("table_info", "columns"),
    Output("table_sent", "data"),
    Output("table_sent", "columns"),
    Output("bar_graph", "figure"),
    Output("hist_graph", "figure"),
    [Input('year_drop', 'value'),
     Input('reddit_drop', 'value'),
     Input('sentim_drop', 'value')],
)
def update_output(YEAR, SUBREDDIT, SENTIMENT):
    return networkGraph(YEAR, SUBREDDIT, SENTIMENT)


if __name__ == '__main__':
    #app.run_server(host='127.0.0.1', port=8050, dev_tools_hot_reload=False)
    app.run_server(debug=True)