import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import dash_cytoscape as cyto
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import networkx as nx
import plotly.express as px
from list_features import feature_properties, subreddits_to_keep, cols_to_keep
import base64

def processing_features(data):
    '''
    This function processing some features will use in dash
    '''
    data['YEAR'] = data['TIMESTAMP'].apply(lambda x: x.split('-')[0])
    data['LINK_SENTIMENT'] = data['LINK_SENTIMENT'].apply(lambda x: 'Neutral or Positive' if x==1 else 'Negative')


    data = (data.query('SOURCE_SUBREDDIT in @subreddits_to_keep or TARGET_SUBREDDIT in @subreddits_to_keep')
    .assign(
    prop=lambda x: x.PROPERTIES.str.split(','),
    directed_sentiment=lambda x: x.prop.apply(lambda y: float(y[20])),    
  ))


    data['undirected_edge'] = data.apply(
        lambda x: str(sorted([x.SOURCE_SUBREDDIT, x.TARGET_SUBREDDIT])),
        axis=1)

    mean_sentiment_df = (
      data.groupby('undirected_edge')
      .mean('directed_sentiment')
      .rename(columns={'directed_sentiment': 'sentiment'})
    )
    print('hhhhhhh')
    print(mean_sentiment_df['sentiment'])

    # Join mean_sentiment_df with data
    data = pd.merge(
      data,
      mean_sentiment_df,
      on='undirected_edge',
      how='left',
    )
    
    return data

def expand_data(data):
    '''
    This function expand data in more columns
    '''
    data_expand = data.PROPERTIES.str.split(",",expand=True)
    data_expand.columns = feature_properties
    data_features = pd.concat([data.drop('PROPERTIES', axis=1), data_expand], axis=1)
    data_features['directed_sentiment'] = data_features['Compound sentiment calculated by VADER']
    
    return data_features

def filter_cases_show(network_df, SUBREDDIT=False, SENTIMENT_LINK=False):
    '''
    This function prepare data will use in dash, input is the filter to apply in the dash and 
    the output is data filtered and options of filter.
    '''
    if SUBREDDIT:
        network_df = network_df.query('SOURCE_SUBREDDIT in @SUBREDDIT or TARGET_SUBREDDIT in @SUBREDDIT')

    if SENTIMENT_LINK:
        network_df = network_df.query('LINK_SENTIMENT in @SENTIMENT_LINK')

    return network_df


#### Read the dataset and processing features will show in dash
df_title = pd.read_csv('data_title.tsv',sep='\t').loc[:, cols_to_keep]
df_body = pd.read_csv('data_body.tsv',sep='\t').loc[:, cols_to_keep]
network_df = pd.concat([df_title, df_body], axis=0).reset_index()
network_df = processing_features(network_df)
network_df = expand_data(network_df)

# Create a NetworkX graph with subreddits from the dataframe
G = nx.from_pandas_edgelist(
  network_df,
  source='SOURCE_SUBREDDIT',
  target='TARGET_SUBREDDIT',
  edge_attr=['sentiment'],
)

# Add number of edges to each node
for n in G.nodes():
  G.nodes[n]['num_edges'] = len(G.edges(n))

# List with number of edges for each node and get maximum number of edges in a node
edges_per_node = nx.get_node_attributes(G, 'num_edges')
max_edges = max(edges_per_node.values())

# Edge weights are the number of times those two nodes were linked
edge_weights_df = (
  network_df
  .groupby(
    ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'],
  )
  .size()
  .sort_values(ascending=False)
)
max_edge_weight = edge_weights_df.max()

# Iterate through edges and add edge weight to the edge attribute
for e in G.edges():
  try:
    G.edges[e]['edge_weight'] = edge_weights_df.loc[e[0], e[1]]
  except:
    G.edges[e]['edge_weight'] = 0

# Convert NetworkX graph into Cytoscape graph
cyto_graph = nx.cytoscape_data(G)
cyto_nodes = [{'data': {'id': n['data']['id'], 'label': n['data']['id'], 'num_edges': n['data']['num_edges']}} for n in cyto_graph['elements']['nodes'] if n['data']['id'] in subreddits_to_keep]
cyto_edges = [{'data': {'source': e['data']['source'], 'target': e['data']['target'], 'edge_weight': e['data']['edge_weight'], 'sentiment': e['data']['sentiment']}} for e in cyto_graph['elements']['edges'] if e['data']['source'] in subreddits_to_keep and e['data']['target'] in subreddits_to_keep]

default_stylesheet = [
    {
      'selector': 'node',
      'style': {
        'width': 'mapData(num_edges, 0, 1300, 5, 50)',
        'height': 'mapData(num_edges, 0, 1300, 5, 50)',
        'label': 'data(label)',
        'text-valign': 'center',
        'font-size': '6px',
      }
    },
    {
      'selector': 'edge',
      'style': {
        'width': 'mapData(edge_weight, 0, 300, 0.2, 7)',
        'height': 'mapData(edge_weight, 0, 300, 0.2, 7)',
      }
    },
]

default_layout = {'name': 'cose'}

nodes_and_edges = [*cyto_nodes, *cyto_edges]

cyto_graph = cyto.Cytoscape(
  id='cytoscape-graph-all-subreddits',
  style={'width': '100%', 'height': '700px'},
  layout=default_layout,
  stylesheet=default_stylesheet,
  elements=nodes_and_edges
)

reddit_options = [dict(label=reddit, value=reddit) for reddit in network_df['SOURCE_SUBREDDIT'].unique()]
dropdown_reddit = dcc.Dropdown(
        id='reddit_drop',
        options=reddit_options,
        style = {"background-color":"rgb(220,220,220)", "color": "rgb(0,0,0)"},
        value=[],
        multi=True
)

sentiment_options = [dict(label=sentim, value=sentim) for sentim in network_df['LINK_SENTIMENT'].unique()]
dropdown_sentim = dcc.Dropdown(
        id='sentim_drop',
        options=sentiment_options,
        style = {"background-color":"rgb(220,220,220)"},
        value=[],
        multi=True
)

def networkGraph(node, SUBREDDIT, SENTIMENT_LINK):
    network_df_filter = filter_cases_show(network_df, SUBREDDIT, SENTIMENT_LINK)

    ############ Bar chart #####################################
    fig2 = px.histogram(network_df, x="YEAR", y="LINK_SENTIMENT",
             color='LINK_SENTIMENT', barmode='group',
             histfunc='count',
             height=400,color_discrete_sequence=['#BDB76B','#3498db'], template="simple_white").update_layout({'plot_bgcolor': 'rgb(220,220,220)','paper_bgcolor': 'rgb(220,220,220)'})
    fig2.update_xaxes(showline=True, linewidth=1, linecolor='grey', mirror=True)
    fig2.update_yaxes(showline=True, linewidth=1, linecolor='grey', mirror=True)

    ############ Table Information post #####################################
    
    table_cols = network_df_filter if SUBREDDIT or SENTIMENT_LINK else network_df

    table_info = table_cols[['Number of words', 'Number of unique works', 'Number of unique stopwords',
                            'Fraction of stopwords', 'Number of sentences']].astype(float)

    table_info_mean = table_info.describe().loc[['mean']].round(3).T.reset_index().rename(columns={"index": "Information about post", "mean": "Mean the Reddit filter"})
    dash_table_info = table_info_mean.to_dict('records')
    columns_table = [{"name": i, "id": i} for i in table_info_mean.columns]

    ############ Table sentiment #####################################

    table_sent = pd.DataFrame(table_cols[['LINK_SENTIMENT']].value_counts()).reset_index().rename(columns={"LINK_SENTIMENT": "Sentiment about link", 0: "Quantity post"})
    dash_table_sent = table_sent.to_dict('records')
    columns_sent = [{"name": i, "id": i} for i in table_sent.columns]
    
    # Create a NetworkX graph with subreddits from the dataframe
    if not node:
        return default_stylesheet, dash_table_info, columns_table, dash_table_sent, columns_sent, fig2
        
    stylesheet = [
      {
        "selector": 'node',
        'style': {
            'opacity': '0.3',
            'width': 'mapData(num_edges, 0, 1300, 5, 50)',
            'height': 'mapData(num_edges, 0, 1300, 5, 50)',
            'label': 'data(label)',
            'text-valign': 'center',
            'font-size': '6px',
        }
      },
      {
        'selector': 'edge',
        'style': {
            'opacity': '0.3',
            'width': 'mapData(edge_weight, 0, 300, 0.2, 7)',
            'height': 'mapData(edge_weight, 0, 300, 0.2, 7)',
        }
      },
      {
        "selector": 'node[id = "{}"]'.format(node['data']['id']),
        "style": {
            'background-color': '#3498db',
            "opacity": '1',
            'width': 'mapData(num_edges, 0, 1300, 5, 50)',
            'height': 'mapData(num_edges, 0, 1300, 5, 50)',
            'label': 'data(label)',
            'text-valign': 'center',
            'font-size': '6px',
            'z-index': 9999
        }
      }
    ]

    for edge in node['edgesData']:
        if edge['source'] == node['data']['id']:
            stylesheet.append({
                "selector": 'node[id = "{}"]'.format(edge['target']),
                "style": {
                    'background-color': '#BDB76B',
                    'opacity': 0.9,
                    'width': 'mapData(num_edges, 0, 1300, 5, 50)',
                    'height': 'mapData(num_edges, 0, 1300, 5, 50)',
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'font-size': '6px',
                    'z-index': 5000
                }
            })
            stylesheet.append({
                "selector": 'edge[id= "{}"]'.format(edge['id']),
                "style": {
                    'width': 'mapData(edge_weight, 0, 300, 0.2, 7)',
                    'height': 'mapData(edge_weight, 0, 300, 0.2, 7)',
                    'line-color': "mapData(sentiment, -0.3, 0.3, #e74c3c, #2ecc71)",
                    'opacity': 0.9,
                    'z-index': 5000
                }
            })

        if edge['target'] == node['data']['id']:
            stylesheet.append({
                "selector": 'node[id = "{}"]'.format(edge['source']),
                "style": {
                    'background-color': '#BDB76B',
                    'opacity': 0.9,
                    'width': 'mapData(num_edges, 0, 1300, 5, 50)',
                    'height': 'mapData(num_edges, 0, 1300, 5, 50)',
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'font-size': '6px',
                    'z-index': 5000
                }
            })
            stylesheet.append({
                "selector": 'edge[id= "{}"]'.format(edge['id']),
                "style": {
                    'width': 'mapData(edge_weight, 0, 300, 0.2, 7)',
                    'height': 'mapData(edge_weight, 0, 300, 0.2, 7)',
                    'line-color': "mapData(sentiment, -0.3, 0.3, #e74c3c, #2ecc71)",
                    'opacity': 0.9,
                    'z-index': 5000
                }
            })

    

    return stylesheet, dash_table_info, columns_table, dash_table_sent, columns_sent, fig2


# Dash app
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__)
app.title = 'Dash Networkx'

encoded_image = base64.b64encode(open('image_reddit.png', 'rb').read())
encoded_image_legend = base64.b64encode(open('graph-legend.jpg', 'rb').read())

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1('Exploring Crypto Reddit'),
            dcc.Markdown('''
            Reddit is a social media platform that allows users to post and comment on content. It is organized into communities called **subreddits**. Here we present an exploration into the connections between cryptocurrency-related subreddits.

            The top part of the dash board shows invidual subreddit-level data, and the bottom part shows aggregated data on the subreddits in question.

            For more information on what Reddit it, check out [this video](https://www.youtube.com/watch?v=tlI022aUWQQ).
        ''')
        ], id='text', style={'width': '80%'}),
    html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),style={'height':'200px', 'width':'100px%'}),
        ], id='imag', style={'width': '40%'}),
    ], id='1st row', style={'display': 'flex'}, className='pretty_box'),
    html.Div([
        html.Div([
                # html.Br(),
                html.H2('Connections between subreddits'),
                dcc.Markdown('Connections are given by mentions of one subreddit by another.'),
                html.Img(src='data:image/jpg;base64,{}'.format(encoded_image_legend.decode()),style={'height':'70px', 'width':'100px%'}),
                cyto_graph,
            ], id='graph', style={'width': '50%'}, className='pretty_box'),
        html.Div([
            html.H2('Choose subreddit and sentiment'),
    
            html.Label('Cripto subreddit'),
            dropdown_reddit,
            html.Br(),

            html.Label('Sentiment'),
            dropdown_sentim,
            html.Br(),

            html.Button('Submit', id='button')
            ],  id='Iteraction', style={'width': '20%'}, className='pretty_box'),
        html.Div([
            html.Div([
                html.H3('Informations about subreddit'),
                html.Br(),
                dash_table.DataTable(id='table_info', 
                                     style_table={
                                         'borderRadius':'7px 7px',
                                         'border': '1px solid grey',
                                         'overflow': 'hidden',},
                                     style_header={ 
                                         'border': '1px solid grey', 
                                         'overflow': 'hidden', 
                                         'background-color': 'rgb(220,220,220)',
                                         'textAlign': 'left'}, 
                                     style_data={
                                         'background-color': 'rgb(220,220,220)',
                                         'border': '1px solid grey', 
                                         'overflow': 'hidden',
                                         'textAlign': 'left'}),
            ]),
            html.Div([
                html.Br(),
                dash_table.DataTable(id='table_sent',
                                    style_table={
                                         'borderRadius':'7px 7px',
                                         'border': '1px solid grey',
                                         'overflow': 'hidden',},
                                     style_header={ 
                                         'border': '1px solid grey',
                                         'overflow': 'hidden', 
                                         'background-color': 'rgb(220,220,220)',
                                         'textAlign': 'left'}, 
                                     style_data={
                                         'background-color': 'rgb(220,220,220)',
                                         'border': '1px solid grey',
                                         'overflow': 'hidden',
                                         'textAlign': 'left'}),
            ]),
        ], id='tables', style={'width': '30%'}, className='pretty_box'),
    ], id='2nd row', style={'display': 'flex'}),
        html.Div([
        html.H4('Sentiment distribution in all subreddits'),
        html.Br(),
        dcc.Graph(id='hist_graph'),
        ],id='3nd row', className='pretty_box')
])


@app.callback(
    Output('cytoscape-graph-all-subreddits', 'stylesheet'),
    Output("table_info", "data"),
    Output("table_info", "columns"),
    Output("table_sent", "data"),
    Output("table_sent", "columns"),
    Output("hist_graph", "figure"),
    [Input('cytoscape-graph-all-subreddits', 'tapNode'),
     Input('reddit_drop', 'value'),
     Input('sentim_drop', 'value')],
)


def update_output(node, SUBREDDIT, SENTIMENT_LINK):
    return networkGraph(node, SUBREDDIT, SENTIMENT_LINK)


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8080, dev_tools_hot_reload=False)
    #app.run_server(debug=True)