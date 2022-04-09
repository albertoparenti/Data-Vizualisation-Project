# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# Imports
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
import plotly.express as px
import pandas as pd
import networkx as nx
import json

cyto.load_extra_layouts()

# Setup
random_state = 1

app = Dash(__name__)

# Read the dataset and process the data
cols_to_keep = [
  'SOURCE_SUBREDDIT',
  'TARGET_SUBREDDIT',
  # 'POST_ID',
  # 'TIMESTAMP',
  # 'LINK_SENTIMENT',
  'PROPERTIES',
  ]

subreddits_to_keep = [
  'bitcoin',
  'btc',
  'cryptocurrency',
  'crypto',
  'ethereum',
  'eth',
  'ethtrader',
  'litecoin',
  'monero',
  'ripple',
  'zcash',
  'cryptomarkets',
  'bitcoinmarkets',
  'altcoin',
  'icocrypto',
  'cardano',
  'bitcoinbeginners',
  'binance',
  'bitcoincash',
  'tether',
  'stratisplatform',
  'decred',
  'dash',
  'dogecoin',
  'litecointraders',
  'bitcoinuk',
  'nem',
  'stellar',
  'iota',
  'dashpay',
  'neo',
  'ethereumclassic',
  'eos',
  'lisk',
  'nanocurrency',
  'tron',
  'verge',
  'tronix',
  'coinbase',
  'defi',
  'omise_go',
  'satoshistreetBets',
  'bitcoin_com',
  'bitcoin_de',
  'bitcoin_fr',
  'bitcoin_it',
  'bitcoin_jp',
  'bitcoin_ru',
  'bitcoin_es',
  'bitcoin_uk',
  'bitcoin_us',
  'bitcoin_world',
  'bitcoin_ch',
  'bitcoin_ca',
  'bitcoin_nl',
  'bitcoinbeginners',
  'cryptotechnology',
  'blockchain',
  'crypto_currency',
  'crypto_coins',
  'crypto_currency_news',
  ]

df_title = (
  pd.read_csv('../data_title.tsv', sep='\t')
  .loc[:, cols_to_keep]
)
df_body = (
  pd.read_csv('../data_body.tsv', sep='\t')
  .loc[:, cols_to_keep]
)
df = (
  pd.concat([df_title, df_body], axis=0)
  .reset_index()
  # Keep only the selected subreddits
  .query('SOURCE_SUBREDDIT in @subreddits_to_keep or TARGET_SUBREDDIT in @subreddits_to_keep')
  # Extract the 'sentiment' property from the PROPERTIES column
  .assign(
    prop=lambda x: x.PROPERTIES.str.split(','),
    directed_sentiment=lambda x: x.prop.apply(lambda y: float(y[20])),    
  )
  .drop(columns=['PROPERTIES', 'prop'])
)

# Create a new column in df which is a sorted list of SOURCESUBREDDIT and TARGETSUBREDDIT
df['undirected_edge'] = df.apply(
  lambda x: str(sorted([x.SOURCE_SUBREDDIT, x.TARGET_SUBREDDIT])),
  axis=1,
)

mean_sentiment_df = (
  df
  .groupby('undirected_edge')
  .mean('directed_sentiment')
  .rename(columns={'directed_sentiment': 'sentiment'})
)

# Join mean_sentiment_df with df
df = pd.merge(
  df,
  mean_sentiment_df,
  on='undirected_edge',
  how='left',
)

# print(df.query('SOURCE_SUBREDDIT == "btc" and TARGET_SUBREDDIT == "bitcoin"'))

# Create a NetworkX graph with subreddits from the dataframe
G = nx.from_pandas_edgelist(
  df,
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
  df
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
  style={'width': '75%', 'height': '700px'},
  layout=default_layout,
  stylesheet=default_stylesheet,
  elements=nodes_and_edges
)

app.layout = html.Div([

  html.Div([
    html.H1('Exploring Crypto Reddit'),

    dcc.Markdown('''
        Reddit is a social media platform that allows users to post and comment on content. It is organized into communities called **subreddits**. Here we present an exploration into the connections between cryptocurrency-related subreddits.
        
        For more information on what Reddit it, check out [this video](https://www.youtube.com/watch?v=tlI022aUWQQ).
    ''')
  ]),

  html.Div([
    # Display the graph
    cyto_graph,
  ])
])

# Highlight on click callback
@app.callback(Output('cytoscape-graph-all-subreddits', 'stylesheet'),
              Input('cytoscape-graph-all-subreddits', 'tapNode'))
def generate_stylesheet(node):
    if not node:
        return default_stylesheet

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
            'background-color': '#B10DC9',
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
                    'background-color': '#f1c40f',
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
                    'line-color': "mapData(sentiment, -0.5, 1, #e74c3c, #2ecc71)",
                    'opacity': 0.9,
                    'z-index': 5000
                }
            })

        if edge['target'] == node['data']['id']:
            stylesheet.append({
                "selector": 'node[id = "{}"]'.format(edge['source']),
                "style": {
                    'background-color': '#f1c40f',
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
                    'line-color': "mapData(sentiment, -0.5, 1, #e74c3c, #2ecc71)",
                    'opacity': 0.9,
                    'z-index': 5000
                }
            })

    return stylesheet

if __name__ == '__main__':
    app.run_server(debug=True)