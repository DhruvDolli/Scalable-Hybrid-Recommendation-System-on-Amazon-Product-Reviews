# # import torch
# # print(torch.version.cuda)

import pandas as pd
import networkx as nx
from itertools import combinations
import dgl
import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


# Load cleaned data
df = pd.read_parquet(r"path\combined_cleaned.parquet")

# Drop missing usernames or product IDs
df = df.dropna(subset=["reviews.username", "id"])

# Initialize undirected graph
G = nx.Graph()

# Group by user: create edges between co-reviewed products
for user, group in df.groupby("reviews.username"):
    products = group["id"].unique()
    if len(products) > 1:
        edges = combinations(products, 2)
        G.add_edges_from(edges)

print("✅ Graph created with:")                                                 #✅ Graph created with: Nodes: 63 Edges: 906
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# Degree Centrality
dc = nx.degree_centrality(G)

# Top 5 most central products
top_dc = sorted(dc.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top central products:", top_dc)

# Convert NetworkX to DGL
g = dgl.from_networkx(G)

# Assign one-hot node features for now
g.ndata['feat'] = torch.eye(g.number_of_nodes())

print("✅ DGL graph ready with", g.number_of_nodes(), "nodes")                  #✅ DGL graph ready with 63 nodes


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, aggregator_type='mean')

    def forward(self, g, x):
        x = F.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x

# Instantiate
model = GraphSAGE(in_feats=g.ndata['feat'].shape[1], hidden_feats=64, out_feats=32)


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = self.conv2(g, x)
        return x

g.ndata['feat'] = torch.eye(g.num_nodes())


# Initialize model
model = GraphSAGE(in_feats=g.ndata['feat'].shape[1], hidden_feats=64, out_feats=32)

# Forward pass
with torch.no_grad():
    embeddings = model(g, g.ndata['feat'])

print("✅ Embeddings shape:", embeddings.shape)


# Map node IDs to actual product IDs
product_ids = list(G.nodes)  # From original NetworkX graph

embedding_df = pd.DataFrame(embeddings.numpy(), index=product_ids)
embedding_df.reset_index(inplace=True)
embedding_df.columns = ['product_id'] + [f'emb_{i}' for i in range(32)]

# Save to file
embedding_df.to_csv(r"path\product_gnn_embeddings.csv", index=False)

# import numpy as np
# print(np.__version__)
