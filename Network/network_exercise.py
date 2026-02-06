"""
Network Visualization Exercise
Creates a network with at least 50 nodes and displays at least 5 properties:
- Node properties: Degree Centrality, Betweenness Centrality
- Edge properties: Weight, Edge Betweenness, Community membership
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Set random seed for reproducibility
np.random.seed(42)

# Create a network using Barabasi-Albert model (scale-free network)
n_nodes = 60  # At least 50 nodes
m_edges = 2   # Number of edges to attach from a new node to existing nodes
G = nx.barabasi_albert_graph(n_nodes, m_edges)

print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# ============================================================================
# CALCULATE NODE PROPERTIES
# ============================================================================

# Node Property 1: Degree Centrality
degree_centrality = nx.degree_centrality(G)
nx.set_node_attributes(G, degree_centrality, 'degree_centrality')

# Node Property 2: Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G)
nx.set_node_attributes(G, betweenness_centrality, 'betweenness_centrality')

# Node Property 3: Clustering Coefficient
clustering_coef = nx.clustering(G)
nx.set_node_attributes(G, clustering_coef, 'clustering_coefficient')

print("\nNode Properties Calculated:")
print(f"- Degree Centrality (avg: {np.mean(list(degree_centrality.values())):.3f})")
print(f"- Betweenness Centrality (avg: {np.mean(list(betweenness_centrality.values())):.3f})")
print(f"- Clustering Coefficient (avg: {np.mean(list(clustering_coef.values())):.3f})")

# ============================================================================
# CALCULATE EDGE PROPERTIES
# ============================================================================

# Edge Property 1: Random weights
edge_weights = {edge: np.random.uniform(0.5, 2.0) for edge in G.edges()}
nx.set_edge_attributes(G, edge_weights, 'weight')

# Edge Property 2: Edge Betweenness Centrality
edge_betweenness = nx.edge_betweenness_centrality(G)
nx.set_edge_attributes(G, edge_betweenness, 'edge_betweenness')

# Edge Property 3: Community detection for edge membership
communities = nx.community.louvain_communities(G, seed=42)
node_to_community = {}
for idx, community in enumerate(communities):
    for node in community:
        node_to_community[node] = idx

nx.set_node_attributes(G, node_to_community, 'community')

# Mark edges that connect different communities
edge_type = {}
for edge in G.edges():
    if node_to_community[edge[0]] == node_to_community[edge[1]]:
        edge_type[edge] = 'intra-community'
    else:
        edge_type[edge] = 'inter-community'

nx.set_edge_attributes(G, edge_type, 'edge_type')

print("\nEdge Properties Calculated:")
print(f"- Weights (avg: {np.mean(list(edge_weights.values())):.3f})")
print(f"- Edge Betweenness (avg: {np.mean(list(edge_betweenness.values())):.3f})")
print(f"- Edge Types: {len([e for e in edge_type.values() if e == 'intra-community'])} intra-community, "
      f"{len([e for e in edge_type.values() if e == 'inter-community'])} inter-community")
print(f"\nNumber of communities detected: {len(communities)}")

# ============================================================================
# EXPORT FOR CYTOSCAPE
# ============================================================================

# Export nodes with attributes to CSV
nodes_data = []
for node in G.nodes():
    nodes_data.append({
        'id': node,
        'degree_centrality': degree_centrality[node],
        'betweenness_centrality': betweenness_centrality[node],
        'clustering_coefficient': clustering_coef[node],
        'community': node_to_community[node],
        'degree': G.degree(node)
    })

nodes_df = pd.DataFrame(nodes_data)
nodes_df.to_csv('network_nodes.csv', index=False)
print("\nNodes exported to: network_nodes.csv")

# Export edges with attributes to CSV
edges_data = []
for edge in G.edges():
    edges_data.append({
        'source': edge[0],
        'target': edge[1],
        'weight': edge_weights[edge],
        'edge_betweenness': edge_betweenness[edge],
        'edge_type': edge_type[edge]
    })

edges_df = pd.DataFrame(edges_data)
edges_df.to_csv('network_edges.csv', index=False)
print("Edges exported to: network_edges.csv")

# Export as GraphML (can be directly imported into Cytoscape)
nx.write_graphml(G, 'network.graphml')
print("Network exported to: network.graphml (for Cytoscape)")

# ============================================================================
# VISUALIZE WITH NETWORKX
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Network Visualization with 5+ Properties', fontsize=16, fontweight='bold')

# Use spring layout for all visualizations
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Plot 1: Degree Centrality
ax = axes[0, 0]
node_colors = [degree_centrality[node] for node in G.nodes()]
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, 
                               cmap='YlOrRd', ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
ax.set_title('Node Property 1: Degree Centrality', fontweight='bold')
ax.axis('off')
plt.colorbar(nodes, ax=ax, label='Degree Centrality')

# Plot 2: Betweenness Centrality
ax = axes[0, 1]
node_colors = [betweenness_centrality[node] for node in G.nodes()]
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, 
                               cmap='viridis', ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
ax.set_title('Node Property 2: Betweenness Centrality', fontweight='bold')
ax.axis('off')
plt.colorbar(nodes, ax=ax, label='Betweenness Centrality')

# Plot 3: Clustering Coefficient
ax = axes[0, 2]
node_colors = [clustering_coef[node] for node in G.nodes()]
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, 
                               cmap='plasma', ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
ax.set_title('Node Property 3: Clustering Coefficient', fontweight='bold')
ax.axis('off')
plt.colorbar(nodes, ax=ax, label='Clustering Coefficient')

# Plot 4: Edge Weights
ax = axes[1, 0]
edge_weights_list = [edge_weights[edge] for edge in G.edges()]
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=200, ax=ax)
edges = nx.draw_networkx_edges(G, pos, width=[w*2 for w in edge_weights_list], 
                               edge_color=edge_weights_list, edge_cmap=plt.cm.Blues, ax=ax)
ax.set_title('Edge Property 1: Weights', fontweight='bold')
ax.axis('off')
sm = ScalarMappable(cmap=plt.cm.Blues, norm=Normalize(vmin=min(edge_weights_list), vmax=max(edge_weights_list)))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Edge Weight')

# Plot 5: Edge Betweenness
ax = axes[1, 1]
edge_bet_list = [edge_betweenness[edge] for edge in G.edges()]
nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=200, ax=ax)
edges = nx.draw_networkx_edges(G, pos, width=2, 
                               edge_color=edge_bet_list, edge_cmap=plt.cm.Reds, ax=ax)
ax.set_title('Edge Property 2: Edge Betweenness', fontweight='bold')
ax.axis('off')
sm = ScalarMappable(cmap=plt.cm.Reds, norm=Normalize(vmin=min(edge_bet_list), vmax=max(edge_bet_list)))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Edge Betweenness')

# Plot 6: Communities and Edge Types
ax = axes[1, 2]
node_colors = [node_to_community[node] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, 
                       cmap='Set3', ax=ax)
# Draw intra-community edges in gray
intra_edges = [e for e in G.edges() if edge_type[e] == 'intra-community']
nx.draw_networkx_edges(G, pos, edgelist=intra_edges, edge_color='gray', 
                      width=1, alpha=0.5, ax=ax)
# Draw inter-community edges in red
inter_edges = [e for e in G.edges() if edge_type[e] == 'inter-community']
nx.draw_networkx_edges(G, pos, edgelist=inter_edges, edge_color='red', 
                      width=2, alpha=0.8, ax=ax)
ax.set_title('Edge Property 3: Communities & Edge Types', fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: network_visualization.png")
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*60)
print("NETWORK SUMMARY STATISTICS")
print("="*60)
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Network density: {nx.density(G):.4f}")
print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
print(f"Number of communities: {len(communities)}")
print(f"\nTop 5 nodes by Degree Centrality:")
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
for node, value in top_degree:
    print(f"  Node {node}: {value:.4f}")
print(f"\nTop 5 nodes by Betweenness Centrality:")
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
for node, value in top_betweenness:
    print(f"  Node {node}: {value:.4f}")
print("="*60)
