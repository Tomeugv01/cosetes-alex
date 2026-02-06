# Network Visualization Exercise

This project creates and visualizes a network with at least 50 nodes, displaying at least 5 properties using NetworkX and exports files for Cytoscape visualization.

## Properties Displayed

### Node Properties (3):
1. **Degree Centrality** - Measures how connected each node is
2. **Betweenness Centrality** - Measures how important a node is for connecting other nodes
3. **Clustering Coefficient** - Measures how much neighbors of a node are connected to each other

### Edge Properties (3):
1. **Weight** - Random weights assigned to each edge
2. **Edge Betweenness** - Measures importance of edges in network connectivity
3. **Edge Type** - Classifies edges as intra-community or inter-community based on community detection

## Files Generated

- `network_visualization.png` - NetworkX visualization showing all 5+ properties
- `network_nodes.csv` - Node list with all attributes (for Cytoscape)
- `network_edges.csv` - Edge list with all attributes (for Cytoscape)
- `network.graphml` - Complete network in GraphML format (direct import to Cytoscape)

## Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script:
```bash
python network_exercise.py
```

This will:
- Generate a scale-free network with 60 nodes
- Calculate all node and edge properties
- Create visualizations
- Export files for Cytoscape

## Importing into Cytoscape

### Method 1: Using GraphML file (Recommended)
1. Open Cytoscape
2. Go to `File -> Import -> Network from File`
3. Select `network.graphml`
4. All properties will be automatically imported

### Method 2: Using CSV files
1. Open Cytoscape
2. Go to `File -> Import -> Network from File`
3. Select `network_edges.csv`
4. Map columns: source → Source Node, target → Target Node
5. Import node attributes: `File -> Import -> Table from File`
6. Select `network_nodes.csv` and map to nodes

## Network Details

- **Model**: Barabási-Albert (scale-free network)
- **Nodes**: 60
- **Average degree**: ~4 edges per node
- **Community detection**: Louvain algorithm

## Visualization Tips for Cytoscape

1. **Node Size**: Map to degree_centrality or betweenness_centrality
2. **Node Color**: Map to community
3. **Edge Width**: Map to weight
4. **Edge Color**: Map to edge_type or edge_betweenness
5. **Layout**: Try Force-Directed or Prefuse Force Directed layouts

## Author

Created for Network Analysis exercise - Spanish High School Network study
