# Cytoscape Import Instructions

## Quick Start - Import GraphML (Recommended)

1. **Open Cytoscape**
2. Go to: `File → Import → Network from File...`
3. Select: `network.graphml`
4. Click **OK** - All properties are automatically imported!

## Alternative - Import from CSV Files

### Step 1: Import the Network
1. Go to: `File → Import → Network from File...`
2. Select: `network_edges.csv`
3. In the import dialog:
   - Set **source** column as "Source Node"
   - Set **target** column as "Target Node"
   - Click **OK**

### Step 2: Import Node Attributes
1. Go to: `File → Import → Table from File...`
2. Select: `network_nodes.csv`
3. In the import dialog:
   - Select "Import data as: **Node Table Columns**"
   - Set **id** as the key column
   - Click **OK**

### Step 3: Import Edge Attributes (if needed)
Edge attributes should already be imported from the edge CSV file.

## Visualization Suggestions

### Node Styling

1. **Node Size based on Degree Centrality:**
   - Go to **Style** panel
   - Click on **Size** property
   - Set Column: `degree_centrality`
   - Set Mapping Type: `Continuous Mapping`
   - Adjust min/max sizes (e.g., 20-80)

2. **Node Color based on Community:**
   - Click on **Fill Color** property
   - Set Column: `community`
   - Set Mapping Type: `Discrete Mapping`
   - Assign different colors to each community

3. **Alternative - Node Color by Betweenness:**
   - Click on **Fill Color** property
   - Set Column: `betweenness_centrality`
   - Set Mapping Type: `Continuous Mapping`
   - Choose a color gradient (e.g., white to red)

### Edge Styling

1. **Edge Width based on Weight:**
   - Go to **Style** panel → **Edge** tab
   - Click on **Width** property
   - Set Column: `weight`
   - Set Mapping Type: `Continuous Mapping`
   - Adjust min/max widths (e.g., 1-5)

2. **Edge Color based on Type:**
   - Click on **Stroke Color** property
   - Set Column: `edge_type`
   - Set Mapping Type: `Discrete Mapping`
   - Assign colors:
     - `intra-community`: Gray (#808080)
     - `inter-community`: Red (#FF0000)

3. **Edge Transparency based on Betweenness:**
   - Click on **Transparency** property
   - Set Column: `edge_betweenness`
   - Set Mapping Type: `Continuous Mapping`
   - Range: 100 (low) to 255 (high)

### Recommended Layouts

Try these layouts in order:
1. **Prefuse Force Directed Layout** (best for communities)
   - `Layout → Prefuse Force Directed Layout`
   - Good default settings

2. **Edge-weighted Spring Embedded**
   - `Layout → Edge-weighted Spring Embedded`
   - Uses edge weights

3. **yFiles Organic Layout** (if you have yFiles installed)
   - Produces very clean visualizations

## Network Properties Summary

- **Nodes**: 60
- **Edges**: 116
- **Communities**: 7
- **Average Clustering**: ~0.18

### Node Attributes Available:
- `degree_centrality` - How well connected (0-1)
- `betweenness_centrality` - Bridge importance (0-1)
- `clustering_coefficient` - Local clustering (0-1)
- `community` - Community ID (0-6)
- `degree` - Number of connections

### Edge Attributes Available:
- `weight` - Edge weight (0.5-2.0)
- `edge_betweenness` - Edge importance (0-1)
- `edge_type` - intra-community or inter-community

## Tips for Analysis in Cytoscape

1. **Find hub nodes**: Sort nodes by `degree_centrality` in the **Table Panel**
2. **Identify bridges**: Sort nodes by `betweenness_centrality`
3. **Analyze communities**: Use `Select → Nodes → By Column Value` to select all nodes in a community
4. **Export visualizations**: `File → Export → Network to Image...`

## Troubleshooting

- If properties don't show up, check the **Table Panel** to verify they were imported
- If layout looks messy, try: `Layout → Apply Preferred Layout` or adjust layout parameters
- For better visualization, hide node labels: **Style** panel → **Label** → set transparency to 0
