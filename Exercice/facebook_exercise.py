"""
Exercise: Community detection in a social network

This script applies community detection methods to the Facebook social network dataset
from SNAP (Stanford Network Analysis Project) and evaluates the results.

Dataset: https://snap.stanford.edu/data/ego-Facebook.html
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
from infomap import Infomap
import gzip
import colorsys
from tqdm import tqdm

# ============================================================================
# STEP 1: Load the dataset
# ============================================================================

def load_facebook_network(filename='facebook_combined.txt.gz'):
    """
    Load the Facebook social network from a gzipped text file.
    
    Parameters:
    -----------
    filename : str
        Path to the gzipped edge list file
        
    Returns:
    --------
    G : networkx.Graph
        The social network graph
    """
    print(f"Loading network from {filename}...")
    
    # Load the graph from the gzipped file
    with gzip.open(filename, 'rt') as f:
        G = nx.read_edgelist(f, nodetype=int)
    
    print(f"Network loaded successfully!")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    return G


# ============================================================================
# STEP 2: Explore the network
# ============================================================================

def explore_network(G, visualize_subgraph=True, subgraph_size=50):
    """
    Explore the network by printing statistics and visualizing a subgraph.
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to explore
    visualize_subgraph : bool
        Whether to visualize a subgraph
    subgraph_size : int
        Number of nodes to include in the subgraph visualization
    """
    print("\n" + "="*60)
    print("Network Statistics")
    print("="*60)
    
    # Basic statistics
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.6f}")
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Max degree: {max(degrees)}")
    print(f"Min degree: {min(degrees)}")
    
    # Connected components
    num_components = nx.number_connected_components(G)
    print(f"Number of connected components: {num_components}")
    
    if num_components > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"Size of largest connected component: {len(largest_cc)}")
    
    # Clustering coefficient
    avg_clustering = nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg_clustering:.4f}")
    
    # Visualize a subgraph
    if visualize_subgraph and G.number_of_nodes() > subgraph_size:
        print(f"\nVisualizing subgraph with {subgraph_size} nodes...")
        
        # Get a connected subgraph
        # Start from the node with highest degree
        center_node = max(G.degree(), key=lambda x: x[1])[0]
        
        # Get neighbors up to a certain depth
        nodes = set([center_node])
        current_nodes = {center_node}
        
        while len(nodes) < subgraph_size:
            next_nodes = set()
            for node in current_nodes:
                neighbors = set(G.neighbors(node))
                next_nodes.update(neighbors)
            
            if not next_nodes - nodes:
                break
            
            nodes.update(list(next_nodes - nodes)[:subgraph_size - len(nodes)])
            current_nodes = next_nodes - nodes
        
        subgraph = G.subgraph(nodes)
        
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        nx.draw(subgraph, pos, node_size=100, node_color='lightblue', 
                edge_color='gray', alpha=0.7, with_labels=False)
        plt.title(f"Subgraph of {len(nodes)} nodes centered around node {center_node}")
        plt.tight_layout()
        plt.savefig('facebook_subgraph.png', dpi=150, bbox_inches='tight')
        print("Subgraph saved to 'facebook_subgraph.png'")
        plt.close()


# ============================================================================
# STEP 3: Apply community detection methods
# ============================================================================

def get_N_colors(N=5):
    """Generate N visually distinct colors."""
    HSV_tuples = [(x * 1.0 / N, 0.7, 0.9) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out


def labels_from_communities(communities, num_nodes):
    """
    Convert communities (list of sets/lists) to labels vector.
    
    Parameters:
    -----------
    communities : list
        List of communities (each community is a set/list of nodes)
    num_nodes : int
        Total number of nodes in the graph
        
    Returns:
    --------
    labels : list
        List of community labels for each node
    """
    # Initialize labels with -1 (unassigned)
    labels = [-1] * num_nodes
    
    for i, comm in enumerate(communities):
        for node in comm:
            labels[node] = i
    
    return labels


def apply_community_detection(G):
    """
    Apply multiple community detection methods to the graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze
        
    Returns:
    --------
    results : dict
        Dictionary containing communities detected by each method
    """
    print("\n" + "="*60)
    print("Applying Community Detection Methods")
    print("="*60)
    
    results = {}
    
    # Method 1: Modularity Optimization (Greedy)
    print("\n1. Modularity Optimization (Greedy Algorithm)...")
    communities_modularity = nx.algorithms.community.greedy_modularity_communities(G)
    communities_modularity = [frozenset(c) for c in communities_modularity]
    results['modularity'] = communities_modularity
    print(f"   Number of communities: {len(communities_modularity)}")
    print(f"   Community sizes: {sorted([len(c) for c in communities_modularity], reverse=True)[:10]}")
    
    # Method 2: Label Propagation
    print("\n2. Label Propagation...")
    communities_lp = nx.algorithms.community.label_propagation_communities(G)
    communities_lp = [frozenset(c) for c in communities_lp]
    results['label_propagation'] = communities_lp
    print(f"   Number of communities: {len(communities_lp)}")
    print(f"   Community sizes: {sorted([len(c) for c in communities_lp], reverse=True)[:10]}")
    
    # Method 3: Girvan-Newman (get first split - 2 communities)
    print("\n3. Girvan-Newman (first level)...")
    print("   Note: This is VERY slow for large networks. Progress bar shows edge removal.")
    print(f"   Total edges to process: {G.number_of_edges()}")
    
    # Create a progress bar that tracks iterations
    with tqdm(total=G.number_of_edges(), desc="   Removing edges", unit="edge") as pbar:
        def girvan_newman_with_progress(G):
            """Wrapper for Girvan-Newman that updates progress bar."""
            last_edge_count = G.number_of_edges()
            for communities in nx.algorithms.community.girvan_newman(G):
                current_edge_count = G.number_of_edges()
                edges_removed = last_edge_count - current_edge_count
                pbar.update(edges_removed)
                last_edge_count = current_edge_count
                yield communities
        
        communities_gn_generator = girvan_newman_with_progress(G)
        communities_gn = next(communities_gn_generator)
    
    communities_gn = [frozenset(c) for c in communities_gn]
    results['girvan_newman'] = communities_gn
    print(f"   Number of communities: {len(communities_gn)}")
    print(f"   Community sizes: {sorted([len(c) for c in communities_gn], reverse=True)}")
    
    # Method 4: Infomap
    print("\n4. Infomap...")
    im = Infomap()
    # Add edges to Infomap
    for edge in G.edges():
        im.add_link(edge[0], edge[1])
    
    # Run Infomap
    im.run()
    
    # Extract communities
    communities_dict = {}
    for node in im.tree:
        if node.is_leaf:
            communities_dict[node.node_id] = node.module_id
    
    # Convert to list of sets
    module_ids = set(communities_dict.values())
    communities_im = []
    for mid in module_ids:
        comm = frozenset([node for node, mod in communities_dict.items() if mod == mid])
        communities_im.append(comm)
    
    results['infomap'] = communities_im
    print(f"   Number of communities: {len(communities_im)}")
    print(f"   Community sizes: {sorted([len(c) for c in communities_im], reverse=True)[:10]}")
    
    return results


# ============================================================================
# STEP 4: Evaluate the results
# ============================================================================

def evaluate_communities(G, results):
    """
    Evaluate community detection results using various metrics.
    
    Parameters:
    -----------
    G : networkx.Graph
        The network
    results : dict
        Dictionary of community detection results
    """
    print("\n" + "="*60)
    print("Evaluation Metrics")
    print("="*60)
    
    # Modularity scores
    print("\nModularity Scores:")
    print("-" * 60)
    for method, communities in results.items():
        mod = nx.algorithms.community.modularity(G, communities)
        print(f"{method:25s}: {mod:.4f}")
    
    # Coverage and Performance
    print("\nCoverage and Performance:")
    print("-" * 60)
    for method, communities in results.items():
        coverage, performance = nx.algorithms.community.partition_quality(G, communities)
        print(f"{method:25s}: coverage={coverage:.4f}, performance={performance:.4f}")
    
    # Normalized Mutual Information (NMI) between methods
    print("\nNormalized Mutual Information (NMI) between methods:")
    print("-" * 60)
    
    # Convert to labels
    num_nodes = G.number_of_nodes()
    labels_dict = {}
    for method, communities in results.items():
        labels_dict[method] = labels_from_communities(communities, num_nodes)
    
    # Compare all pairs
    methods = list(results.keys())
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            nmi = normalized_mutual_info_score(labels_dict[method1], labels_dict[method2])
            print(f"{method1:25s} vs {method2:25s}: {nmi:.4f}")


def visualize_community_sizes(results):
    """
    Visualize the distribution of community sizes for each method.
    
    Parameters:
    -----------
    results : dict
        Dictionary of community detection results
    """
    print("\nGenerating community size distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (method, communities) in enumerate(results.items()):
        sizes = sorted([len(c) for c in communities], reverse=True)
        
        axes[idx].bar(range(len(sizes)), sizes)
        axes[idx].set_xlabel('Community rank')
        axes[idx].set_ylabel('Community size')
        axes[idx].set_title(f'{method.replace("_", " ").title()}\n({len(communities)} communities)')
        axes[idx].set_yscale('log')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('community_size_distributions.png', dpi=150, bbox_inches='tight')
    print("Community size distributions saved to 'community_size_distributions.png'")
    plt.close()


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main function to run the community detection exercise."""
    
    print("="*60)
    print("Community Detection in Facebook Social Network")
    print("="*60)
    
    # Step 1: Load the dataset
    G = load_facebook_network('facebook_combined.txt.gz')
    
    # Step 2: Explore the network
    explore_network(G, visualize_subgraph=True, subgraph_size=100)
    
    # Step 3: Apply community detection methods
    results = apply_community_detection(G)
    
    # Step 4: Evaluate the results
    evaluate_communities(G, results)
    
    # Visualize community sizes
    visualize_community_sizes(results)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nSummary of findings:")
    print("-" * 60)
    
    # Find best method by modularity
    best_method = max(results.items(), 
                     key=lambda x: nx.algorithms.community.modularity(G, x[1]))
    
    print(f"\nBest method by modularity: {best_method[0]}")
    print(f"Modularity score: {nx.algorithms.community.modularity(G, best_method[1]):.4f}")
    print(f"Number of communities: {len(best_method[1])}")
    
    print("\nStrengths and limitations:")
    print("-" * 60)
    print("Modularity Optimization (Greedy):")
    print("  + Fast and efficient for large networks")
    print("  + Good modularity scores")
    print("  - May miss small communities (resolution limit)")
    print("  - Deterministic (always same result)")
    
    print("\nLabel Propagation:")
    print("  + Very fast, linear complexity")
    print("  + Can find overlapping structures")
    print("  - Non-deterministic (different results each run)")
    print("  - May produce unbalanced partitions")
    
    print("\nGirvan-Newman:")
    print("  + Hierarchical structure")
    print("  + Good theoretical foundation")
    print("  - Very slow for large networks (O(m²n))")
    print("  - Need to choose level in hierarchy")
    
    print("\nInfomap:")
    print("  + Based on information theory")
    print("  + Finds communities with high information flow")
    print("  + Good for directed networks")
    print("  - Can be sensitive to network structure")
    print("  - May produce many small communities")


if __name__ == "__main__":
    main()
