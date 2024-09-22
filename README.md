# Identifying-Influencers
Work with social graphs to compute centrality measures in parallel using MPI. Implemented algorithms to compute degree centrality and betweenness centrality of nodes in the graph, and then combine these measures to identify the top k most influential nodes in the graph.

# Problem Statement
You are given a directed graph representing social connections, where nodes are individuals
and directed edges represent relationships. Your task is to compute two types of centrality
measures:
1. Degree Centrality: This measure indicates how connected a node is within the graph.
For a node u, the degree centrality is the sum of its incoming and outgoing edges.
2. Betweenness Centrality: This measure indicates the importance of a node in terms
of its ability to connect other nodes via the shortest paths. For a node v, it is calculated
as the sum of the fraction of all-pairs shortest paths that pass through v.
After computing these centrality measures, you will normalize the values and then combine
them to compute a combined centrality score for each node. Finally, you will use this combined
score to find the top k most influential nodes in the graph.
