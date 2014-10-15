from graph import WeightedEdge

def make_graph(som, values):
    """make graph from an iterable with vertices and their values
       edge weight between vertices with values a and b
       is defined as min(a, b)
       """
    edges = set()

    for node, value in zip(som.nodes, values):
        if node.x < som.width - 1:
            node2 = som.node_at(node.x + 1, node.y)
            value2 = values[node2.n]
            edges.add(WeightedEdge(node, node2, min(value, value2)))
        if node.y < som.height - 1:
            node2 = som.node_at(node.x, node.y + 1)
            value2 = values[node2.n]
            edges.add(WeightedEdge(node, node2, min(value, value2)))

    return {'vertices': set(som.nodes), 'edges': edges}