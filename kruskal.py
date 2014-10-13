def make_set(vertice, parent, rank):
    parent[vertice] = vertice
    rank[vertice] = 0

def find(vertice, parent):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice], parent)
    return parent[vertice]

def union(vertice1, vertice2, parent, rank):
    root1 = find(vertice1, parent)
    root2 = find(vertice2, parent)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
            if rank[root1] == rank[root2]: rank[root2] += 1

def kruskal(graph, negative=False):
    parent = dict()
    rank = dict()

    for vertice in graph['vertices']:
        make_set(vertice, parent, rank)

    minimum_spanning_tree = set()
    edges = graph['edges']
    edges.sort(reverse=negative)
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1, parent) != find(vertice2, parent):
            union(vertice1, vertice2, parent, rank)
            minimum_spanning_tree.add(edge)
    return minimum_spanning_tree

def _test():
    graph = {
        'vertices': ['A', 'B', 'C', 'D', 'E', 'F'],
        'edges': set([
            (1, 'A', 'B'),
            (5, 'A', 'C'),
            (3, 'A', 'D'),
            (4, 'B', 'C'),
            (2, 'B', 'D'),
            (1, 'C', 'D'),
            ])
        }
    minimum_spanning_tree = set([
        (1, 'A', 'B'),
        (2, 'B', 'D'),
        (1, 'C', 'D'),
        ])
    assert kruskal(graph) == minimum_spanning_tree

if __name__ == '__main__':
    _test()