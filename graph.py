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
        parent[vertice] = vertice
        rank[vertice] = 0

    minimum_spanning_tree = []
    edges = graph['edges']
    edges.sort(reverse=negative)
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1, parent) != find(vertice2, parent):
            union(vertice1, vertice2, parent, rank)
            minimum_spanning_tree.append(edge)
    return {'vertices': graph['vertices'], 'edges': minimum_spanning_tree}

def dfs(graph):
    """non-recursive depth-first search"""
    result = []
    white = set(graph['vertices'])
    stack = []
    n = 0
    for u in graph['vertices']:
        if u in white:
            n += 1
            stack.append(u)
            component = set()
            result.append(component)
            while stack:
                v = stack.pop()
                if v in white:
                    white.remove(v)
                    component.add(v)
                for e in graph['edges']:
                    _, left, right = e
                    if left == v:
                        w = right
                    elif right == v:
                        w = left
                    else:
                        continue
                    if w in white:
                        stack.append(w)
    return result, n

def _test():
    graph = {
        'vertices': ['A', 'B', 'C', 'D', 'E', 'F'],
        'edges': [
            (1, 'A', 'B'),
            (5, 'A', 'C'),
            (3, 'A', 'D'),
            (4, 'B', 'C'),
            (2, 'B', 'D'),
            (1, 'C', 'D'),
            ]
        }
    minimum_spanning_tree = set([
        (1, 'A', 'B'),
        (2, 'B', 'D'),
        (1, 'C', 'D'),
        ])
    assert set(kruskal(graph)) == minimum_spanning_tree

    dfs_result = ([set(['A', 'B', 'C', 'D']), set(['E']), set(['F'])], 3)
    assert dfs(graph) == dfs_result

if __name__ == '__main__':
    _test()