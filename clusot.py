import random, math
from som import SOM

SQRT_2_PI = math.sqrt(2 * math.pi)
MAX_NORMALIZED_DISTANCE = 0.99

def max_dist(som):
    """calculate maximum distance between code vectors"""
    max_distance = 0
    for q in som.nodes:
        for i in som.nodes:
            d = SOM.vector_distance(q.weights, i.weights)
            if d > max_distance:
                max_distance = d
    return max_distance

def normalized_distance(n1, n2, max_dist):
    return SOM.vector_distance(n1.weights, n2.weights) * MAX_NORMALIZED_DISTANCE / max_dist

def phi_q(p, q, som, f, max_dist, num_samples):
    """partial clusot"""
    f_q = f.get(q.n, 0)
    if f_q > 0:
        sx = q.x + 1 if q.x < p.x or q.x == 0 else q.x - 1
        sy = q.y + 1 if q.y < p.y or q.y == 0 else q.y - 1
        ax = 1 - normalized_distance(som.node_at(q.x, sx), q, max_dist)
        ay = 1 - normalized_distance(som.node_at(q.y, sy), q, max_dist)
        D = (p.x - q.x) * (p.x - q.x) / (ax * ax) + (p.y - q.y) * (p.y - q.y) / (ay * ay)
        return (f_q / num_samples) / SQRT_2_PI * math.exp(-0.5 * D)
    else:
        return 0

def clusot(p, som, data):
    """calculate Clusot surface function for node p"""
    md = max_dist(som)
    num_samples = len(data)
    # calculate neuron hit frequency
    f = {}
    for x in data:
        node = som.find_bmu(x)
        f[node.n] = f.get(node.n, 0) + 1
    return(
        sum(phi_q(p, q, som, f, md, num_samples) for q in som.nodes))