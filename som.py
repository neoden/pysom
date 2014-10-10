# coding: utf-8

import random, math

def md_linear(**kwargs):
    """linear decline from start to end over num_epochs"""
    start, end, num_epochs = \
        float(kwargs['start']), float(kwargs['end']), int(kwargs['epochs'])
    factor = (end - start) / num_epochs
    def f(t):
        return start + (t * factor)
    return f

def md_exp(**kwargs):
    """exponential decline from start with factor"""
    start, factor = float(kwargs['start']), float(kwargs['factor'])
    def f(t):
        return math.exp(-t * factor) * start
    return f

def nh_const(dist, r):
    """neighbourhood function that is constant within r"""
    return 1 if dist <= r else 0

def nh_linear(dist, r):
    """neighbourhood function that is linearly decline to 0 within r"""
    return dist / r if dist <= r and r > 0 else 0

def nh_normal(dist, r):
    """neighbourhood function that is normally distributed from center to r
       outside radius r it is equal to 0"""
    return normal_linear_approximated(dist, r) if dist <= r and r > 0 else 0

# normal distribution with mu=0, sigma=0.4
# equals 1 at x=0
t_normal = [
    0.00000372, 0.00001257, 0.00003996, 0.00011930, 0.00033458,
    0.00088149, 0.00218171, 0.00507262, 0.01107962, 0.02273391,
    0.04382075, 0.07934913, 0.13497742, 0.21569330, 0.32379399,
    0.45662271, 0.60492681, 0.75284358, 0.88016332, 0.96667029,
    0.99735570, 0.96667029, 0.88016332, 0.75284358, 0.60492681,
    0.45662271, 0.32379399, 0.21569330, 0.13497742, 0.07934913,
    0.04382075, 0.02273391, 0.01107962, 0.00507262, 0.00218171,
    0.00088149, 0.00033458, 0.00011930, 0.00003996, 0.00001257,
    0.00000372
]

def normal_rough(x, factor):
    """approximate optimized normal distribution
       round to closest value from t_normal"""
    cell = round(x / factor * 10)
    if cell < -20 or cell > 20:
        return 0
    else:
        return t_normal[cell+20]

def normal_linear_approximated(x, factor):
    """linear approximation when x falls between known values"""
    xn = x / factor * 10
    xf = math.floor(xn)
    xc = math.ceil(xn)
    if xf < -20 or xc > 20:
        return 0
    else:
        frac = xn - xf
        yf = t_normal[xf+20]
        yc = t_normal[xc+20]
        return yf + (yc - yf) * frac

class Node:
    def __init__(self, n, x, y, w):
        self.n = n
        self.x = x
        self.y = y
        self.weights = w

    def distance_to(self, node):
        """euclidian distance to another node based on (x, y) node location"""
        return math.sqrt(
            (self.x - node.x) * (self.x - node.x) 
            + (self.y - node.y) * (self.y - node.y)
        )

class SOM:
    def __init__(self):
        self.nodes = []

    def setup(self, width, height, num_inputs):
        self.width = width
        self.height = height
        self.num_inputs = num_inputs

    def init_random(self, magnitude, bias):
        """create node map initialized with random values"""
        self.nodes = []
        for y in range(self.height):
            for x in range(self.width):
                n = y * self.width + x
                self.nodes.append(Node(n, x, y, 
                    [random.random() * magnitude + bias for i in range(self.num_inputs)]))

    def node_at(self, x, y):
        pos = y * self.width + x
        return self.nodes[pos]

    @staticmethod
    def shuffle(d):
        """shuffle dataset"""
        for i in reversed(range(1, len(d))):
            j = int(random.random() * i)
            d[i], d[j] = d[j], d[i]

    @staticmethod
    def vector_distance(v1, v2):
        """euclidian distance between n-dimensional vectors"""
        s = 0
        for x1, x2 in zip(v1, v2):
            s += (x1 - x2) * (x1 - x2)
        return math.sqrt(s)

    def find_bmu(self, vd):
        """find Best Matching Unit for a given data vector"""
        min_distance = float('+inf')
        bmu = []
        
        for node in self.nodes:
            # calculate distance
            dist = 0
            for x1, x2 in zip(vd, node.weights):
                dist += (x1 - x2) * (x1 - x2)
            dist = math.sqrt(dist)

            if dist < min_distance:
                min_distance = dist
                bmu = [node]
            elif dist == min_distance:
                bmu.append(node)
        
        return bmu[int(random.random() * len(bmu))]

    def adjust_weights(self, vd, bmu, t, alpha, radius, nh_func):
        """adjust map according to BMU"""
        r_inputs = range(self.num_inputs)
        for node in self.nodes:
            _nhf = nh_func(bmu.distance_to(node), radius)
            if _nhf > 0:
                w = node.weights
                for j in r_inputs:
                    w[j] = w[j] + _nhf * alpha * (vd[j] - w[j])

    def train(self, data, max_iterations, alpha_func, radius_func, nh_func, verbose):
        """train SOM against a given dataset"""

        # copy dataset
        data = list(data)
        
        if verbose:
            print('epoch\talpha\tradius\tAQE')
        for t in range(max_iterations):
            SOM.shuffle(data)
            alpha = alpha_func(t)
            radius = radius_func(t)
            if verbose:
                aqe = self.avg_quantization_error(data)
                print('%d\t%f\t%f\t%f' % (t, alpha, radius, aqe))
            for i in data:
                bmu = self.find_bmu(i)
                self.adjust_weights(i, bmu, t, alpha, radius, nh_func)

    def avg_quantization_error(self, data):
        """calculate map total error"""
        dist = 0
        for i in data:
            bmu = self.find_bmu(i)
            dist += SOM.vector_distance(i, bmu.weights)
        return dist / len(data)

    def umatrix(self):
        nn = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1), (),      ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]
        dist = 0
        nd = 0
        for i in self.nodes:
            for j in nn:
                if not j:
                    continue
                dx, dy = j
                nx, ny = i.x + dx, i.y + dy
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                nd += 1
                dist += SOM.vector_distance(self.node_at(nx, ny).weights, i.weights)
            yield i, dist / nd

    def save_state(self, filename, columns=None):
        """save weights to file"""
        f = open(filename, 'w')

        # headers
        f.write('n\tx\ty')
        for i in range(self.num_inputs):
            colname = columns[i] if columns else 'w%d' % i
            f.write('\t%s' % colname)
        f.write('\n')

        # nodes
        for i in self.nodes:
            f.write('%d\t%d\t%d' % (i.n, i.x, i.y))
            for j in range(self.num_inputs):
                f.write('\t%f' % i.weights[j])
            f.write('\n')

        f.close()

    def load_state(self, filename):
        """load weights from file"""
        self.nodes = []

        f = open(filename)
        next(f)

        for r in f:
            row = r.strip().split('\t')
            n, x, y, weights = int(row[0]), int(row[1]), int(row[2]), [float(x) for x in row[3:]]
            self.nodes.append(Node(n, x, y, weights))

        self.width = x + 1
        self.height = y + 1
        self.num_inputs = len(weights)

        f.close()

    def load_data(self, filename):
        """load dataset from file
           skip columns with name starting from '-'
           """
        data = []

        f = open(filename)
        columns = [x for x in next(f).strip().split('\t')]
        col_inputs = [x for x in columns if x[0] != '-']

        for r in f:
            row = r.strip().split('\t')
            data.append([float(x) for c, x in zip(columns, row) if c[0] != '-'])

        if len(col_inputs) != self.num_inputs:
            raise Exception("number of inputs in the file doesn't match network setup")

        return (col_inputs, data)