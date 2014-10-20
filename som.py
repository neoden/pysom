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

    def __lt__(self, other):
        return self.n < other.n

    def __repr__(self):
        return '%s(%d)' % (self.__class__.__name__, self.n)

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

    def simple_node_distance(self, node1, node2):
        """euclidian distance to another node based on (x, y) node location"""
        return math.sqrt(
            (node1.x - node2.x) * (node1.x - node2.x) 
            + (node1.y - node2.y) * (node1.y - node2.y)
        )

    def toroidal_node_distance(self, node1, node2):
        """euclidian distance to another node on a toroidal map"""
        x_min = min(node1.x, node2.x)
        x_max = max(node1.x, node2.x)
        y_min = min(node1.y, node2.y)
        y_max = max(node1.y, node2.y)
        delta_x = min(x_max - x_min, x_min + self.width - x_max)
        delta_y = min(y_max - y_min, y_min + self.height - y_max)
        return math.sqrt(delta_x * delta_x + delta_y * delta_y)

    @staticmethod
    def shuffle(d):
        """shuffle dataset"""
        for i in reversed(range(1, len(d))):
            j = int(random.random() * i)
            d[i], d[j] = d[j], d[i]

    @staticmethod
    def vector_distance(v1, v2):
        """euclidian distance between n-dimensional vectors
           missing values are skipped"""
        s = 0
        for x1, x2 in zip(v1, v2):
            if x1 and x2:
                s += (x1 - x2) * (x1 - x2)
        return math.sqrt(s)

    def find_bmu(self, vd):
        """find Best Matching Unit for a given data vector"""
        min_distance = float('+inf')
        bmu = []
        
        for node in self.nodes:
            dist = SOM.vector_distance(vd, node.weights)

            if dist < min_distance:
                min_distance = dist
                bmu = [node]
            elif dist == min_distance:
                bmu.append(node)
        
        return bmu[int(random.random() * len(bmu))]

    def adjust_weights(self, vd, bmu, t, alpha, radius, nh_func, node_distance):
        """adjust map according to BMU"""
        r_inputs = range(self.num_inputs)
        for node in self.nodes:
            _nhf = nh_func(node_distance(bmu, node), radius)
            if _nhf > 0:
                w = node.weights
                for j in r_inputs:
                    if vd[j]:
                        w[j] = w[j] + _nhf * alpha * (vd[j] - w[j])


    def set_columns(self, cols):
        self.columns = cols

    def train(self, data, max_iterations, alpha_func, radius_func, nh_func, toroidal,
              verbose, brief):
        """train SOM against a given dataset"""

        node_distance = self.toroidal_node_distance if toroidal else self.simple_node_distance

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
            tdata = data if not brief else data[:int(len(data)/10)]
            for i in tdata:
                bmu = self.find_bmu(i)
                self.adjust_weights(i, bmu, t, alpha, radius, nh_func, node_distance)

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
        for i in self.nodes:
            dist = 0
            nd = 0
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

    def save_state(self, filename):
        """save weights to file"""
        f = open(filename, 'w')

        # headers
        f.write('n\tx\ty')
        for i in range(self.num_inputs):
            f.write('\t%s' % self.columns[i])
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
        r = next(f)
        self.columns = r.strip().split('\t')[3:]

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
           empty cell or 'NULL' is treated as missing data
           """
        data = []

        f = open(filename)
        columns = [x for x in next(f).strip().split('\t')]
        col_inputs = [x for x in columns if x[0] != '-']

        for r in f:
            row = r.strip().split('\t')
            data.append(
                [float(x) if x != '' and x != 'NULL' else None for c, x in zip(columns, row) if c[0] != '-'])

        if len(col_inputs) != self.num_inputs:
            raise Exception("number of inputs in the file doesn't match network setup")

        return (col_inputs, data)