# coding: utf-8

import argparse, csv, random, math, sys
import configparser
from schema import *

class DictParser(configparser.SafeConfigParser):
    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(self._defaults, **d[k])
            d[k].pop('__name__', None)
        return d

class M:
    @staticmethod
    def md_linear(start, factor):
        def f(t):
            return max(start - (t * factor), 0)
        return f

    @staticmethod
    def md_exp(start, factor):
        def f(t):
            return math.exp(-t * factor) * start
        return f

    @staticmethod
    def nh_const(dist, r):
        return 1 if dist <= r else 0

    @staticmethod
    def nh_linear(dist, r):
        return dist / r if dist <= r and r > 0 else 0

    @staticmethod
    def nh_normal(dist, r):
        return M.normal_linear_approximated(dist, r) if dist <= r and r > 0 else 0

    @staticmethod
    def init_random(magnitude, bias):
        def f(num_inputs):
            return [
                random.random() * magnitude + bias
                for i in range(num_inputs)
            ]
        return f   

    # normal distribution with mu=0, sigma=0.4
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

    @staticmethod
    def normal_rough(x, factor):
        # approximate optimized normal distribution
        # round to closest value from t_normal
        cell = round(x / factor * 10)
        if cell < -20 or cell > 20:
            return 0
        else:
            return M.t_normal[cell+20]

    @staticmethod
    def normal_linear_approximated(x, factor):
        # linear approximation when x falls between known values
        xn = x / factor * 10
        xf = math.floor(xn)
        xc = math.ceil(xn)
        if xf < -20 or xc > 20:
            return 0
        else:
            frac = xn - xf
            yf = M.t_normal[xf+20]
            yc = M.t_normal[xc+20]
            return yf + (yc - yf) * frac

class Node:
    def __init__(self, n, x, y, w):
        self.n = n
        self.x = x
        self.y = y
        self.weights = w

    def distance_to(self, node):
        return math.sqrt(
            (self.x - node.x) * (self.x - node.x) 
            + (self.y - node.y) * (self.y - node.y)
        )

class SOM:
    def __init__(self, config=None, data_file=None):
        # defaults
        self.width = 20
        self.height = 20
        self.max_iterations = 100

        self.init_func = M.init_random(1, 0)
        self.radius_func = M.md_linear(5, 3 / self.max_iterations)
        self.alpha_func = M.md_linear(1, 1 / self.max_iterations)
        self.nh_func = M.nh_const

        if config is not None:
            self.load_config(config)

        self.size = self.height * self.width
        self.num_inputs = 0

        if data_file is not None:
            self.load_data(data_file)

    def load_config(self, c):
        # validate against schema
        config = Schema({
            Optional('som'): {
                'max_iterations': Use(int),
                'width': Use(int),
                'height': Use(int)
            },
            Optional('alpha_func'):  {
                'type': lambda x: x in ['linear', 'exp'],
                'factor': Use(float)
            },
            Optional('radius_func'): {
                'type': lambda x: x in ['linear', 'exp'],
                'start_radius': Use(float),
                'factor': Use(float)
            },
            Optional('nh_func'): {
                'type': lambda x: x in ['const', 'linear', 'normal']
            },
            Optional('init_func'): {
                'type': 'random',
                'magnitude': Use(float),
                'bias': Use(float)
            }
        }).validate(c)

        # load properties
        if 'som' in config:
            self.max_iterations = config['som']['max_iterations']
            self.width = config['som']['width']
            self.height = config['som']['height']

        if 'alpha_func' in config:
            cf = config['alpha_func']
            self.alpha_func = {
                'linear': M.md_linear(1, cf['factor']),
                'exp': M.md_exp(1, cf['factor'])
            }[cf['type']]

        if 'radius_func' in config:
            cf = config['radius_func']
            self.radius_func = {
                'linear': M.md_linear(cf['start_radius'], cf['factor']),
                'exp': M.md_exp(cf['start_radius'], cf['factor'])
            }[cf['type']]

        if 'nh_func' in config:
            cf = config['nh_func']
            self.nh_func = {
                'const': M.nh_const,
                'linear': M.nh_linear,
                'normal': M.nh_normal
            }[cf['type']]

        if 'init_func' in config:
            cf = config['init_func']
            self.init_func = {
                'random': M.init_random(cf['magnitude'], cf['bias'])
            }[cf['type']]

        # post-processing
        self.size = self.height * self.width

    def init_state(self):
        self.nodes = []
        for i in range(self.size):
            x, y = self.node_xy(i)
            self.nodes.append(Node(i, x, y, self.init_func(self.num_inputs)))

    @staticmethod
    def _conv(v, f):
        if f == 'f':
            return float(v)
        elif f == 's':
            return v
        else:
            print('conv: bad format - %s' % f)

    def load_data(self, filename):
        self.data_table = {'colspec': [], 'inputs': [], 'other': []}
        reader = csv.reader(open(filename, 'r'), delimiter='\t')
        for n, row in enumerate(reader):
            if n == 0:
                for i in list(row):
                    cs = {}
                    cs['name'] = i
                    cs['input'] = i[0] != '-'
                    self.data_table['colspec'].append(cs)
                self.num_inputs = sum(1 for i in self.data_table['colspec'] if i['input'])
            else:
                dr = [SOM._conv(v, 'f') for v in row]
                self.data_table['rows'].append(dr)
        self.data = ()

    @staticmethod
    def shuffle(d):
        for i in reversed(range(1, len(d))):
            j = int(random.random() * i)
            d[i], d[j] = d[j], d[i]

    @staticmethod
    def vector_distance(v1, v2):
        s = 0
        for x1, x2 in zip(v1, v2):
            s += (x1 - x2) * (x1 - x2)
        return math.sqrt(s)

    def node_xy(self, pos):
        if pos < self.size:
            x, y = pos % self.width, int(pos / self.height)
            return (x, y)
        else:
            print('node_xy: out of bounds')

    def find_bmu(self, vd):
        # find Best Matching Unit
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

    def adjust_weights(self, vd, bmu, t, alpha, radius):
        r_inputs = range(self.num_inputs)
        for node in self.nodes:
            _nhf = self.nh_func(bmu.distance_to(node), radius)
            if _nhf > 0:
                w = node.weights
                for j in r_inputs:
                    w[j] = w[j] + _nhf * alpha * (vd[j] - w[j])

    def train(self, verbose):
        # copy dataset
        data = list(self.data['input'])
        
        for t in range(self.max_iterations):
            SOM.shuffle(data)
            alpha = self.alpha_func(t)
            radius = self.radius_func(t)
            if verbose: 
                aqe = self.avg_quantization_error()
                print('epoch: %d\talpha: %f\tradius: %f\tAQE: %f' % (t, alpha, radius, aqe))
            for i in data:
                bmu = self.find_bmu(i)
                self.adjust_weights(i, bmu, t, alpha, radius)

    def avg_distance_to_nh(self, node):
        n = 0
        d = 0
        radius = 2
        for i in self.nodes:
            nh = M.nh_const(node.distance_to(i), radius)
            if nh > 0:
                n += 1
                d += SOM.vector_distance(node.weights, i.weights)
        return d / n

    def avg_quantization_error(self):
        dist = 0
        for i in self.data['input']:
            bmu = self.find_bmu(i)
            dist += SOM.vector_distance(i, bmu.weights)
        return dist / len(self.data['input'])

    def print_state(self, filename):
        f = open(filename, 'w')

        # headers
        f.write('x\ty')
        for i in range(self.num_inputs):
            f.write("\tw%d" % i)
        f.write("\tavg_dist")
        f.write('\n')

        # bmu map
        m = {}
        for i in self.data['input']:
            bmu_n = self.find_bmu(i).n
            m[bmu_n] = 1 if m.get(bmu_n) is None else m[bmu_n] + 1

        # nodes
        for i in self.nodes:
            f.write('%d\t%d' % (i.x, i.y))
            for j in range(self.num_inputs):
                f.write('\t%f' % i.weights[j])
            f.write('\t%f' % self.avg_distance_to_nh(i))
            f.write('\n')

        f.close()

    def print_data(self, filename):
        f = open(filename, 'w')

        for i in range(self.num_inputs):
            f.write('i%d\t' % i)
        f.write('node_x\tnode_y')
        f.write('\n')

        for i in self.data:
            for j in i:
                f.write('%f\t' % j)
            bmu = self.find_bmu(i)
            f.write('%d\t%d' % (bmu.x, bmu.y))
            f.write('\n')

        f.close()

def main():
    parser = argparse.ArgumentParser(description='Train SOM network')
    parser.add_argument('--config', help='Configuration file', required=True)
    parser.add_argument('--data', help='Training dataset', required=True)
    parser.add_argument('--state', help='File to save network state', required=True)
    parser.add_argument('--odata', help='Show BMUs for training data')
    parser.add_argument('-v', '--verbose', help='Additional information while training', action='store_true')
    args = parser.parse_args()

    som = SOM()

    parser = DictParser()
    parser.read(args.config)
    config = parser.as_dict()

    som.load_config(config)
    som.load_data(args.data) 
    som.init_state()

    som.train(args.verbose)
    som.print_state(args.state)

    if args.odata is not None:
        som.print_data(args.odata)

if __name__ == '__main__':
    main()