
# coding: utf-8

# In[4]:

#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import csv


# In[5]:

class Node:
    def __init__(self, n, x, y, w):
        self.n = int(n)
        self.x = int(x)
        self.y = int(y)
        self.weights = w

    def distance_to(self, node):
        return math.sqrt(
            (self.x - node.x) * (self.x - node.x) 
            + (self.y - node.y) * (self.y - node.y)
        )


# In[6]:

def load_state(file, width, height):
    nodes = []
    reader = csv.reader(open(filename, 'r'), delimiter='\t')
    for n, row in enumerate(reader):
        if n == 0:
            continue
        x, y, w = int(row[0]), int(row[1]), [float(x) for x in row[2:-1]]
        nodes.append(Node(n-1, x, y, w))
    return nodes


# In[7]:

def load_data(file):
    data = []
    reader = csv.reader(open(file, 'r'), delimiter='\t')
    for n, row in enumerate(reader):
        if n == 0:
            continue
        data.append(list(row))
    return data


# In[8]:

filename = '/home/xtal/Shared/Lab/pysom/run/fc-state-huge.txt'
width, height = 20, 20


# In[289]:

data_file = '/home/xtal/Shared/Lab/pysom/sample/iris/iris.txt'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
colspec = [1, 1, 1, 1, 0]


# In[9]:

data_file = '/home/xtal/Shared/Lab/pysom/run/fc-huge.txt'
columns = ['desktop_app_chrome',
'desktop_app_firefox',
'desktop_app_ie',
'desktop_app_opera',
'desktop_app_safari',
'desktop_app_other',
'mobile_app_chrome',
'mobile_app_firefox',
'mobile_app_ie',
'mobile_app_opera',
'mobile_app_safari',
'mobile_app_other',
'desktop_os_windows',
'desktop_os_mac',
'desktop_os_linux',
'desktop_os_other',
'mobile_os_android',
'mobile_os_ios',
'mobile_os_windows',
'mobile_os_other',
'conversion_rate',
'avg_dist']
colspec = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]


# In[10]:

nodes = load_state(filename, width, height)


# In[11]:

data = load_data(data_file)


# In[12]:

def input_data(data):
    for i in data:
        yield [float(x) for x, c  in zip(i, colspec) if c == 1]


# In[13]:

def vector_distance(v1, v2):
    s = 0
    for x1, x2 in zip(v1, v2):
        s += (x1 - x2) * (x1 - x2)
    return math.sqrt(s)


# In[14]:

import random, math

def find_bmu(nodes, vd):
    # find Best Matching Unit
    min_distance = float('+inf')
    bmu = []

    for node in nodes:
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


# In[15]:

f = {}
for x in input_data(data):
    node = find_bmu(nodes, x)
    f[node.n] = f.get(node.n, 0) + 1


# In[16]:

sqrt_2_pi = math.sqrt(2 * math.pi)


# In[17]:

def get_node_at(nodes, x, y):
    pos = y * width + x
    return nodes[pos]


# In[18]:

max_distance = []

for q in nodes:
    max_distance.append(max([vector_distance(i.weights, q.weights) for i in nodes]) / 0.99)


# In[22]:

def phi_q(f, p, q, nodes):
    f_q = f.get(q.n, 0)
    if f_q > 0:
        max_dist = max_distance[q.n]
        sx = q.x + 1 if q.x < p.x or q.x == 0 else q.x - 1
        sy = q.y + 1 if q.y < p.y or q.y == 0 else q.y - 1
        ax = 1 - vector_distance(get_node_at(nodes, q.x, sx).weights, q.weights) / max_dist
        ay = 1 - vector_distance(get_node_at(nodes, q.y, sy).weights, q.weights) / max_dist
        D = (p.x - q.x) * (p.x - q.x) / (ax * ax) + (p.y - q.y) * (p.y - q.y) / (ay * ay)
        return (f_q / 150) / sqrt_2_pi * math.exp(-0.5 * D)
    else:
        return 0


# In[23]:

def Phi(f, p, nodes):
    return(sum(phi_q(f, p, q, nodes) for q in nodes))


# In[24]:

m = [[0 for i in range(width)] for i in range(height)]

for i in nodes:
    m[i.y][i.x] = Phi(f, i, nodes)


# In[25]:

#plt.imshow(m, interpolation='bicubic')


# In[28]:

for y in range(height):
    for x in range(width):
        print(x, y, m[y][x])


# In[ ]:



