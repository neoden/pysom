#!/usr/bin/env python
# coding: utf-8

import argparse, sys
from som import *
from clusot import clusot

DEFAULT_MAX_ITERATIONS = 100
DEFAULT_INIT_RANDOM_MAGN = 1
DEFAULT_INIT_RANDOM_BIAS = 0
DEFAULT_ALPHA_START = 0.5
DEFAULT_ALPHA_END = 0.01
DEFAULT_RADIUS_SPAN = 2/3
DEFAULT_RADIUS_END = 0

def main():
    parser = argparse.ArgumentParser(description='SOM network command-line tool')
    parser.add_argument('command', nargs='?', help='Command to perform: init|train|clusot|umatrix|process')
    parser.add_argument('-i', '--init', help='Init with dimensions: width*height*inputs')
    parser.add_argument('-s', '--state', help='Map state file name', required=True)
    parser.add_argument('-d', '--data', help='Training dataset')
    parser.add_argument('--alpha', help='Training function parameters: variant,arg1,arg2...')
    parser.add_argument('--radius', help='Radius function parameters: variant,arg1,arg2...')
    parser.add_argument('--nh', help='Neighbourhood function variant')
    parser.add_argument('--maxiter', help='Maximum iterations')
    parser.add_argument('-v', '--verbose', help='Additional information while training', action='store_true')
    args = parser.parse_args()

    som = SOM()

    def init_som():
        width, height, num_inputs = [int(x) for x in args.init.split('*')]
        som.setup(width, height, num_inputs)
        som.init_random(DEFAULT_INIT_RANDOM_MAGN, DEFAULT_INIT_RANDOM_BIAS)        

    if args.command == 'init':
        if not args.m:
            raise Exception('map dimensions not specified')
        init_som()
        som.save_state(args.state)
    elif args.command == 'train':
        if not args.data:
            raise Exception('training dataset not specified')

        if args.init:
            init_som()
        else:
            som.load_state(args.state)

        max_iterations = int(args.maxiter) if args.maxiter else DEFAULT_MAX_ITERATIONS

        if args.alpha:
            p = args.alpha.split(',')
            variant = p[0]
            if variant == 'linear':
                alpha_func = md_linear(start=p[1], end=p[2], epochs=max_iterations)
            elif variant == 'exp':
                alpha_func = md_exp(start=p[1], factor=p[2])
            else:
                raise Exception('unknown alpha function variant')
        else:
            alpha_func = md_linear(
                start=DEFAULT_ALPHA_START, end=DEFAULT_ALPHA_END, epochs=max_iterations)

        if args.radius:
            p = args.radius.split(',')
            variant = p[0]
            if variant == 'linear':
                radius_func = md_linear(start=p[1], end=p[2], epochs=max_iterations)
            elif variant == 'exp':
                radius_func = md_exp(start=p[1], factor=p[2])
            else:
                raise Exception('unknown radius function variant')
        else:
            radius_func = md_linear(
                start=DEFAULT_RADIUS_SPAN * max(som.width, som.height),
                end=DEFAULT_RADIUS_END,
                epochs=max_iterations)

        if args.nh:
            variant = args.nh
            if variant == 'const':
                nh_func = nh_const
            elif variant == 'linear':
                nh_func = nh_linear
            elif variant == 'normal':
                nh_func = nh_normal
            else:
                raise Exception('unknown neighbourhood function variant')
        else:
            nh_func = nh_normal

        columns, data = som.load_data(args.data)
        som.train(data, max_iterations, alpha_func, radius_func, nh_func, args.verbose)
        som.save_state(args.state, columns)
    elif args.command == 'clusot':
        if not args.data:
            raise Exception('dataset not specified')
        som.load_state(args.state)
        columns, data = som.load_data(args.data)
        print('n\tx\ty\tclusot')
        for i in som.nodes:
            print('%d\t%d\t%d\t%f' % (i.n, i.x, i.y, clusot(i, som, data)))
    elif args.command == 'umatrix':
        som.load_state(args.state)
        print('n\tx\ty\tdist')
        for i in som.umatrix():
            node, dist = i
            print('%d\t%d\t%d\t%f' % (node.n, node.x, node.y, dist))
    elif args.command == 'bmu':
        if not args.data:
            raise Exception('dataset not specified')        
        som.load_state(args.state)
        columns, data = som.load_data(args.data)
        umatrix = [dist for node, dist in som.umatrix()]
        print('row\tnode\tx\ty\tumatrix')
        for n, i in enumerate(data):
            node = som.find_bmu(i)
            print('%d\t%d\t%d\t%d\t%f' % (n, node.n, node.x, node.y, umatrix[node.n]))
    else:
        print('unknown command: %s' % args.command)

if __name__ == '__main__':
    main()