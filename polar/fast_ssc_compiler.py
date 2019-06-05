#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:16:50 2019

@author: alexander
"""
from functools import lru_cache

import numpy as np

from polar.polar_common import polar_hpw

@lru_cache()
def polar_calc_R0_R1_set(N, K, *, ENABLE_R0=True, ENABLE_R1=True, ENABLE_REP=True, ENABLE_SPC=True):
    n = int(np.log2(N))

    A_idx = polar_hpw(N)[-K:]
    A = np.zeros(N,np.uint)
    A[A_idx] = 1

    rates = np.zeros((n+1, N))
    rates[0] = A.copy()


    # Start at the bottom level and propagate the rates
    for n_iter in range(n+1):
        for i in range(int(2**(n - n_iter - 1))):
            rates[n_iter+1, i] = 0.5 * (rates[n_iter, 2*i] + rates[n_iter, 2*i+1])

    # Then trim the tree down again
    alive = [(n,0)]
    R1_nodes, R0_nodes, SPC_nodes, REP_nodes = ([], [], [], [])
    for n_iter in range(n - 1, -1, -1):
        new_alive = []

        for node in alive:
            beta = node[1]

            parent_rate = rates[node[0], beta]
            node = (n_iter + 1, beta)
            if ENABLE_R1 and parent_rate == 1:
                R1_nodes.append(node)
            elif ENABLE_R0 and parent_rate == 0:
                R0_nodes.append(node)
            elif ENABLE_SPC and n_iter == 1 and parent_rate == 0.75:
                SPC_nodes.append(node)
            elif ENABLE_REP and n_iter == 1 and parent_rate == 0.25:
                REP_nodes.append(node)
            else:
                child0 = (n_iter, 2 * beta)
                child1 = (n_iter, 2 * beta + 1)
                new_alive.extend([child0, child1])

        alive = new_alive

    return (set(R0_nodes), set(R1_nodes), set(REP_nodes), set(SPC_nodes))

if __name__ == "__main__":
    N = 8
    K = 0
    DRAW_FULL = False
    ENABLE_R0  = True
    ENABLE_R1  = True
    ENABLE_REP = False
    ENABLE_SPC = False

    import argparse

    from graphviz import Digraph
    import networkx as nx
    import networkx.drawing.nx_agraph as nx_agraph

    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int)
    parser.add_argument('--N', type=int)
    args = parser.parse_args()

    if args.K and args.N:
        N = args.N
        K = args.K
        DRAW_FULL = False


    color_dict = {
        'b': {
            'style': 'filled',
            'fillcolor':'black',
            'fontcolor':'white',
            'lblstyle':'white'
        },
        'w': {
            'style': 'filled',
            'fillcolor':'white',
            'fontcolor':'black',
            'lblstyle':'black'
        }
    }

    dot = Digraph()
    A_idx = polar_hpw(N)[-K:]
    A = np.zeros(N,np.uint)
    A[A_idx] = 1
    n = int(np.log2(N))

    # Calculate rates
    rates = np.zeros((n+1, N))
    rates[0] = A.copy()
    for i in range(N):
        if K > 0 and A[i] == 1:
            dot.attr('node', color_dict['b'])
        else:
            dot.attr('node', color_dict['w'])
        dot.node(f'$v_0^{i}$')

    for n_iter in range(n+1):
        # Start at the bottom level
        for i in range(int(2**(n - n_iter - 1))):
            rates[n_iter+1, i] = 0.5 * (rates[n_iter, 2*i] + rates[n_iter, 2*i+1])

            if K > 0:
                if rates[n_iter+1, i] == 1:
                    dot.attr('node', color_dict['b'])

                elif rates[n_iter+1, i] == 0:
                    dot.attr('node', color_dict['w'])

                else:
                    # color = "#000000" + hex(int(0xff * rates[n_iter+1, i]))[2:]
                    color = "#" + hex(int(0xff * (1 - rates[n_iter+1, i])))[2:] * 3 + "FF"

                    if (1 - rates[n_iter+1, i]) < 0.5:
                        dot.attr('node', style='filled', fillcolor=color, fontcolor='white', lblstyle='white')
                    else:
                        dot.attr('node', style='filled', fillcolor=color, fontcolor='black', lblstyle='black')


            else:
                dot.attr('node', color_dict['w'])

            dot.node(f'$v_{n_iter+1}^{i}$')
            dot.edge(f'$v_{n_iter+1}^{i}$', f'$v_{n_iter}^{2*i}$')
            dot.edge(f'$v_{n_iter+1}^{i}$', f'$v_{n_iter}^{2*i+1}$')

    dot.save(f'polar_code_full_N{N}_K{K}.gv', 'tmp')
    if DRAW_FULL:
        dot.view(f'polar_code_full_N{N}_K{K}.gv', 'tmp')

    # Trim the tree
    graph = nx.DiGraph()

    alive = [(n,0)]

    ssc_instructions = {'F' : 0, 'G' : 0, 'R0' : 0, 'R1' : 0, 'SPC' : 0, 'REP' : 0}
    ssc_nodes = {'F' : 0, 'G' : 0, 'R0' : 0, 'R1' : 0, 'SPC' : 0, 'REP' : 0}
    ssc_time_steps = 0

    R1_nodes, R0_nodes, SPC_nodes, REP_nodes = ([], [], [], [])

    for n_iter in range(n - 1, -1, -1):
        new_alive = []

        for node in alive:
            beta = node[1]

            parent_rate = rates[node[0], beta]
            node = (n_iter + 1, beta)
            if ENABLE_R1 and parent_rate == 1:
                R1_nodes.append(node)
                ssc_instructions['R1'] += 2**node[0]
                ssc_nodes['R1'] += 1
                ssc_time_steps += 1

            elif ENABLE_R0 and parent_rate == 0:
                R0_nodes.append(node)

            elif ENABLE_SPC and n_iter == 1 and parent_rate == 0.75:
                SPC_nodes.append(node)
                ssc_instructions['SPC'] += 1
                ssc_nodes['SPC'] += 1
                ssc_time_steps += 1

            elif ENABLE_REP and n_iter == 1 and parent_rate == 0.25:
                REP_nodes.append(node)
                ssc_instructions['REP'] += 1
                ssc_nodes['REP'] += 1
                ssc_time_steps += 1

            else:
                child0 = (n_iter, 2 * beta)
                child1 = (n_iter, 2 * beta + 1)
                new_alive.extend([child0, child1])

                node_str   = r"$v_{}^{}$".format(node[0], node[1])
                child0_str = r"$v_{}^{}$".format(child0[0], child0[1])
                child1_str = r"$v_{}^{}$".format(child1[0], child1[1])


                if rates[child0[0], child0[1]] == 0:
                    graph.add_edge(node_str, child0_str)
                    ssc_instructions['R0'] += 2**n_iter
                    ssc_nodes['R0'] += 1
                else:
                    graph.add_edge(node_str, child0_str, label=f'f{2**n_iter}')
                    ssc_instructions['F'] += 2**n_iter
                    ssc_nodes['F'] += 1
                    ssc_time_steps += 1

                graph.add_node(child0_str)
                graph.add_node(child1_str)
                graph.add_edge(node_str, child1_str, label=f'g{2**n_iter}')
                ssc_instructions['G'] += 2**n_iter
                ssc_nodes['G'] += 1
                ssc_time_steps += 1


        alive = new_alive


    # Draw the trimmed tree
    agraph = nx_agraph.to_agraph(graph)

    for node in agraph.nodes_iter():
        node.attr['style'] = "filled"
        node.attr['fillcolor'] = "lightgray"

    # I want to keep the R1 and R0 lists "clean" for later processing, so create a new list here
    black_nodes, white_nodes = (R1_nodes.copy(), R0_nodes.copy())
    for node in alive:
        if node[1] in A_idx:
            black_nodes.append(node)
        else:
            white_nodes.append(node)

    for node in black_nodes:
        node_str = r"$v_{}^{}$".format(node[0], node[1])
        n = agraph.get_node(node_str)
        n.attr['fillcolor'] = "black"
        n.attr['fontcolor'] = "white"
        n.attr['lblstyle']  = "white"

    for node in white_nodes:
        node_str = r"$v_{}^{}$".format(node[0], node[1])
        n = agraph.get_node(node_str)
        n.attr['fillcolor'] = "white"

    for node in REP_nodes:
        node_str = r"$v_{}^{}$".format(node[0], node[1])
        n = agraph.get_node(node_str)
        n.attr['fillcolor'] = 'orange'

    for node in SPC_nodes:
        node_str = r"$v_{}^{}$".format(node[0], node[1])
        n = agraph.get_node(node_str)
        n.attr['fillcolor'] = 'green'

    agraph.draw(f'tmp/polar_code_ssc_N{N}_K{K}.png', prog='dot')
    agraph.write(f'tmp/polar_code_ssc_N{N}_K{K}.gv')


    num_instructions_full = N * np.log2(N)
    print("F & G operations performed in the full SC:", num_instructions_full)

    print("Operations performed in SSC:", ssc_instructions)
    print("Number of nodes in SSC:", ssc_nodes)
    print("Expensive SSC operations (not R0):", sum([ssc_instructions[key] for key in ssc_instructions if key != "R0"]))
    print("Time steps, full: {}, SSC: {}".format(2*N-2, ssc_time_steps))
