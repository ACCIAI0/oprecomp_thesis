#!/usr/bin/python

import json
import networkx as nx
import matplotlib.pyplot as plt


def __parse_assignment(graph, op):
    index = op[1]
    left = op[2][0]
    right = op[2][1]

    res_node = 'V{}'.format(index)
    res_node_temp = 'T{}'.format(index)

    left_op, left_nodes, left_top = __parse_default(graph, left)
    right_op, right_nodes, right_top = __parse_default(graph, right)

    left_nodes.extend(right_nodes)
    nodes = list(set(left_nodes))

    if res_node_temp in nodes:
        res_node = res_node_temp
    if left_top is not None and left_top != res_node and left_top != res_node_temp:
        graph.add_edge(left_top, res_node)
    if right_top is not None and right_top != res_node and right_top != res_node_temp:
        graph.add_edge(right_top, res_node)

    return op[0], nodes, res_node


def __parse_expression(graph, op):
    op_type = op[1]
    index = op[2]
    left = op[3]
    right = None
    if 5 == len(op):
        right = op[4]

    res_node = 'V{}'.format(index)
    res_node_temp = 'T{}'.format(index)

    left_op, left_nodes, left_top = __parse_default(graph, left)
    right_op, right_nodes, right_top = __parse_default(graph, right)

    left_nodes.extend(right_nodes)
    nodes = list(set(left_nodes))

    if res_node not in nodes and res_node_temp not in nodes:
        nodes.append(res_node)
        graph.add_node(res_node)
    if res_node_temp in nodes:
        res_node = res_node_temp
    if left_top is not None and left_top != res_node and left_top != res_node_temp:
        graph.add_edge(left_top, res_node)
    if right_top is not None and right_top != res_node and right_top != res_node_temp:
        graph.add_edge(right_top, res_node)

    return op[0], nodes, res_node


def __parse_conditional_exp(graph, op):
    op_type = op[1]
    index = op[2]
    left = op[3][0]
    right = op[3][1]
    node = 'V{}'.format(index)
    return op[0], [], node


def __parse_primitive(graph, op):
    op_type = op[1]
    return op[0], [], None


def __parse_var(graph, op):
    node = 'V{}'.format(op[1])
    if node not in graph.nodes():
        graph.add_node(node)
    return op[0], [node], node


def __parse_temp(graph, op):
    index = op[1]
    content = op[2]

    node = 'T{}'.format(index)
    if node not in graph.nodes():
        graph.add_node(node)
    nodes = [node]
    cast_op, cast_nodes, _ = __parse_default(graph, content)
    nodes.extend(cast_nodes)
    if cast_op in ['VV', 'V', 'F', 'E']:
        for n in cast_nodes:
            graph.add_edge(n, node)
            nodes.remove(n)
    nodes = list(set(nodes))
    return op[0], nodes, node


def __parse_function(graph, op):
    return_type = op[1]
    _, nodes, top_node = __parse_default(graph, return_type)
    return op[0], nodes, top_node


def __parse_constant(graph, op):
    const_type = op[1]
    value = op[2]
    return op[0], [], None


__parser = {
    'A': __parse_assignment,
    'R': __parse_conditional_exp,
    'P': __parse_primitive,
    'V': __parse_var,
    'E': __parse_expression,
    'T': __parse_temp,
    'F': __parse_function,
    'C': __parse_constant
}


def __parse_default(graph, op):
    if op is None or op[0] is None:
        return 'VV', [], None
    if 1 == len(op):
        op = op[0]
    return __parser[op[0]](graph, op)


def parse_vars_file(file: str) -> nx.DiGraph:
    with open(file) as jfile:
        data = json.load(jfile)
    graph = nx.DiGraph()
    for op in data:
        __parse_default(graph, op)
    return graph


def plot(graph: nx.DiGraph):
    # Creates the figure the draw call will use
    fig = plt.figure()
    nx.draw_kamada_kawai(graph, with_labels=True, node_size=512, alpha=1, font_weight='bold')
    plt.show()
