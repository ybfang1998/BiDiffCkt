import igraph
import numpy as np
from utils import *

direction_node = {
    0: None,
    1: None,
    2: ['+'],
    3: ['+'],
    4: ['+'],
    5: ['+'],
    6: ['+'],
    7: ['+'],
    8: ['-'],
    9: ['-'],
    10: ['+'],
    11: ['+'],
    12: ['-'],
    13: ['-'],
    14: ['+'],
    15: ['+'],
    16: ['-'],
    17: ['-'],
    18: ['+'],
    19: ['+'],
    20: ['-'],
    21: ['-'],
    22: ['+'],
    23: ['+'],
    24: ['-'],
    25: ['-']
}

# r, c, gm
polarity_node = {
    0: None,
    1: None,
    2: [1, 0, 0],
    3: [0, 1, 0],
    4: [1, 1, 0],
    5: [1, 1, 0],
    6: [0, 0, 1],
    7: [0, 0, -1],
    8: [0, 0, 1],
    9: [0, 0, -1],
    10: [0, 1, 1],
    11: [0, 1, -1],
    12: [0, 1, 1],
    13: [0, 1, -1],
    14: [1, 0, 1],
    15: [1, 0, -1],
    16: [1, 0, 1],
    17: [1, 0, -1],
    18: [1, 1, 1],
    19: [1, 1, -1],
    20: [1, 1, 1],
    21: [1, 1, -1],
    22: [1, 1, 1],
    23: [1, 1, -1],
    24: [1, 1, 1],
    25: [1, 1, -1],
}

# subg_node_integreted_convert = {
#     2: 0,
#     3: 1,
#     4: 2,
#     5: 3,
#     6: 4,
#     7: 5,
#     8: 4,
#     9: 5,
#     10: 6,
#     11: 7,
#     12: 6,
#     13: 7,
#     14: 8,
#     15: 9,
#     16: 8,
#     17: 9,
#     18: 10,
#     19: 11,
#     20: 10,
#     21: 11,
#     22: 12,
#     23: 13,
#     24: 12,
#     25: 13,
# }

# Convert OCB dataset to be bipartite netlist
# 1. Each device node has one input and one output pins, where in each edge
#    0 represents no connection
#    1 represents the input pin,
#    2 represents the output, and
# 2. the nets as well as input/output nodes are integrated into nets list, where 0/1 represents input/output node, and 2 represent net node
def ocb_to_bipartite(g, n_U_max, n_V_max, i_IN=1, i_OUT=2):
    U = []
    V = []
    E = []
    pos_list = [v['pos'] for v in g.vs]
    number_of_sub_graph = g.vcount()
    if 4 in pos_list:
        op_stage_type = 3
    else:
        op_stage_type = 2

    inputnet_id_dic = {}
    outputnet_id_dic = {}

    if op_stage_type == 3:
        inputnet_id_dic.update({1: 3})
        inputnet_id_dic.update({2: 0})
        inputnet_id_dic.update({3: 1})
        inputnet_id_dic.update({4: 2})

        outputnet_id_dic.update({0: 0})
        outputnet_id_dic.update({2: 1})
        outputnet_id_dic.update({3: 2})
        outputnet_id_dic.update({4: 3})

        # These are for other node net update, it's meaningless
        inputnet_id_dic.update({0: 0})
        outputnet_id_dic.update({1: 3})

        # Init the nets
        V.append(0)
        V.append(2)
        V.append(2)
        V.append(1)

    if op_stage_type == 2:
        inputnet_id_dic.update({1: 2})
        inputnet_id_dic.update({2: 0})
        inputnet_id_dic.update({3: 1})

        outputnet_id_dic.update({0: 0})
        outputnet_id_dic.update({2: 1})
        outputnet_id_dic.update({3: 2})

        # These are for other node net update, it's meaningless
        inputnet_id_dic.update({0: 0})
        outputnet_id_dic.update({1: 2})

        # Init the nets
        V.append(0)
        V.append(2)
        V.append(1)

    for i, v in enumerate(g.vs):
        node_type = v['type']
        node_pos = v['pos']

        if node_pos in [5, 6, 7]:
            # find the neighbours
            pred_i = g.neighbors(i, mode='in')[0]
            child_i = g.neighbors(i, mode='out')[0]

            pred_pos = g.vs[pred_i]['pos']
            child_pos = g.vs[child_i]['pos']

            if direction_node[node_type] == ['-']:
                # Negative Direction
                inputnet_id = outputnet_id_dic[child_pos]
                outputnet_id = outputnet_id_dic[pred_pos]
            else:
                inputnet_id = outputnet_id_dic[pred_pos]
                outputnet_id = outputnet_id_dic[child_pos]

            inputnet_id_dic.update({node_pos: inputnet_id})
            outputnet_id_dic.update({node_pos: outputnet_id})

    # remove the input/output node from the device side
    inputnet_id_dic.pop(0)
    inputnet_id_dic.pop(1)
    outputnet_id_dic.pop(0)
    outputnet_id_dic.pop(1)

    # Create the BipartiteGraph
    E = np.zeros((n_U_max, n_V_max))  # (n_U, n_V)
    for v in g.vs:
        node_pos = v['pos']
        if node_pos != 0 and node_pos != 1:
            # remove the input, output node from original v
            U.append(v['type']-2)
            i = len(U) - 1
            E[i][inputnet_id_dic[node_pos]] = i_IN
            E[i][outputnet_id_dic[node_pos]] = i_OUT

    ## Mask
    U_mask = [True] * len(U) + [False] * (n_U_max - len(U))
    V_mask = [True] * len(V) + [False] * (n_V_max - len(V))

    return {'U': np.array(U), 'V': np.array(V), 'E': E, 'U_mask': np.array(U_mask), 'V_mask': np.array(V_mask)}





