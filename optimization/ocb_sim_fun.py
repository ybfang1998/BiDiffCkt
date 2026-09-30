import os
import random
import numpy as np
from pathlib import Path
node_to_netlist_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    10: 6,
    11: 7,
    12: 8,
    13: 9,
    14: 8,
    15: 9,
    16: 10,
    17: 11,
    18: 10,
    19: 11,
    20: 12,
    21: 13,
    22: 12,
    23: 13,
}

OCB_subnetlist_paras = {
    0: ['r'],
    1: ['c'],
    2: ['r', 'c'],
    3: ['r', 'c'],
    4: ['gm', 'rprs', 'cprs'],
    5: ['gm', 'rprs', 'cprs'],
    6: ['c', 'gm', 'rprs', 'cprs'],
    7: ['c', 'gm', 'rprs', 'cprs'],
    8: ['r', 'gm', 'rprs', 'cprs'],
    9: ['r', 'gm', 'rprs', 'cprs'],
    10: ['c', 'r', 'gm', 'rprs', 'cprs'],
    11: ['c', 'r', 'gm', 'rprs', 'cprs'],
    12: ['c', 'r', 'gm', 'rprs', 'cprs'],
    13: ['c', 'r', 'gm', 'rprs', 'cprs'],
}


OCB_subnetlist_unit = {
    0: ['M'],
    1: ['p'],
    2: ['M', 'p'],
    3: ['M', 'p'],
    4: ['m', 'M', 'p'],
    5: ['m', 'M', 'p'],
    6: ['p', 'm', 'M', 'p'],
    7: ['p', 'm', 'M', 'p'],
    8: ['M', 'm', 'M', 'p'],
    9: ['M', 'm', 'M', 'p'],
    10: ['p', 'M', 'm', 'M', 'p'],
    11: ['p', 'M', 'm', 'M', 'p'],
    12: ['p', 'M', 'm', 'M', 'p'],
    13: ['p', 'M', 'm', 'M', 'p'],
}

OCB_subnetlist_node = {0: "subckt single_r_{id} IN OUT \n" + "    R0 (IN OUT) resistor r={} \n" + "ends single_r_{id} \n",
                   1: "subckt single_c_{id} IN OUT \n" + "    C0 (IN OUT) capacitor c={} \n" + "ends single_c_{id} \n",
                   2: "subckt r_c_s_{id} IN OUT \n" + "    R0 (IN net1) resistor r={} \n" +
                       "  C0 (net1 OUT) capacitor c={} \n" + "ends r_c_s_{id} \n",
                   3: "subckt r_c_p_{id} IN OUT \n" + "    R0 (IN OUT) resistor r={} \n" +
                       "  C0 (IN OUT) capacitor c={} \n" + "ends r_c_p_{id} \n",
                   4: "subckt tc_stage_{id} GND IN OUT \n" + "    G0 (OUT GND IN GND) vccs gm={} \n" +
                       "    R0 (OUT GND) resistor r={} \n" + "    C0 (IN GND) capacitor c={} \n" +
                       "ends tc_stage_{id} \n",
                   5: "subckt tc_stage_neg_{id} GND IN OUT \n" + "    G0 (OUT GND IN GND) vccs gm=-{} \n" +
                      "    R0 (OUT GND) resistor r={} \n" + "    C0 (IN GND) capacitor c={} \n" +
                      "ends tc_stage_neg_{id} \n",
                   6: "subckt ts_c_p_{id} GND IN OUT \n" + "    C1 (IN OUT) capacitor c={} \n" +
                        "    G0 (OUT GND IN GND) vccs gm={} \n" + "    R0 (OUT GND) resistor r={} \n" +
                        "    C0 (IN GND) capacitor c={} \n" +
                        "ends ts_c_p_{id} \n",
                   7: "subckt ts_c_p_neg_{id} GND IN OUT \n" + "    C1 (IN OUT) capacitor c={} \n" +
                      "    G0 (OUT GND IN GND) vccs gm=-{} \n" + "    R0 (OUT GND) resistor r={} \n" +
                      "    C0 (IN GND) capacitor c={} \n" +
                      "ends ts_c_p_neg_{id} \n",
                   8: "subckt ts_r_p_{id} GND IN OUT \n" + "    R1 (IN OUT) resistor r={} \n" +
                        "    G0 (OUT GND IN GND) vccs gm={} \n" + "    R0 (OUT GND) resistor r={} \n" +
                        "    C0 (IN GND) capacitor c={} \n" +
                        "ends ts_r_p_{id} \n",
                   9: "subckt ts_r_p_neg_{id} GND IN OUT \n" + "    R1 (IN OUT) resistor r={} \n" +
                      "    G0 (OUT GND IN GND) vccs gm=-{} \n" + "    R0 (OUT GND) resistor r={} \n" +
                      "    C0 (IN GND) capacitor c={} \n" +
                      "ends ts_r_p_neg_{id} \n",
                   10: "subckt ts_r_c_p_{id} GND IN OUT \n" + "    C1 (IN OUT) capacitor c={} \n" +
                        "    R1 (IN OUT) resistor r={} \n" + "    G0 (OUT GND IN GND) vccs gm={} \n" +
                        "    R0 (OUT GND) resistor r={} \n" + "    C0 (IN GND) capacitor c={} \n" +
                        "ends ts_r_c_p_{id} \n",
                   11: "subckt ts_r_c_p_neg_{id} GND IN OUT \n" + "    C1 (IN OUT) capacitor c={} \n" +
                       "    R1 (IN OUT) resistor r={} \n" + "    G0 (OUT GND IN GND) vccs gm=-{} \n" +
                       "    R0 (OUT GND) resistor r={} \n" + "    C0 (IN GND) capacitor c={} \n" +
                       "ends ts_r_c_p_neg_{id} \n",
                   12: "subckt ts_r_c_s_{id} GND IN OUT \n" + "    C1 (net1 OUT) capacitor c={} \n" +
                        "    R1 (IN net1) resistor r={} \n" + "    G0 (OUT GND IN GND) vccs gm={} \n" +
                        "    R0 (OUT GND) resistor r={} \n" + "    C0 (IN GND) capacitor c={} \n" +
                        "ends ts_r_c_s_{id} \n",
                   13: "subckt ts_r_c_s_neg_{id} GND IN OUT \n" + "    C1 (net1 OUT) capacitor c={} \n" +
                       "    R1 (IN net1) resistor r={} \n" + "    G0 (OUT GND IN GND) vccs gm=-{} \n" +
                       "    R0 (OUT GND) resistor r={} \n" + "    C0 (IN GND) capacitor c={} \n" +
                       "ends ts_r_c_s_neg_{id} \n"
                   }

def r_prs_calculate(gm, Gain_A):
    # return the parasitic R in M ohm
    return Gain_A / abs(gm) * 1e-3

def c_prs_calculate(gm):
    # return the parasitic C in pf
    return gm / 6.28 * 5

def valid_bidiffckt_OCB(g):
    U = g['U']
    V = g['V']
    E = g['E']

    # 1. check there is only one input/output node in V
    # 2. Check the Input and Output Node has at least one connection
    E_T = E.transpose(0, 1)
    n_input = 0
    n_output = 0
    for i, v in enumerate(V):
        if v == 0:
            n_input += 1
            if E_T.sum(-1)[i] == 0:
                return False
        elif v == 1:
            n_output += 1
            if E_T.sum(-1)[i] == 0:
                return False
    if n_input != 1 or n_output != 1:
        return False

    # 3. check each device has only one input and one output connection to nets
    for e in E:
        if (e.eq(1).sum() == 1) and (e.eq(2).sum() == 1):
            continue
        else:
            return False

    return True

def OCB_bipartite_graph_to_netlist(g, ids, output_dir=Path("netlists") ,attributes=None):
    # INPUT_INDEX = 1
    # OUT_INDEX = 2
    if attributes is None:
        n_devices = len(g['U'])
        attributes = []
        for i in range(0, n_devices):
            attributes.append({
                'gm': random.uniform(0.01, 1),
                'c': random.uniform(0.01, 10),
                'r': random.uniform(0.01, 1),
                'Gain_A': random.randint(40, 80)
            })

    # Nodes netlist for sub-circuit create
    gm_list = []
    subg_netlist_scripts = []
    subg_netlist_name_list = []
    subg_netlist_port_name_list = []
    node_type_count = {}
    for i in range(0, 14):
        node_type_count[i] = 0
    for v_type, v_attr in zip(g['U'], attributes):
        v_type = node_to_netlist_mapping[v_type.item()]
        paras = OCB_subnetlist_paras[v_type]
        units = OCB_subnetlist_unit[v_type]
        v_attr_list = []
        for para, unit in zip(paras, units):
            if para == "rprs":
                rprs_val = r_prs_calculate(v_attr['gm'], v_attr['Gain_A'])
                v_attr_list.append(str(rprs_val) + unit)
            elif para == "cprs":
                cprs_val = c_prs_calculate(v_attr['gm'])
                v_attr_list.append(str(cprs_val) + unit)
            else:
                v_attr_list.append(str(abs(v_attr[para])) + unit)
            if para == "gm":
                gm_list.append(abs(v_attr[para]))

        node_type_count[v_type] += 1
        subg_script = OCB_subnetlist_node[v_type].format(id=node_type_count[v_type], *v_attr_list)
        subg_netlist_scripts.append(subg_script)

        subg_netlist_name_list.append(subg_script.split(" ")[1])
        subg_netlist_port_name_list.append(subg_script.split(" \n")[0].split(" ")[2:])

    # Net/Port Name create
    net_name_list = []
    current_net_index = 1
    for v_n in g['V']:
        v_n = v_n.item()
        if v_n == 0:
            net_name_list.append('VIN')
        elif v_n == 1:
            net_name_list.append('OUT')
        else:
            net_name_list.append('net' + str(current_net_index))
            current_net_index+=1

    subg_netlist_body_scripts = []
    E = g["E"].tolist()

    i = 0
    while i < len(g['U']):
        subg_node_name = subg_netlist_name_list[i]
        net_script = "I" + str(i) + " ({}) " + subg_node_name + "\n"
        subg_node_port_name = subg_netlist_port_name_list[i]
        j = 0
        while j < len(g['V']):
            e = E[i][j]
            if e == 1:
                in_net = net_name_list[j]
            elif e == 2:
                out_net = net_name_list[j]
            j+=1
        port_script = ""
        for port_name in subg_node_port_name:
            if port_name == 'GND':
                port_script += '0 '
            elif port_name == 'IN':
                port_script += in_net + " "
            elif port_name == 'OUT':
                port_script += out_net
        subg_netlist_body_scripts.append(net_script.format(port_script))
        i+=1


    if output_dir is None:
        return True
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    os.makedirs(file_dir / output_dir, exist_ok=True)

    file1 = open(file_dir / output_dir / Path("netlist_{id}.scs".format(id=ids)), 'w')
    # print('\n')
    file1.writelines('simulator lang=spectre\n')
    file1.writelines('global 0\n')
    file1.writelines('\n')

    for subg_script in subg_netlist_scripts:
        file1.writelines(subg_script)

    file1.writelines('\n')
    for net_script in subg_netlist_body_scripts:
        file1.writelines(net_script)

    file1.writelines('V0 (VIN 0) vsource dc=0 mag=1m type=sine ampl=1m freq=1K \n')
    file1.writelines('\n')
    # R,C Loaders
    file1.writelines('C_L (OUT 0) capacitor c=10p  \n')
    file1.writelines('R_L (OUT 0) resistor r=10M  \n')
    file1.writelines('\n')

    file1.writelines('ac ac start=1 stop=1e9 dec=100 annotate=status \n')
    file1.close()
    return gm_list


# def fom(gain, ugw, pm):
#     return 1.2 * np.abs(gain) / 100 + 1.6 * pm / (-90) + 10 * np.abs(ugw) / 1e9

def pwr(gm_list, v_source=1.8):
    gm_total = 0
    for gm in gm_list:
        gm_total += abs(gm)
    pwr = gm_total/20 * v_source
    return pwr

def fom(gbw, pwr):
    return (gbw/1e6) * 10 / (pwr)

def init_attr(g):
    attributes = []
    node_types = g['U']
    for i, node in enumerate(node_types):
        attributes_types = OCB_subnetlist_paras[node_to_netlist_mapping[node.item()]]
        current_node_attr = {}
        for attr_type in attributes_types:
            if attr_type == "rprs":
                current_node_attr["Gain_A"] = random.randint(40, 80)
            elif attr_type == "cprs":
                pass
            elif attr_type == "c":
                current_node_attr[attr_type] = random.uniform(0.01, 10)
            else:
                current_node_attr[attr_type] = random.uniform(0.01, 1)
        attributes.append(current_node_attr)
    return attributes
