import pandas as pd
import numpy as np


def parse_input(scheme_name):
    """
    Prepare csv to usable dataframes or numpy arrays
    :param scheme_name: name of the directory with necessary files
    :return: nodes, transitions - parsed csv files describing the chain of the process
    """
    transitions = pd.read_csv(f'{scheme_name}/transitions', sep=' ', header=None).values
    nodes = pd.read_csv(f'{scheme_name}/nodes', sep=' ', names=['birth_rate', 'lifetime'])
    nodes.index += 1
    return nodes, transitions


def simulate_multibox(nodes, transitions, t):
    """
    Simulates box process with custom graph transitions and birth and death intensities.
    :param nodes: dataframe with columns: 'birth_rate', 'lifetime', that describes chain nodes
    :param transitions: matrix of transition (probabilities) of the chain in the process:
        Where do points go after they die in one node.
        first column is special and defines the probability of dying (node 0 is special, you can't leave it)
    :param t: length of the process
    :return: History of events (Event is jumping from one node to another) in the span of length t
    """
    # starting points
    current_state_arr = []
    birth_rates = nodes[nodes['birth_rate'] > 0]
    for ind, row in birth_rates.iterrows():
        N = np.random.poisson(t * row['birth_rate'])
        births = np.random.uniform(0, t, N)
        df = pd.DataFrame({'time': births, 'node_entered': ind, 'node_left': 0})
        current_state_arr.append(df)

    current_state = pd.concat(current_state_arr).reset_index(drop=True)
    history = [current_state.copy()]

    # following points
    NODES = transitions.shape[0]
    current_node = 1
    while current_state.loc[current_state['node_entered'] != 0, 'time'].min() < t:
        active_points = current_state[(current_state['node_entered'] != 0) & (current_state['time'] < t)]
        active_points_in_curr_node = active_points[active_points['node_entered'] == current_node]

        curr_lifetime = nodes.loc[current_node, 'lifetime']
        num_points = active_points_in_curr_node.shape[0]

        live_times = np.random.exponential(1 / curr_lifetime, num_points)
        new_nodes = np.random.choice(NODES + 1, size=num_points, p=transitions[current_node - 1])

        current_state.loc[active_points_in_curr_node.index, 'time'] += live_times
        current_state.loc[active_points_in_curr_node.index, 'node_entered'] = new_nodes
        current_state.loc[active_points_in_curr_node.index, 'node_left'] = current_node

        history.append(current_state.loc[active_points_in_curr_node.index].copy())
        current_node = current_node % NODES + 1
    df_history = pd.concat(history).reset_index(drop=True)
    df_history = df_history[df_history['time'] < t]
    return df_history


def simulate_multibox_csv(scheme_name, t):
    """
    Simulate multibox process on the base of scheme named scheme_name
    :param scheme_name: name of the directory with description files
    :param t: length of the process
    :return: dataframe with process events history
    """
    nodes, transitions = parse_input(scheme_name)
    process = simulate_multibox(nodes, transitions, t)
    return process
