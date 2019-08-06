import pandas as pd
import numpy as np


def simulate_two_boxes_times(t, a_N, a_S1, a_S2):
    """
    Simulates the times of arrival and leaving boxes in scenario with two boxes
    :param t: time interval on which process occurs
    :param a_N: rate of a birth poisson process
    :param a_S1: parameter of exponential distribution for survival time for box1
    :param a_S2: parameter of exponential distribution for survival time for box2
    :return: births, t1s, deaths - vectors of time someone was appeared in the first box, moved into the second or died
    """
    N = np.random.poisson(t * a_N)
    births = np.random.uniform(0, t, N)
    live_times_1 = np.random.exponential(1 / a_S1, N)
    t1s = births + live_times_1
    live_times_2 = np.random.exponential(1 / a_S2, N)
    deaths = t1s + live_times_2
    return births, t1s, deaths


def simulate_two_boxes_process(t, a_N, a_S1, a_S2):
    """
    Simulates the two boxes process
    :param t: time interval on which process occurs
    :param a_N: rate of a birth poisson process
    :param a_S1: parameter of exponential distribution for survival time in first box
    :param a_S2: parameter of exponential distribution for survival time in second box
    :return: df - dataframe with raw data of two boxes process
    """
    births, t1s, deaths = simulate_two_boxes_times(t, a_N, a_S1, a_S2)
    df_births = pd.DataFrame({'time': births})
    df_births['type'] = 'birth'

    df_t1s = pd.DataFrame({'time': t1s})
    df_t1s['type'] = 'leave_1'

    df_deaths = pd.DataFrame({'time': deaths})
    df_deaths['type'] = 'death'
    df_total = pd.concat([df_births, df_t1s, df_deaths])
    df_total.sort_values(by='time', inplace=True)

    df_total['balance_1'] = (df_total['type'] == 'birth').astype(int) - (df_total['type'] == 'leave_1').astype(int)
    df_total['balance_2'] = (df_total['type'] == 'leave_1').astype(int) - (df_total['type'] == 'death').astype(int)

    df_total['in_box_1'] = df_total['balance_1'].cumsum()
    df_total['in_box_2'] = df_total['balance_2'].cumsum()
    return df_total[df_total['time'] < t]
