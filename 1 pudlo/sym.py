import numpy as np
import pandas as pd


def simulate_birth_death_times(t, a_N, a_S):
    """
    Simulates the times of birth and death
    :param t: time interval on which process occurs
    :param a_N: rate of a birth poisson process
    :param a_S: parameter of exponential distribution for survival time
    :return: births, deaths - vectors of time someone was born or died
    """
    N = np.random.poisson(t * a_N)
    births = np.random.uniform(0, t, N)
    live_times = np.random.exponential(1 / a_S, N)
    deaths = births + live_times
    return births, deaths


def simulate_birth_death_process(t, a_N, a_S):
    """
    Simulates the birth death process
    :param t: time interval on which process occurs
    :param a_N: rate of a birth poisson process
    :param a_S: parameter of exponential distribution for survival time
    :return: births, deaths - vectors of time someone was born or died
    """
    births, deaths = simulate_birth_death_times(t, a_N, a_S)
    df_births = pd.DataFrame({'time': births})
    df_births['type'] = 'birth'

    df_deaths = pd.DataFrame({'time': deaths})
    df_deaths['type'] = 'death'

    df_total = pd.concat([df_births, df_deaths])
    df_total.sort_values(by='time', inplace=True)
    df_total['balance'] = 2 * (df_total['type'] == 'birth') - 1
    df_total['alive'] = df_total['balance'].cumsum()
    return df_total
