import numpy as np
import pandas as pd
# import model
import model
import itertools
import multiprocessing as mp
import pickle

# Prepare network and data
prep_data = model.data_and_network_prep()

# Create data for the fit
data_for_fit_i = model.create_data_for_fit_influenza()

data_for_fit_v = model.create_data_for_fit(prep_data)

# Load parameters
# Load parameteres
with open('../../Data/coupled_model/parameters_updated.pickle', 'rb') as pickle_in:
    parameters = pickle.load(pickle_in)

parameters_i = parameters['i']
parameters_v = parameters['v']

# Change vaccination model parameter for different vaccination coverage
# For 10% coverage
# parameters_v['beta'] = 0.01402
# For 30% coverage
# parameters_v['beta'] = 0.0203
# For 40% coverage
# parameters_v['beta'] = 0.025
# For 45 coverage
# parameters_v['beta'] = 0.028

# Number of simulations
m = 100

# Vaccination season start time:  1.8, 1.9, 1.10, 1.11
start_times = [30, 44, 61, 75, 92, 106, 122, 136, 153, 167, 183]


# Define a function for multiprocessing
def intervention_mp(start_time):
    print(mp.current_process())

    # Run the model with current intervention
    inter_res = model.start_date_analysis_coupled_model(parameters_i, parameters_v, prep_data, data_for_fit_i,
                                                        start_time, num_of_simulations=m)

    return {start_time: inter_res}


def mp_handler():
    # Create a pool of processes
    pool = mp.Pool(11)
    # Process in parallel
    results = pool.map(intervention_mp, start_times)
    return results


if __name__ == '__main__':
    res = mp_handler()

    # Saving the results
    with open(f'../../data/coupled_model/start_date_analysis/new/start_date_analysis_20per_m{m}_new.pickle', 'wb') as pickle_out:
        pickle.dump(res, pickle_out)
