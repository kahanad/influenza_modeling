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
data_for_fit = model.create_data_for_fit(prep_data)


# Get parameters - vaccination model
with open('../../Data/vaccination_model/grid_search/grid_search_res.pickle', 'rb') as pickle_in:
# with open('../../Data/vaccination_model/grid_search/grid_search_res_homo.pickle', 'rb') as pickle_in:  # HOMOGENOUS
    grid_search_res = pickle.load(pickle_in)

# Max likelihood subdist
liklihood_subdist = max(grid_search_res, key=lambda x: x['log_likelihood_subdist'])

# Set parameters
parameters_v = liklihood_subdist['parameters']


# Define a function for multiprocessing
def run_model_mp(x):
    print(mp.current_process())

    # Run the model
    model_results = model.run_model(parameters_v, prep_data)
    # model_results = model.run_model(parameters_v, prep_data, homogenous=True)  # HOMOGENOUS

    # Calculate likelihood for each season
    likelihood = model.log_likelihood_agg_by_subdist(model_results['lambdas'], data_for_fit['data_for_fit_subdist'], prep_data)

    return {'model_results': model_results, 'likelihood': likelihood}


def mp_handler(m):
    # Create a pool of processes
    pool = mp.Pool(10)
    # Process in parallel
    results = pool.map(run_model_mp, range(m))
    return results


if __name__ == '__main__':
    # Get results using mp
    m = 10
    for i in range(10):
        mp_res = mp_handler(m)

        # Saving the results
        with open(f'../../data/vaccination_model/model_results_updated/model_results_m{m}_{i+1}_.pickle', 'wb') as pickle_out:
            pickle.dump(mp_res, pickle_out)
