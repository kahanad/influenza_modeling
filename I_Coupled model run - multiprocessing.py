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

# Get parameters
# Load parameters
with open('../../data/coupled_model/parameters_updated.pickle', 'rb') as pickle_in:
    parameters = pickle.load(pickle_in)

parameters_i = parameters['i']
parameters_v = parameters['v']

# Flu season start date analysis
# For 45% vaccination coverage
# parameters_v['beta'] = 0.028
# Vaccination start date
# vacc_start = 92
# vacc_start = 153
# Season
# season = 2016


# Define a function for multiprocessing
def run_model_mp(x):
    print(mp.current_process())

    # Run the model for all 7 seasons
    model_results_all_seasons = [model.run_coupled_model(parameters_i, parameters_v, prep_data, season) for season in model.seasons]

    # Run the model for 1 season
    # model_results = model.run_coupled_model(parameters_i, parameters_v, prep_data, season, vacc_start=vacc_start)

    # Calculate likelihood for each season
    likelihood_by_season = model.calculate_likelihood_lists_all_seasons(model_results_all_seasons, data_for_fit_i, prep_data, single=True)

    # Filter model results - keep only relevant fields
    model_results_filtered = [{'new_Is': res['new_Is'], 'new_Is_by_age': res['new_Is_by_age'], 'Is_by_clinic_age': res['Is_by_clinic_age'],
                               'new_Iv': res['new_Iv'], 'new_Iv_by_age': res['new_Iv_by_age'], 'Iv_by_clinic_age': res['Iv_by_clinic_age']}
                              for res in model_results_all_seasons]

    return {'model_results': model_results_filtered, 'likelihood_by_season': likelihood_by_season}
    # return model_results


def mp_handler(m):
    # Create a pool of processes
    # pool = mp.Pool(5)
    # Process in parallel
    results = pool.map(run_model_mp, range(m))

    # # Saving the results
    # with open(f'../../data/coupled_model/model_results/model_results_m{m}_{i}.pickle', 'wb') as pickle_out:
    #     pickle.dump(results, pickle_out)

    return results


if __name__ == '__main__':
    # Create a pool of processes
    pool = mp.Pool(10)  # 5

    # Get results using mp
    m = 10  # 5
    for i in range(10):
        mp_res = mp_handler(m)

        # Saving the results
        with open(f'../../data/coupled_model/model_results_updated/model_results_m{m}_{i}__.pickle', 'wb') as pickle_out:
        # with open(f'../../data/coupled_model/flu_season_start_date/new/vacc_start_late/model_results_low_vacc_m{m}_{i}.pickle', 'wb') as pickle_out:
            pickle.dump(mp_res, pickle_out)
