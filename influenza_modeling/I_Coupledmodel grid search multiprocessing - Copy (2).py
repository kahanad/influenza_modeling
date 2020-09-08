import numpy as np
import pandas as pd
# import model
import model
import itertools
import multiprocessing as mp
import pickle

# Prepare network and data
prep_data = model.data_and_network_prep()


def limits(x):
    return min(max(0, x), 1)


# Season for grid search
season = 2017
# Number of sets to check
k = 500
# Number of simulations
m = 11


# Load vaccination model parameters - network
with open('../../data/vaccination_model/grid_search/grid_search_res.pickle', 'rb') as pickle_in:
    grid_search_res = pickle.load(pickle_in)

# # Load vaccination model parameters - homogenous
# with open('D:/LEMA Shared Folder/Dor/Data/vaccination_model/grid_search/grid_search_res_homo.pickle', 'rb') as pickle_in:
#     grid_search_res_homo = pickle.load(pickle_in)

grid_search_res_v = grid_search_res
# grid_search_res_v = grid_search_res_homo

# Max likelihood subdist
liklihood_subdist_v = max(grid_search_res_v, key=lambda x: x['log_likelihood_subdist'])
parameters_v = liklihood_subdist_v['parameters']

# Parameters to check for influenza model
parameters_grid = [{2011: {'beta': limits(np.random.normal(0.0017, 0.0017/10)),
                             'delta': 1,
                             'phi': (-1*np.random.uniform(28, 35)/52)*2*np.pi,
                             'epsilon': 1},

                    2012: {'beta': limits(np.random.normal(0.00144, 0.00144/10)),
                             'delta': 1,
                             'phi': (-1*np.random.uniform(30, 34)/52)*2*np.pi,
                             'epsilon': 1},

                    2013: {'beta': limits(np.random.normal(0.00162, 0.00162/10)),
                             'delta': 1,
                             'phi': (-1*np.random.uniform(28, 35)/52)*2*np.pi,
                             'epsilon': 1},

                    2014: {'beta': limits(np.random.normal(0.0016, 0.0016/10)),
                             'delta': 1,
                             'phi': (-1*np.random.uniform(28, 35)/52)*2*np.pi,
                             'epsilon': 1},

                    2015: {'beta': limits(np.random.normal(0.001525, 0.001525/10)),
                             'delta': 1,
                             'phi': (-1*np.random.uniform(28, 35)/52)*2*np.pi,
                             'epsilon': 1},

                    2016: {'beta': limits(np.random.normal(0.00163, 0.00163/10)),
                             'delta': 1,
                             'phi': (-1*np.random.uniform(28, 32)/52)*2*np.pi,
                             'epsilon': 1},

                    2017: {'beta': limits(np.random.normal(0.00152, 0.00152/10)),
                             'delta': 1,
                             'phi': (-1*np.random.uniform(28, 32)/52)*2*np.pi,
                             'epsilon': 1}}

                   for i in range(k)]


# # Parameters grid homogeneous
# parameters_grid = [{season: {'beta': limits(np.random.normal(0.000000675, 0.000000675/10)),
#                              'delta': 1,
#                              'phi': (-1*np.random.uniform(28, 35)/52)*2*np.pi,
#                              'epsilon': 1}}
#                     for i in range(k)]

# Save current grid
# with open(f'../../data/coupled_model/grid_search/grid_search_{season}_m{m}_k{k}.pickle', 'wb') as pickle_out:
#     pickle.dump(parameters_grid, pickle_out)

# Create data for fit
data_for_fit_v = model.create_data_for_fit(prep_data)
data_for_fit_i = model.create_data_for_fit_influenza()


# Define a function for multiprocessing
def run_model_mp(parameters_i):
    print(mp.current_process())

    # Calculate and save the aggregated log-likelihood - total, children and adult (median of 5 runs)
    # likelihoods = []
    # likelihoods_by_age = []
    likelihoods_by_subdist = []

    for i in range(m):
        # Run model and get results
        model_results = model.run_coupled_model(parameters_i, parameters_v, prep_data, season)  # TODO: CHANGE BACK TO NETWORK MODEL
        # model_results = model.run_coupled_model(parameters_i, parameters_v, prep_data, season, homogenous=True)

        # Calculate log likelihood
        # By clinic and age
        # log_likelihood = model.log_likelihood_influenza(model_results['lambdas'],
        #                                                 data_for_fit_i['by_clinic_age'], season=season)
        # # By age
        # log_likelihood_age = model.log_likelihood_agg_age_influenza(model_results, data_for_fit_i, season=season)

        # By subdist and age
        log_likelihood_age_subdist = model.log_likelihood_agg_by_subdist_influenza(model_results['lambdas'],
                                                                                   data_for_fit_i['by_subdist'],
                                                                                   season, prep_data)

        # Add to the lists
        # likelihoods.append(log_likelihood)
        # likelihoods_by_age.append(log_likelihood_age)
        likelihoods_by_subdist.append(log_likelihood_age_subdist)

    # Calculate medians
    # med_likelihood = np.argsort(np.array(likelihoods))[len(likelihoods)//2]
    # med_likelihood_age = np.argsort(np.array(likelihoods_by_age))[len(likelihoods_by_age)//2]
    med_likelihood_subdist = np.argsort(np.array(likelihoods_by_subdist))[len(likelihoods_by_subdist)//2]

    # Return parameters, log-likelihoods and MSEs
    return {'parameters': parameters_i,
            'log_likelihood_subdist': likelihoods_by_subdist[med_likelihood_subdist]}
            # ,'log_likelihood': likelihoods[med_likelihood],
            # 'log_likelihood_age': likelihoods_by_age[med_likelihood_age]}


def mp_handler():
    # Create a pool of processes
    pool = mp.Pool(14)
    # Process in parallel
    results = pool.map(run_model_mp, parameters_grid)
    return results


if __name__ == '__main__':
    res = mp_handler()

    # Saving the results
    with open(f'../../data/coupled_model/grid_search/corrected/grid_search_{season}_m{m}_k{k}_results.pickle', 'wb') as pickle_out:
        pickle.dump(res, pickle_out)
