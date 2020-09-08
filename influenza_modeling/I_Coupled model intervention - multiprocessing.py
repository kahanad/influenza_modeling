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


# Get parameters - vaccination model
with open('../../Data/vaccination_model/grid_search_5_res.pickle', 'rb') as pickle_in:
    grid_search_res_1 = pickle.load(pickle_in)

with open('../../Data/vaccination_model/grid_search_6_res.pickle', 'rb') as pickle_in:
    grid_search_res_2 = pickle.load(pickle_in)

with open('../../Data/vaccination_model/grid_search_7_res.pickle', 'rb') as pickle_in:
    grid_search_res_3 = pickle.load(pickle_in)

grid_search_res_v = grid_search_res_1 + grid_search_res_2 + grid_search_res_3

# Max likelihood subdist
liklihood_subdist_v = max(grid_search_res_v, key=lambda x: x['log_likelihood_subdist'])
liklihood_subdist_v['parameters']['beta'] = liklihood_subdist_v['parameters']['beta_2']

# Get parameters - influenza model
# with open('L:/Dor/data/coupled_model/grid_search_2016_1_res.pickle', 'rb') as pickle_in:
#     grid_search_res_i = pickle.load(pickle_in)

# # Max likelihood subdist
# liklihood_subdist_i = max(grid_search_res_i, key=lambda x: x['log_likelihood_subdist'])

# Load parameters_i all seasons
with open('../../data/coupled_model/parameters_i_all_seasons.pickle', 'rb') as pickle_in:
    parameters_i = pickle.load(pickle_in)

# Set parameters
parameters_v = liklihood_subdist_v['parameters']
# parameters_i = liklihood_subdist_i['parameters']


# intervention type
# inter_type = 'random'
# inter_type = 'by_area'
# inter_type = 'by_subdist'
# inter_type = 'by_yeshuv'


# intervention parameters
length = 5

# Number of simulations
m = 40

# Set intervention parameters
# Intervention percents
inter_percents = [0.005, 0.01, 0.02, 0.03, 0.05]
# inter_percents = [0.01, 0.025, 0.05]
# inter_percents = [0.04]

# Vaccination season start time:  1.8, 1.9, 1.10, 1.11
start_times = [61, 92, 122, 153]

# Intervention times (by start time)
all_inter_times = [61, 92, 122, 153, 183, 214]  # 1.8, 1.9, 1.10, 1.11, 1.12, 1.1
intervention_times = {61: all_inter_times, 92: all_inter_times[1:], 122: all_inter_times[2:], 153: all_inter_times[3:]}

# Random intervention dict
interventions_dict_random = {inter_percent:
                                 {start_time: [{'time': time, 'percent': inter_percent, 'len': length,
                                                'vacc_start': start_time, 'type': 'random'}
                                               for time in intervention_times[start_time]]
                                  for start_time in start_times}
                             for inter_percent in inter_percents}

# Intervention by area
# Load nodes by area and age
with open(model.nodes_by_area_age_dict_path, 'rb') as pickle_in:
    nodes_by_area_age = pickle.load(pickle_in)

# Load nodes by area and age
with open(model.nodes_by_area_age_dict_path, 'rb') as pickle_in:
    nodes_by_area_age = pickle.load(pickle_in)

# Load page ranks
with open(model.pagerank_by_area_age_path, 'rb') as pickle_in:
    pageranks = pickle.load(pickle_in)

# Sort areas and age groups by page rank (descending)
areas_age_by_rank_with_rank = sorted(list(pageranks.items()), key=lambda x: x[1], reverse=True)
areas_age_by_rank = list(map(lambda x: x[0], areas_age_by_rank_with_rank))

# Filter irrelevant areas
areas_age_by_rank = list(filter(lambda x: x in nodes_by_area_age, areas_age_by_rank))
areas_age_by_rank_with_rank = list(filter(lambda x: x[0] in nodes_by_area_age, areas_age_by_rank_with_rank))

# Create a list of nodes by PageRank
nodes_by_rank = []
for (area, age) in areas_age_by_rank:
    nodes_by_rank += list(nodes_by_area_age[(area, age)])

# Create intervention dict
interventions_dict_area = {inter_percent:
                               {start_time: [{'time': time, 'percent': inter_percent, 'len': length, 'vacc_start': start_time, 'type': 'by_area',
                                         'nodes_by_rank': nodes_by_rank} for time in intervention_times[start_time]]
                           for start_time in start_times}
                      for inter_percent in inter_percents}

# Load page ranks by subdist
with open('../../Data/vaccination_data/pagerank_by_subdist_age.pickle', 'rb') as pickle_in:
    pageranks_subdist = pickle.load(pickle_in)

# Sort areas and age groups by page rank (descending)
subdists_age_by_rank_with_rank = sorted(list(pageranks_subdist.items()), key=lambda x: x[1], reverse=True)
subdists_age = list(map(lambda x: x[0], subdists_age_by_rank_with_rank))
subdist_ranks = list(map(lambda x: x[1], subdists_age_by_rank_with_rank))

# Create intervention dict
interventions_dict_subdist = {inter_percent:
                                  {start_time: [{'time': time, 'percent': inter_percent, 'len': length, 'vacc_start': start_time,
                               'type': 'by_subdist', 'subdists_age': subdists_age, 'subdist_ranks': subdist_ranks}
                                                for time in intervention_times[start_time]]
                                   for start_time in start_times}
                              for inter_percent in inter_percents}

# Load page ranks by yeshuv
with open('../../Data/vaccination_data/pagerank_by_yeshuv_age.pickle', 'rb') as pickle_in:
    pageranks_yeshuv = pickle.load(pickle_in)

# Sort areas and age groups by page rank (descending)
yeshuv_age_by_rank_with_rank = sorted(list(pageranks_yeshuv.items()), key=lambda x: x[1], reverse=True)
yeshuv_age = list(map(lambda x: x[0], yeshuv_age_by_rank_with_rank))
yeshuv_ranks = list(map(lambda x: x[1], yeshuv_age_by_rank_with_rank))

# Create intervention dict
interventions_dict_yeshuv = {inter_percent:
                                 {start_time: [{'time': time, 'percent': inter_percent, 'len': length, 'vacc_start': start_time,
                               'type': 'by_yeshuv', 'yeshuv_age': yeshuv_age, 'yeshuv_ranks': yeshuv_ranks}
                                               for time in intervention_times[start_time]]
                                  for start_time in start_times}
                             for inter_percent in inter_percents}


# Define a function for multiprocessing
def intervention_mp(intervention):
    print(mp.current_process())

    # Run the model with current intervention
    inter_res = model.intervention_coupled_model(parameters_i, parameters_v, prep_data, data_for_fit_i, data_for_fit_v, intervention,
                                                 intervention['vacc_start'], num_of_simulations=m)

    return {(intervention['percent'], intervention['vacc_start'], intervention['time']): inter_res}


def mp_handler(interventions_list):
    # Create a pool of processes
    pool = mp.Pool(24)
    # Process in parallel
    results = pool.map(intervention_mp, interventions_list)
    return results


if __name__ == '__main__':
    # Intervention dicts to go over
    # inter_dicts = [interventions_dict_random, interventions_dict_subdist, interventions_dict_yeshuv,
    #                interventions_dict_area]
    inter_dicts = [interventions_dict_yeshuv]

    # File names for save
    # file_names = [f'random_intervention_res_all_seasons_m{m}', f'subdist_intervention_res_all_seasons_m{m}',
    #               f'yeshuv_intervention_res_all_seasons_m{m}', f'area_intervention_res_all_seasons_m{m}']
    file_names = [f'yeshuv_intervention_res_all_seasons_m{m}_all_per']

    # Go over the dicts and run all interventions
    for i, interventions_dict in enumerate(inter_dicts):
        # Unpack dictionary
        interventions_list = []
        for inter_percent in interventions_dict:
            for start_time, inter in interventions_dict[inter_percent].items():
                interventions_list.extend(inter)

        # Get results using mp
        res = mp_handler(interventions_list)

        # Saving the results
        with open(f'../../data/coupled_model/{file_names[i]}.pickle', 'wb') as pickle_out:
            pickle.dump(res, pickle_out)
