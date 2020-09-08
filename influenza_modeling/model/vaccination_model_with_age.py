import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import networkx as nx
from itertools import product
import pickle
from scipy.stats import pearsonr

# Default paths - with age
# network_path = '../../data/network/static_network_with_age_100K_corrected.gpickle'
network_path = '../../data/network/static_network_with_age_100K_updated.gpickle'

# contact_matrix_path = '../data/matrix/contact_matrix_final_with_age.pickle'
contact_matrix_path = '../data/matrix/contact_matrix_final_with_age_updated.pickle'

vaccination_data_path = '../../Data/vaccination_data/vaccinated_patients.csv'
vaccination_data_all_seasons_adjusted_path = '../../Data/vaccination_data/vaccination_data_all_seasons_adjusted.pickle'
stat_areas_clinics_path = '../../Data/vaccination_data/stat_areas_with_clinics.csv'
clinics_stat_areas_path = '../../data/vaccination_data/clinics_with_stat_area.pickle'
population_by_clinic_age_path = '../../Data/vaccination_data/population_by_clinic_age.pickle'

# vaccination_coverage_with_age_path = '../../Data/vaccination_data/vaccination_coverage_with_age.pickle'
vaccination_coverage_with_age_path = '../../Data/vaccination_data/vaccination_coverage_with_age_updated.pickle'

contact_matrix_age_path = './model/contact_matrix_age.pickle'

# nodes_by_clinic_age_path = '../../Data/vaccination_data/nodes_by_clinic_age.pickle'
# nodes_by_area_age_dict_path = '../../Data/vaccination_data/nodes_by_area_age_dict.pickle'
# nodes_by_subdist_age_path = '../../Data/vaccination_data/nodes_by_subdist_age.pickle'
# nodes_by_yeshuv_age_path = '../../Data/vaccination_data/nodes_by_yeshuv_age_dict.pickle'

nodes_by_clinic_age_path = '../../Data/vaccination_data/nodes_by_clinic_age_updated.pickle'
nodes_by_area_age_dict_path = '../../Data/vaccination_data/nodes_by_area_age_dict_updated.pickle'
nodes_by_subdist_age_path = '../../Data/vaccination_data/nodes_by_subdist_age_updated.pickle'
nodes_by_yeshuv_age_path = '../../Data/vaccination_data/nodes_by_yeshuv_age_dict_updated.pickle'

pagerank_by_area_age_path = '../../Data/vaccination_data/pagerank_by_area_age.pickle'

default_paths = [network_path, contact_matrix_path, vaccination_data_path, stat_areas_clinics_path,
                 population_by_clinic_age_path, vaccination_coverage_with_age_path]

# Influenza paths
vaccination_coverage_influenza_path = '../../Data/influenza_model/data/vaccination_coverage_influenza.pickle'
infection_rates_path = '../../Data/influenza_model/data/infection_rates_clinics.pickle'
influenza_weekly_cases_by_clinic_age_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_clinic_age.pickle'
influenza_weekly_cases_by_age_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_age.pickle'
influenza_weekly_cases_by_subdist_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_subdist.pickle'
total_influenza_weekly_cases_path = '../../Data/influenza_model/data/total_influenza_weekly.pickle'


def data_and_network_prep(network_path=default_paths[0], contact_matrix_path=default_paths[1],
                          vaccination_data_path=default_paths[2], stat_areas_clinics_path=default_paths[3],
                          population_by_clinic_age_path=default_paths[4]):
    """Receives paths for the data (network, contact matrix, vaccination data, stat_areas-clinic data
    and clinic population data).
    Execute the pre-process on these data
    Returns: processed network, processed vaccination data, list of relevant_clinics, node_clinic dictionary,
            population_by_clinic and age dictionary and list of relevant days in season"""
    #################################
    # ---------- Network ---------- #
    #################################
    # Read the network from gpickle
    network = nx.read_gpickle(network_path)

    # Remove nodes with number of connection lower than threshold (0 in this case)
    thresh = 0
    nodes_to_remove = []
    for n in network.nodes:
        if network.degree[n] <= thresh and network.nodes[n]['contacts'] > thresh:
            nodes_to_remove.append(n)

    # Remove from the network
    network.remove_nodes_from(nodes_to_remove)

    # Network size
    N = len(network.nodes)

    ################################
    # ------ Contact Matrix ------ #
    ################################

    # Loading stat area contact matrix
    contact_matrix = pd.read_pickle(contact_matrix_path)
    contact_matrix.fillna(0, inplace=True)

    ################################
    # ----- Vaccination Data ----- #
    ################################

    # Load vaccination data
    vaccination_data = pd.read_csv(vaccination_data_path)
    vaccination_data['vac_date'] = pd.to_datetime(vaccination_data['vac_date'])

    # Remove incomplete seasons (2007 and 2018)
    vaccination_data = vaccination_data[~vaccination_data.vac_season.isin([2007, 2018])].copy()

    # Add age group
    vaccination_data['age'] = ((vaccination_data.vac_season - vaccination_data.birth_year) > 18).astype(int)

    # Short list of dates (1.9-28.2) and days in season
    dates_2017_short = [pd.Timestamp(2016, 9, 1) + pd.Timedelta(days=1) * i for i in range(181)]
    day_in_season_short = [(date - pd.datetime(date.year if date.month > 5 else date.year - 1, 6, 1)).days
                           for date in dates_2017_short]

    ###################################
    # --- Stat_areas-clinics Data --- #
    ###################################

    # Read stat_area-clinics data
    stat_areas_clinics = pd.read_csv(stat_areas_clinics_path)

    # Load data population by clinic data
    population_by_clinic_age = pd.read_pickle(population_by_clinic_age_path)

    # Get only relevant stat areas
    stat_areas_clinics = stat_areas_clinics[stat_areas_clinics.stat_area_id.isin(set([x[0]
                                                                                      for x in contact_matrix.index]))]
    stat_areas_clinics.set_index('stat_area_id', inplace=True)

    # Create a dictionary of stat area: clinic
    stat_area_clinics_dict = {stat_area_id: stat_areas_clinics.loc[stat_area_id].clinic_code for stat_area_id in
                              stat_areas_clinics.index}

    # Clinics stat areas and subdists
    clinics_stat_areas = pd.read_pickle(clinics_stat_areas_path)

    # Initialize a dictionary for the network population by clinic
    network_pop_by_clinic = {}

    # Add the relevant clinic code for each node and update the population by clinic dictionary
    for n in network.nodes:
        network.nodes[n]['clinic'] = stat_area_clinics_dict[network.nodes[n]['area']]
        network_pop_by_clinic[(network.nodes[n]['clinic'], network.nodes[n]['age'])] = \
            network_pop_by_clinic.get((network.nodes[n]['clinic'], network.nodes[n]['age']), 0) + 1

    # Relevant clinics with age
    relevant_clinics_age = list(network_pop_by_clinic.keys())

    # Get population data only data for relevant clinics
    population_by_clinic_age = population_by_clinic_age.loc[relevant_clinics_age].copy()

    # Create a dictionary {node: (clinic, age)}
    node_clinic_age = {node: (network.nodes[node]['clinic'], network.nodes[node]['age']) for node in network.nodes()}

    # Add to the population data frame
    population_by_clinic_age['network_population'] = population_by_clinic_age.index.map(
        lambda idx: network_pop_by_clinic.get(idx, 0))

    # Calculate the factor between the real data and the network data
    population_by_clinic_age['factor'] = population_by_clinic_age['network_population'] / \
                                         population_by_clinic_age['data_population']

    # Add subdist column to population by clinic and age
    population_by_clinic_age['subdist'] = population_by_clinic_age.apply(lambda row: clinics_stat_areas.loc[row.name[0]].subdist, axis=1)

    # Calcuate proportion out of network population
    population_by_clinic_age['prop_network'] = population_by_clinic_age.network_population / population_by_clinic_age.network_population.sum()

    # Get relevant subdists
    vaccination_coverage_with_age = pd.read_pickle(vaccination_coverage_with_age_path)
    relevant_subdists = vaccination_coverage_with_age.loc[relevant_clinics_age].subdist.unique()

    # Nodes by clinic and age
    with open(nodes_by_clinic_age_path, 'rb') as pickle_in:
        nodes_by_clinic_age = pickle.load(pickle_in)

    # Nodes by nodes by subdist and age
    with open(nodes_by_subdist_age_path, 'rb') as pickle_in:
        nodes_by_subdist_age = pickle.load(pickle_in)

    # Nodes by nodes by yeshuv and age
    with open(nodes_by_yeshuv_age_path, 'rb') as pickle_in:
        nodes_by_yeshuv_age = pickle.load(pickle_in)

    # Influenza cases and diagnoses data
    vaccination_coverage_influenza = pd.read_pickle(vaccination_coverage_influenza_path)
    infection_rates = pd.read_pickle(infection_rates_path)

    prep_data = {'network': network, 'vaccination_data': vaccination_data, 'relevant_clinics_age': relevant_clinics_age, 'relevant_subdists': relevant_subdists,
                 'relevant_subdists_age': list(product(relevant_subdists, [0, 1])), 'node_clinic_age': node_clinic_age,
                 'population_by_clinic_age': population_by_clinic_age, 'clinics_stat_areas': clinics_stat_areas,
                 'day_in_season_short': day_in_season_short, 'nodes_by_clinic_age': nodes_by_clinic_age, 'nodes_by_subdist_age': nodes_by_subdist_age,
                 'nodes_by_yeshuv_age': nodes_by_yeshuv_age, 'vaccination_coverage_with_age': vaccination_coverage_with_age,
                 'vaccination_coverage_influenza': vaccination_coverage_influenza, 'infection_rates': infection_rates, 'N': network.number_of_nodes()}

    return prep_data


def create_data_for_fit(prep_data):
    # Get prep data
    vaccination_data = prep_data['vaccination_data']
    relevant_clinics_age = prep_data['relevant_clinics_age']
    population_by_clinic_age = prep_data['population_by_clinic_age']
    day_in_season_short = prep_data['day_in_season_short']
    vaccination_coverage_with_age = prep_data['vaccination_coverage_with_age']
    relevant_subdists_age = prep_data['relevant_subdists_age']

    ###############################################################
    # --- Create data for the fit - average vaccination count --- #
    ###############################################################
    # Get only relevant seasons
    years = 7  # Only last 7 seasons
    seasons = np.arange(2008, 2017 + 1)[-years:]
    vaccination_data_relevant_seasons = vaccination_data[vaccination_data.vac_season.isin(set(seasons))]

    # Get dates for aggregation
    dates = [pd.Timestamp(2016, 9, 1) + pd.Timedelta(i, unit='d') for i in range(len(prep_data['day_in_season_short']))]

    # Get relevant days
    relevant_days = np.array(day_in_season_short)

    # Get only relevant data (according to the short season definition)
    vaccination_data_short_season = vaccination_data_relevant_seasons[
        vaccination_data_relevant_seasons.vac_day_of_season.isin(set(relevant_days))].copy()

    # Create a dictionary for vaccination count by clinic and age at each stage (day of the season)
    # data_for_fit = dict.fromkeys(list(population_by_clinic_age.index), [0] * len(day_in_season_short))
    data_for_fit = {key: [0] * len(day_in_season_short) for key in (list(population_by_clinic_age.index))}

    # Go over the clinics and age groups
    for clinic, age in relevant_clinics_age:
        # Get only data of current clinic and age
        cur_clinic_age_data = vaccination_data_short_season[(vaccination_data_short_season.clinic_code == clinic) &
                                                            (vaccination_data_short_season.age == age)]

        # Group by dates and count the number of vaccination at each day
        cur_clinic_age_avg_gb = cur_clinic_age_data.groupby('vac_day_of_season').count()[['random_ID']] / years

        # Get current clinic and age vaccination average at each day (including 0 if no vaccination)
        cur_clinic_age_avg_vacc = np.array(
            [cur_clinic_age_avg_gb.loc[day].random_ID if day in cur_clinic_age_avg_gb.index else 0
             for day in relevant_days])

        # Multiply by the factor between the real and model data
        vacc_data_adj = cur_clinic_age_avg_vacc * population_by_clinic_age['factor'].loc[(clinic, age)]

        # Aggregate weekly
        # Create a DF of the cases with the dates as index
        vacc_by_day_df = pd.DataFrame(vacc_data_adj, index=np.array(dates), columns=['vacc_count'])

        # Aggregate weekly
        vacc_by_week = vacc_by_day_df.resample('W').sum().fillna(0).copy()

        # Save to dict
        data_for_fit[(clinic, age)] = vacc_by_week

    ########################################################
    # --- Aggregated data for fit - by subdist and age --- #
    ########################################################
    # Initialize dict to all arrays of 0s
    data_for_fit_subdist = {key: data_for_fit[list(data_for_fit.keys())[0]].copy() * 0 for key in relevant_subdists_age}

    # Go over the clinics and age groups and aggregate according to the clinic's subdist
    for (clinic, age), data in data_for_fit.items():
        subdist = vaccination_coverage_with_age.loc[clinic].subdist[0]
        data_for_fit_subdist[(subdist, age)] = data_for_fit_subdist[(subdist, age)] + data

    ############################################
    # --- Aggregated data for fit - by age --- #
    ############################################
    # Initialize dict to all arrays of 0s
    data_for_fit_by_age = {key: data_for_fit[list(data_for_fit.keys())[0]].copy() * 0 for key in [0, 1, 'total']}

    # Go over the clinics and age groups and aggregate according to the clinic's subdist
    for (clinic, age), data in data_for_fit.items():
        data_for_fit_by_age[age] = data_for_fit_by_age[age] + data
        data_for_fit_by_age['total'] = data_for_fit_by_age['total'] + data

    return {'data_for_fit': data_for_fit, 'infected_data_agg': data_for_fit_by_age['total'],
            'infected_data_agg_children': data_for_fit_by_age[0],
            'infected_data_agg_adult': data_for_fit_by_age[1], 'data_for_fit_subdist': data_for_fit_subdist}


def initialize_vaccination_model(parameters, prep_data, homogenous=False):
    """For using in the coupled model only!"""
    ##############################
    # ----- Initialization ----- #
    ##############################
    # Get prep data
    network, relevant_clinics_age, node_clinic_age = prep_data['network'], prep_data['relevant_clinics_age'], \
                                                     prep_data['node_clinic_age']
    season_length = len(prep_data['day_in_season_short'])

    # Get model parameters
    I_0_size = parameters['I_0_size']

    # Infected - initialize I_0_size of the population
    I_0 = set(np.random.choice(list(network.nodes), replace=False, size=int(round(network.number_of_nodes() * I_0_size))))

    # Susceptible
    S_0 = set(network.nodes) - I_0

    # Recovered - initialize to empty set
    R_0 = set()

    # Initialize lists to save all the states
    S = [S_0]
    I = [I_0]
    R = [R_0]

    # Initialize a list to save the newly infected - total and by age group
    new_I = [set()]
    new_I_by_age = [[set()], [set()]]

    # Initialize a dict for infected by clinic and age
    infected_by_clinic_age = {key: np.array([0.] * season_length) for key in relevant_clinics_age}

    # Initialize a dictionary lambdas_kt
    lambdas = {key: np.array([0.] * season_length) for key in relevant_clinics_age}

    # Initialize an array for lambda_t (aggregated) - total, children, adult
    lambdas_agg_total = np.array([0.] * season_length)
    lambdas_agg_children = np.array([0.] * season_length)
    lambdas_agg_adult = np.array([0.] * season_length)

    # If homogenous model
    if homogenous:
        # Load age contact matrix
        C = pd.read_pickle(contact_matrix_age_path).values

        # Susceptible nodes by age (will be updated each iteration)
        S_by_age = [set(), set()]
        S_by_clinic_age = {(clinic, age): set() for clinic, age in relevant_clinics_age}
        for node in S[0]:
            cur_clinic, cur_age = node_clinic_age[node]
            S_by_age[cur_age].add(node)
            S_by_clinic_age[(cur_clinic, cur_age)].add(node)

        return {'S': S, 'I': I, 'R': R, 'new_I': new_I, 'new_I_by_age': new_I_by_age, 'infected_by_clinic_age': infected_by_clinic_age,
                'S_by_age': S_by_age, 'S_by_clinic_age': S_by_clinic_age, 'C': C,
                'lambdas': lambdas, 'lambdas_agg_total': lambdas_agg_total, 'lambdas_agg_children': lambdas_agg_children,
                'lambdas_agg_adult': lambdas_agg_adult}

    return {'S': S, 'I': I, 'R': R, 'new_I': new_I, 'new_I_by_age': new_I_by_age, 'infected_by_clinic_age': infected_by_clinic_age,
            'lambdas': lambdas, 'lambdas_agg_total': lambdas_agg_total, 'lambdas_agg_children': lambdas_agg_children,
            'lambdas_agg_adult': lambdas_agg_adult}


def run_model(parameters, prep_data, initial_I=None, homogenous=False):
    ##############################
    # ----- Initialization ----- #
    ##############################
    # Get prep data
    network, relevant_clinics_age, node_clinic_age = prep_data['network'], prep_data['relevant_clinics_age'],\
                                                     prep_data['node_clinic_age']
    season_length = len(prep_data['day_in_season_short'])

    # Get model parameters
    beta = parameters['beta']
    delta = [parameters['delta'], 1]
    gamma = parameters['gamma']
    I_0_size = parameters['I_0_size']
    epsilon = parameters.get('epsilon', 1)

    # Exposed - initialize to empty set or initial_E if exists
    I_0 = initial_I if initial_I else set()

    # Infected - initialize I_0_size of the population
    I_0 = I_0.union(set(np.random.choice(list(network.nodes), replace=False, size=int(round(network.number_of_nodes()*I_0_size)))))

    # Susceptible
    S_0 = set(network.nodes) - I_0

    # Recovered - initialize to empty set
    R_0 = set()

    # Initialize lists to save all the states
    S = [S_0]
    I = [I_0]
    R = [R_0]

    # Initialize a list to save the newly infected - total and by age group
    new_I = [set()]
    new_I_by_age = [[set()], [set()]]
    total_infected_by_beta = [0, 0]

    # Initialize a dictionary lambdas_kt
    lambdas = {key: np.array([0.] * season_length) for key in relevant_clinics_age}

    # Initialize an array for lambda_t (aggregated) - total, children, adult
    lambdas_agg_total = np.array([0.] * season_length)
    lambdas_agg_children = np.array([0.] * season_length)
    lambdas_agg_adult = np.array([0.] * season_length)

    # Initialize a dict for infected by clinic and age
    infected_by_clinic_age = {key: np.array([0.] * season_length) for key in relevant_clinics_age}

    # If homogenous model
    if homogenous:
        # Load age contact matrix
        C = pd.read_pickle(contact_matrix_age_path).values

        # Susceptible nodes by age (will be updated each iteration)
        S_by_age = [set(), set()]
        S_by_clinic_age = {(clinic, age): set() for clinic, age in relevant_clinics_age}
        for node in S[0]:
            cur_clinic, cur_age = node_clinic_age[node]
            S_by_age[cur_age].add(node)
            S_by_clinic_age[(cur_clinic, cur_age)].add(node)

    #############################
    # ------- Run Model ------- #
    #############################

    # for t in tqdm(range(season_length)):
    for t in (range(season_length)):
        new_infected_t = set()
        new_recovered_t = set()
        new_infected_by_age_t = [set(), set()]

        # --- Infection from friends --- #
        # Go over the infected individuals
        if not homogenous:
            for node in I[-1]:
                for contact in network[node]:
                    # If the contact is susceptible and not exposed in this stage yet
                    if contact in S[-1] and contact not in new_infected_t:
                        # Get age and clinic
                        cur_clinic, cur_age = node_clinic_age[contact]
                        # Contact is infected with probability beta_2 * delta
                        if np.random.rand() < beta * delta[cur_age]:
                            new_infected_t.add(contact)
                            new_infected_by_age_t[cur_age].add(contact)
                            infected_by_clinic_age[(cur_clinic, cur_age)][t] += 1
                        # Update lambda_kt
                        lambdas[(cur_clinic, cur_age)][t] += beta * delta[cur_age]
                        lambdas_agg_total[t] += beta * delta[cur_age]
                        if cur_age == 0:
                            lambdas_agg_children[t] += beta * delta[cur_age]
                        else:
                            lambdas_agg_adult[t] += beta * delta[cur_age]

            # Add "epsilon" - constant infection of 1 random node each day
            if len(S[-1] - new_infected_t) <= epsilon:
                epsilon_nodes = S[-1] - new_infected_t
            else:
                epsilon_nodes = set(np.random.choice(list(S[-1] - new_infected_t), epsilon, replace=False))
            for i, node in enumerate(epsilon_nodes):
                # Get age and clinic
                cur_clinic, cur_age = node_clinic_age[node]
                new_infected_t.add(node)
                new_infected_by_age_t[cur_age].add(node)
                infected_by_clinic_age[(cur_clinic, cur_age)][t] += 1

        else:  # If homogenous
            # Get number of nodes to infect (for each age group
            num_of_infected = np.zeros(2)
            for node in I[-1]:
                # Get age of infected individual
                cur_age = node_clinic_age[node][1]

                # Calculate lambdas by clinic and age - by current infected individual
                # to all susceptible individuals (by clinic and age)
                for contact_clinic, contact_age in lambdas:
                    lambdas[(contact_clinic, contact_age)][t] += \
                        C[cur_age, contact_age]*beta*delta[contact_age]*len(S_by_clinic_age[(contact_clinic, contact_age)])

                # Calculate lambda for children and adults (from current infected individual)
                cur_lambda_children = C[cur_age, 0]*beta*delta[0]*len(S_by_age[0])
                cur_lambda_adult = C[cur_age, 1]*beta*delta[1]*len(S_by_age[1])
                # Update total lambda for current iteration
                lambdas_agg_children[t] += cur_lambda_children
                lambdas_agg_adult[t] += cur_lambda_adult
                # Generate number of infected form current individual from each age group (form poisson with lambda)
                cur_num_of_infected = np.array([np.random.poisson(C[cur_age, 0]*beta*delta[0]*len(S_by_age[0])),
                                                np.random.poisson(C[cur_age, 1]*beta*delta[1]*len(S_by_age[1]))])
                num_of_infected += cur_num_of_infected

            # Choose nodes to infect (randomly)
            if num_of_infected[0] < (len(S_by_age[0])):
                new_infected_by_age_t[0] = set(np.random.choice(list(S_by_age[0]), size=int(num_of_infected[0]), replace=False))
            else:
                new_infected_by_age_t[0] = S_by_age[0]

            if num_of_infected[1] < (len(S_by_age[1])):
                new_infected_by_age_t[1] = set(np.random.choice(list(S_by_age[1]), size=int(num_of_infected[1]), replace=False))
            else:
                new_infected_by_age_t[1] = S_by_age[1]

            # Add "epsilon" - constant infection of 1 random node each day
            if len(S[-1] - new_infected_t) <= epsilon:
                epsilon_nodes = S[-1] - new_infected_t
            else:
                epsilon_nodes = set(np.random.choice(list(S[-1] - new_infected_t), epsilon, replace=False))
            for i, node in enumerate(epsilon_nodes):
                # Get age and clinic
                cur_clinic, cur_age = node_clinic_age[node]
                new_infected_t.add(node)
                new_infected_by_age_t[cur_age].add(node)

            # Infect nodes
            new_infected_t = new_infected_by_age_t[0].union(new_infected_by_age_t[1])

            # Update infected by clinic and age and S_by_clinic_age
            for node in new_infected_t:
                cur_clinic, cur_age = node_clinic_age[node]
                infected_by_clinic_age[(cur_clinic, cur_age)][t] += 1
                S_by_clinic_age[(cur_clinic, cur_age)].remove(node)

            # Update S_by_age
            S_by_age[0] = S_by_age[0] - new_infected_by_age_t[0]
            S_by_age[1] = S_by_age[1] - new_infected_by_age_t[1]

        # Transmission from I to R
        for node in I[-1]:
            # Individuals transmitted from I to R with probability gamma
            new_recovered_t.add(node) if np.random.rand() < gamma else None

        # Update stages
        S.append(S[-1] - new_infected_t)
        I.append(I[-1].union(new_infected_t) - new_recovered_t)
        R.append(R[-1].union(new_recovered_t))

        # Save the newly infected
        new_I.append(new_infected_t)
        new_I_by_age[0].append(new_infected_by_age_t[0])
        new_I_by_age[1].append(new_infected_by_age_t[1])

    # Save results to dictionary - full version
    model_results = {'S': S, 'I': I, 'R': R, 'new_I': new_I, 'new_I_by_age': new_I_by_age,
                     'lambdas': lambdas, 'lambdas_agg_total': lambdas_agg_total,
                     'lambdas_agg_children': lambdas_agg_children, 'lambdas_agg_adult': lambdas_agg_adult,
                     'parameters': parameters, 'N': network.number_of_nodes(),
                     'total_infected_by_beta': total_infected_by_beta, 'infected_by_clinic_age': infected_by_clinic_age}

    # # Save results to dictionary - short version
    # model_results = {'new_I': new_I, 'new_I_by_age': new_I_by_age,
    #                  'lambdas': lambdas, 'lambdas_agg_total': lambdas_agg_total,
    #                  'lambdas_agg_children': lambdas_agg_children, 'lambdas_agg_adult': lambdas_agg_adult,
    #                  'parameters': parameters, 'N': network.number_of_nodes(),
    #                  'total_infected_by_beta': total_infected_by_beta, 'infected_by_clinic_age': infected_by_clinic_age}

    return model_results


############################
# --- Helper Functions --- #
############################
def get_vacc_coverage_by_clinic(model_results, prep_data):
    # Get prep data
    network, relevant_clinics_age, population_by_clinic_age = prep_data['network'], prep_data['relevant_clinics_age'],\
                                                              prep_data['population_by_clinic_age']

    # Initialize a dictionary to save the vaccination coverage by clinic
    vacc_coverage_by_clinic_age = dict.fromkeys(relevant_clinics_age, 0)

    # Get vaccinated nodes (I+R in the final stage)
    vaccinated_nodes = model_results['R'][-1].union(model_results['I'][-1])

    # Go over clinics and count the number of vaccinated
    for node in vaccinated_nodes:
        node_clinic = network.nodes[node]['clinic']
        node_age = network.nodes[node]['age']
        vacc_coverage_by_clinic_age[(node_clinic, node_age)] += 1

    # Normalize by clinic network population to receive the coverage %
    for clinic, age in vacc_coverage_by_clinic_age:
        vacc_coverage_by_clinic_age[(clinic, age)] /= population_by_clinic_age.loc[(clinic, age)].network_population

    # Add the total vaccination coverage
    # vacc_coverage_by_clinic_age['total'] = len(vaccinated_nodes)/prep_data['N']

    return vacc_coverage_by_clinic_age


def get_vaccination_coverage(model_results, prep_data, by_clinic=False):
    # Get model coverage by clinic and age
    model_coverage = get_vacc_coverage_by_clinic(model_results, prep_data)
    model_coverage = pd.DataFrame(pd.Series(model_coverage), columns=['model_coverage'])
    model_coverage['clinic_code'] = model_coverage.index.map(lambda x: x[0])
    model_coverage['age'] = model_coverage.index.map(lambda x: x[1])
    model_coverage.set_index(['clinic_code', 'age'], inplace=True)

    # Get vaccination coverage_data and merge with model data
    data_coverage = pd.read_pickle(vaccination_coverage_with_age_path)
    data_coverage = data_coverage.merge(model_coverage, left_index=True, right_index=True)

    # If by_clinic is True, return infection rates in model and data by clinic
    if by_clinic:
        data_coverage = data_coverage[['data_coverage', 'model_coverage']].copy()
        data_coverage.columns = ['data_coverage', 'model_coverage']
        return data_coverage

    # Multiply vaccination coverage by proportion out of subdist for weighted average calculation
    data_coverage['data_mul'] = data_coverage['data_coverage'] * data_coverage['prop_data_pop']
    data_coverage['net_mul'] = data_coverage['model_coverage'] * data_coverage['prop_net_pop']

    # Group by subdist and calculate mean
    vacc_prop_gb_subdist = data_coverage.reset_index().groupby(['subdist', 'age']).sum()[
        ['data_mul', 'net_mul']]
    vacc_prop_gb_subdist.columns = ['data_coverage', 'model_coverage']

    return vacc_prop_gb_subdist


def get_vaccination_coverage_by_age(model_results, data_for_fit, prep_data):
    # Model vaccination coverage
    # Get number of vaccinated by age
    vacc_coverage_by_age = {age: sum([len(new_I) for new_I in model_results['new_I_by_age'][age]]) for age in [0, 1]}
    # Add initial_I nodes
    for node in model_results['I'][0]:
        cur_age = prep_data['node_clinic_age'][node][1]
        vacc_coverage_by_age[cur_age] += 1
    # Add total
    vacc_coverage_by_age['total'] = vacc_coverage_by_age[0] + vacc_coverage_by_age[1]
    # Normalize
    for age in [0, 1]:
        vacc_coverage_by_age[age] /= prep_data['population_by_clinic_age'].loc[pd.IndexSlice[:, age], 'network_population'].sum()
    # Normalize total
    vacc_coverage_by_age['total'] /= prep_data['N']

    # Data vaccination coverage
    children_vacc_coverage = data_for_fit['infected_data_agg_children'].vacc_count.sum() / prep_data['population_by_clinic_age'].loc[
        pd.IndexSlice[:, 0], 'network_population'].sum()
    adult_vacc_coverage = data_for_fit['infected_data_agg_adult'].vacc_count.sum() / prep_data['population_by_clinic_age'].loc[
        pd.IndexSlice[:, 1], 'network_population'].sum()
    total_vacc_coverage = data_for_fit['infected_data_agg'].vacc_count.sum() / prep_data['N']
    # Save to list
    data_vacc_coverage = [children_vacc_coverage, adult_vacc_coverage, total_vacc_coverage]

    # Save to df
    vaccination_coverage = pd.DataFrame(columns=['data', 'model'])
    vaccination_coverage['data'] = data_vacc_coverage
    vaccination_coverage['model'] = [vacc_coverage_by_age[0], vacc_coverage_by_age[1], vacc_coverage_by_age['total']]
    vaccination_coverage['age'] = ['children', 'adults', 'total']
    vaccination_coverage.set_index('age', inplace=True)

    return vaccination_coverage


def get_vacc_model_weekly_cases(model_results, age=None, by_subdist=False, season=None):
    """Receives model results dictionary and returns the number of weekly cases aggregated weekly, by age group"""
    # Get dates for aggregation
    dates = [pd.Timestamp(2016, 9, 1) + pd.Timedelta(i, unit='d') for i in range(181)]
    if season is not None:
        dates = [pd.Timestamp(season-1, 9, 1) + pd.Timedelta(i, unit='d') for i in range(181)]

    if not by_subdist:
        # Get the model new symptomatic cases by season according to age
        if age == 0:
            model_cases_nodes = model_results['new_I_by_age'][0]
        elif age == 1:
            model_cases_nodes = model_results['new_I_by_age'][1]
        else:  # Total
            model_cases_nodes = model_results['new_I']

        # Get number of cases by day
        model_cases_by_day = np.array([len(st) for st in model_cases_nodes[1:]])

    else:  # by subdist
        model_cases_by_day = model_results.copy()

    # Create a DF of the cases with the dates as index
    model_cases_df = pd.DataFrame(model_cases_by_day, index=np.array(dates), columns=['vacc_count'])

    # Aggregate weekly
    model_weekly_cases = model_cases_df.resample('W').sum().fillna(0).copy()

    return model_weekly_cases


#############################
# ---------- Fit ---------- #
#############################
def calc_weekly_lambdas_vacc(lambdas):
    # Get dates for aggregation
    dates = [pd.Timestamp(2016, 9, 1) + pd.Timedelta(i, unit='d') for i in range(181)]

    # Create a DF of the lambdas with the dates as index
    lambdas_df = pd.DataFrame(lambdas, index=np.array(dates), columns=['lambdas'])

    # Aggregate weekly
    weekly_lambdas = lambdas_df.resample('W').sum().fillna(0).copy()
    return weekly_lambdas.values.flatten()


def log_likelihood(lambdas, data_for_fit):
    # Initialize a variable to sum the log-likelihood
    log_like = 0

    # Go over the clinics
    for (clinic, age), cur_lambdas in lambdas.items():
        # Get weekly lambdas
        weekly_lambdas = calc_weekly_lambdas_vacc(cur_lambdas)

        # Sum the log-likelihood for each stage
        log_like += np.sum(-weekly_lambdas + 1e-10 + data_for_fit[(clinic, age)].vacc_count.values * np.log(weekly_lambdas + 1e-10))

    return log_like


def log_likelihood_agg_by_subdist(lambdas, data_for_fit, prep_data):
    relevant_subdists_age = prep_data['relevant_subdists_age']
    vaccination_coverage_with_age = prep_data['vaccination_coverage_with_age']
    day_in_season_short = prep_data['day_in_season_short']

    # Aggregate lambda by subdist
    # Initialize dict to all arrays of 0s
    season_len_weeks = list(data_for_fit.values())[0].shape[0]
    lambdas_subdist = {key: np.array([0] * season_len_weeks) for key in relevant_subdists_age}

    # Go over the clinics and age groups and aggregate according to the clinic's subdist
    for (clinic, age), cur_data in lambdas.items():
        # Get current subdist
        subdist = vaccination_coverage_with_age.loc[clinic].subdist[0]
        # Get weekly lambdas
        weekly_lambdas = calc_weekly_lambdas_vacc(cur_data)
        # Sum
        lambdas_subdist[(subdist, age)] = lambdas_subdist[(subdist, age)] + weekly_lambdas

    # Initialize a variable to sum the log-likelihood
    log_like = 0

    # Go over the clinics
    for subdist, age in lambdas_subdist:
        # Sum the log-likelihood for each stage
        log_like += np.sum(-lambdas_subdist[(subdist, age)] + 1e-10 +
                           data_for_fit[(subdist, age)].vacc_count.values * np.log(lambdas_subdist[(subdist, age)] + 1e-10))

    return log_like


def log_likelihood_agg(lambdas_agg, data_for_fit_agg):
    """Receives aggregated lambdas (total/adult/children) and returns the log likelihood"""
    # Calc weekly lambdas
    weekly_lambdas = calc_weekly_lambdas_vacc(lambdas_agg)

    # Sum the log-likelihood for each stage
    log_like = np.sum(-weekly_lambdas + 1e-10 + data_for_fit_agg.vacc_count.values * np.log(weekly_lambdas + 1e-10))

    return log_like


def log_likelihood_agg_with_age(model_results, data_for_fit):
    """Receives model results and data for fit, calculates the log likelihood for children and adult separately
    and reutrns the sum of these log likelihood"""
    log_like_agg_children = log_likelihood_agg(model_results['lambdas_agg_children'],
                                               data_for_fit['infected_data_agg_children'])
    log_like_agg_adult = log_likelihood_agg(model_results['lambdas_agg_adult'],
                                            data_for_fit['infected_data_agg_adult'])

    return log_like_agg_children + log_like_agg_adult


def calc_correlation_fit_vacc(model_results, data_for_fit, prep_data, by_subdist=False, by_subdist_age=False, weighted=False, smooth=False,
                              window=None):
    # If aggregated correlation fit
    if by_subdist_age:
        # Get model weekly cases by subdist and age
        # Initialize a dict for infected by subdist and age
        I_by_subdist_age = {(subdist, age): np.array([0] * 181) for subdist, age in prep_data['relevant_subdists_age']}

        # Go over clinic and sum the data to subdist leve
        for (clinic, age), data in model_results['infected_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            I_by_subdist_age[(subdist, age)] = I_by_subdist_age[(subdist, age)] + data

        # Get dates for aggregation
        dates = [pd.Timestamp(2016, 9, 1) + pd.Timedelta(i, unit='d') for i in range(181)]

        # Go over subdists and age groupsand aggregate weekly
        model_weekly_vacc_by_subdist_age = {}
        for subdist, age in prep_data['relevant_subdists_age']:
            # Create a DF of the cases with the dates as index
            model_vacc_df = pd.DataFrame(I_by_subdist_age[(subdist, age)], index=np.array(dates), columns=['vacc_count'])
            # Aggregate weekly
            model_weekly_vacc = model_vacc_df.resample('W').sum().fillna(0).copy()
            # Save to dict
            model_weekly_vacc_by_subdist_age[(subdist, age)] = model_weekly_vacc.vacc_count.values

        # Go over subdists and age groups and calculate correlation
        corrs = {}
        pvals = {}
        for subdist, age in prep_data['relevant_subdists_age']:
            # Model weekly vaccination for current subdist and age
            model_weekly_vacc = model_weekly_vacc_by_subdist_age[(subdist, age)]

            # Data weekly vaccination for current subdsit and age
            data_weekly_vacc = data_for_fit['data_for_fit_subdist'][(subdist, age)].vacc_count.values

            # Smooth
            if smooth:
                # Smooth model
                model_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), model_weekly_vacc]))
                model_weekly_vacc = model_weekly_vacc.rolling(window).mean()[0].values[window - 1:]
                # Smooth data
                data_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), data_weekly_vacc]))
                data_weekly_vacc = data_weekly_vacc.rolling(window).mean()[0].values[window - 1:]

            # Calculate correlation for current subdist
            # corrs[(subdist, age)] = np.corrcoef(data_weekly_vacc, model_weekly_vacc)[0, 1]
            corrs[(subdist, age)] = pearsonr(data_weekly_vacc, model_weekly_vacc)[0]
            pvals[(subdist, age)] = pearsonr(data_weekly_vacc, model_weekly_vacc)[1]

        # If weighted - return weighted average of correlation fit
        if weighted:
            # Calculate population proportion by subdist and age
            pop_prop_by_subdist = prep_data['population_by_clinic_age'].reset_index().groupby(['subdist', 'age']).sum().prop_network

            total_corr = sum([pop_prop_by_subdist.loc[key] * corr for key, corr in corrs.items()])
            total_pval = sum([pop_prop_by_subdist.loc[key] * pv for key, pv in pvals.items()])
            return total_corr, total_pval

        else:  # not weighted
            # Return average correlation
            total_corr = sum(list(corrs.values())) / len(list(corrs.values()))
            total_pval = sum(list(pvals.values())) / len(list(pvals.values()))

            return total_corr, total_pval

    elif by_subdist:  # correlation fit by subdist
        # Get model weekly cases by subdist
        # Initialize a dict for infected by subdist
        I_by_subdist = {subdist: np.array([0] * 181) for subdist in prep_data['relevant_subdists']}

        # Go over clinic and sum the data to subdist leve
        for (clinic, age), data in model_results['infected_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            I_by_subdist[subdist] = I_by_subdist[subdist] + data

        # Get dates for aggregation
        dates = [pd.Timestamp(2016, 9, 1) + pd.Timedelta(i, unit='d') for i in range(181)]

        # Go over subdist and aggregate weekly
        model_weekly_vacc_by_subdist = {}
        for subdist in prep_data['relevant_subdists']:
            # Create a DF of the cases with the dates as index
            model_vacc_df = pd.DataFrame(I_by_subdist[subdist], index=np.array(dates), columns=['vacc_count'])
            # Aggregate weekly
            model_weekly_vacc = model_vacc_df.resample('W').sum().fillna(0).copy()
            # Save to dict
            model_weekly_vacc_by_subdist[subdist] = model_weekly_vacc.vacc_count.values

        # Go over subdists and calculate correlation
        corrs = {}
        pvals = {}
        for subdist in prep_data['relevant_subdists']:
            # Model weekly vaccination for current subdist
            model_weekly_vacc = model_weekly_vacc_by_subdist[subdist]

            # Data weekly vaccination for current subdsit
            data_weekly_vacc = data_for_fit['data_for_fit_subdist'][(subdist, 0)].vacc_count.values + \
                               data_for_fit['data_for_fit_subdist'][(subdist, 1)].vacc_count.values
            # Smooth
            if smooth:
                # Smooth model
                model_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), model_weekly_vacc]))
                model_weekly_vacc = model_weekly_vacc.rolling(window).mean()[0].values[window - 1:]
                # Smooth data
                data_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), data_weekly_vacc]))
                data_weekly_vacc = data_weekly_vacc.rolling(window).mean()[0].values[window - 1:]

            # Calculate correlation for current subdist
            # corrs[subdist] = np.corrcoef(data_weekly_vacc, model_weekly_vacc)[0, 1]
            corrs[subdist] = pearsonr(data_weekly_vacc, model_weekly_vacc)[0]
            pvals[subdist] = pearsonr(data_weekly_vacc, model_weekly_vacc)[1]

        # If weighted - return weighted average of correlation
        if weighted:
            # Calculate population proportion by subdist and age
            pop_prop_by_subdist = prep_data['population_by_clinic_age'].reset_index().groupby(['subdist', 'age']).sum().prop_network
            total_corr = sum([(pop_prop_by_subdist.loc[(subdist, 0)] + pop_prop_by_subdist.loc[(subdist, 1)]) * corr
                        for subdist, corr in corrs.items()])
            total_pval = sum([(pop_prop_by_subdist.loc[(subdist, 0)] + pop_prop_by_subdist.loc[(subdist, 1)]) * pv
                        for subdist, pv in pvals.items()])

            return total_corr, total_pval

        else:  # not weighted
            # Return average correlation
            total_corr = sum(list(corrs.values())) / len(list(corrs.values()))
            total_pval = sum(list(pvals.values())) / len(list(pvals.values()))
            return total_corr, total_pval

    # If aggregated correlation fit
    else:
        # Get model weekly cases
        model_weekly_vacc = get_vacc_model_weekly_cases(model_results).vacc_count.values

        # Get vaccination data weekly cases
        data_weekly_vacc = data_for_fit['infected_data_agg'].vacc_count.values

        # Smooth
        if smooth:
            # Smooth model
            model_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), model_weekly_vacc]))
            model_weekly_vacc = model_weekly_vacc.rolling(window).mean()[0].values[window - 1:]
            # Smooth data
            data_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), data_weekly_vacc]))
            data_weekly_vacc = data_weekly_vacc.rolling(window).mean()[0].values[window - 1:]

        # Calculate correlation
        # return np.corrcoef(data_weekly_vacc, model_weekly_vacc)[0, 1]
        return pearsonr(data_weekly_vacc, model_weekly_vacc)


def calc_correlation_fit_vacc_separatly(model_results, prep_data, season, by_subdist=False, by_subdist_age=False, weighted=False,
                                        smooth=False, window=None):
    # Load vaccination data all seasons
    with open(vaccination_data_all_seasons_adjusted_path, 'rb') as pickle_in:
        vacc_data_all_seasons = pickle.load(pickle_in)

    # If correlation fit by_subdist_age
    if by_subdist_age:
        # Get model weekly cases by subdist and age
        # Initialize a dict for infected by subdist and age
        I_by_subdist_age = {(subdist, age): np.array([0] * 181) for subdist, age in prep_data['relevant_subdists_age']}

        # Go over clinic and sum the data to subdist leve
        for (clinic, age), data in model_results['infected_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            I_by_subdist_age[(subdist, age)] = I_by_subdist_age[(subdist, age)] + data

        # Get dates for aggregation
        dates = [pd.Timestamp(season - 1, 9, 1) + pd.Timedelta(i, unit='d') for i in range(181)]

        # Go over subdists and age groups and aggregate weekly
        model_weekly_vacc_by_subdist_age = {}
        for subdist, age in prep_data['relevant_subdists_age']:
            # Create a DF of the cases with the dates as index
            model_vacc_df = pd.DataFrame(I_by_subdist_age[(subdist, age)], index=np.array(dates), columns=['vacc_count'])
            # Aggregate weekly
            model_weekly_vacc = model_vacc_df.resample('W').sum().fillna(0).copy()
            # Save to dict
            model_weekly_vacc_by_subdist_age[(subdist, age)] = model_weekly_vacc.vacc_count.values

        # Go over subdists and age groups and calculate correlation
        corrs = {}
        pvals = {}
        for subdist, age in prep_data['relevant_subdists_age']:
            # Model weekly vaccination for current subdist and age
            model_weekly_vacc = model_weekly_vacc_by_subdist_age[(subdist, age)]

            # Data weekly vaccination for current subdsit and age
            cur_subdist_weekly_vacc = vacc_data_all_seasons['by_subdist_age'][(subdist, age)]
            data_weekly_vacc = cur_subdist_weekly_vacc[cur_subdist_weekly_vacc.season == season].vacc_count.values

            # Smooth
            if smooth:
                # Smooth model
                model_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), model_weekly_vacc]))
                model_weekly_vacc = model_weekly_vacc.rolling(window).mean()[0].values[window - 1:]
                # Smooth data
                data_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), data_weekly_vacc]))
                data_weekly_vacc = data_weekly_vacc.rolling(window).mean()[0].values[window - 1:]

            # Calculate correlation for current subdist
            # corrs[(subdist, age)] = np.corrcoef(data_weekly_vacc, model_weekly_vacc)[0, 1]
            corrs[(subdist, age)] = pearsonr(data_weekly_vacc, model_weekly_vacc)[0]
            pvals[(subdist, age)] = pearsonr(data_weekly_vacc, model_weekly_vacc)[1]

        # If weighted - return weighted average of correlation fit
        if weighted:
            # Calculate population proportion by subdist and age
            pop_prop_by_subdist = prep_data['population_by_clinic_age'].reset_index().groupby(['subdist', 'age']).sum().prop_network

            total_corr = sum([pop_prop_by_subdist.loc[key] * corr for key, corr in corrs.items()])
            total_pval = sum([pop_prop_by_subdist.loc[key] * pv for key, pv in pvals.items()])
            return total_corr, total_pval

        else:  # not weighted
            # Return average correlation
            total_corr = sum(list(corrs.values())) / len(list(corrs.values()))
            total_pval = sum(list(pvals.values())) / len(list(pvals.values()))

            return total_corr, total_pval

    elif by_subdist:  # correlation fit by subdist
        # Get model weekly cases by subdist
        # Initialize a dict for infected by subdist
        I_by_subdist = {subdist: np.array([0] * 181) for subdist in prep_data['relevant_subdists']}

        # Go over clinic and sum the data to subdist leve
        for (clinic, age), data in model_results['infected_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            I_by_subdist[subdist] = I_by_subdist[subdist] + data

        # Get dates for aggregation
        dates = [pd.Timestamp(season - 1, 9, 1) + pd.Timedelta(i, unit='d') for i in range(181)]

        # Go over subdist and aggregate weekly
        model_weekly_vacc_by_subdist = {}
        for subdist in prep_data['relevant_subdists']:
            # Create a DF of the cases with the dates as index
            model_vacc_df = pd.DataFrame(I_by_subdist[subdist], index=np.array(dates), columns=['vacc_count'])
            # Aggregate weekly
            model_weekly_vacc = model_vacc_df.resample('W').sum().fillna(0).copy()
            # Save to dict
            model_weekly_vacc_by_subdist[subdist] = model_weekly_vacc.vacc_count.values

        # Go over subdists and calculate correlation
        corrs = {}
        pvals = {}
        for subdist in prep_data['relevant_subdists']:
            # Model weekly vaccination for current subdist
            model_weekly_vacc = model_weekly_vacc_by_subdist[subdist]

            # Data weekly vaccination for current subdsit
            cur_subdist_weekly_vacc = vacc_data_all_seasons['by_subdist_age'][(subdist, 0)].copy()
            cur_subdist_weekly_vacc.vacc_count += vacc_data_all_seasons['by_subdist_age'][(subdist, 1)].vacc_count

            data_weekly_vacc = cur_subdist_weekly_vacc[cur_subdist_weekly_vacc.season == season].vacc_count.values

            # Smooth
            if smooth:
                # Smooth model
                model_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), model_weekly_vacc]))
                model_weekly_vacc = model_weekly_vacc.rolling(window).mean()[0].values[window - 1:]
                # Smooth data
                data_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), data_weekly_vacc]))
                data_weekly_vacc = data_weekly_vacc.rolling(window).mean()[0].values[window - 1:]

            # Calculate correlation for current subdist
            # corrs[subdist] = np.corrcoef(data_weekly_vacc, model_weekly_vacc)[0, 1]
            corrs[subdist] = pearsonr(data_weekly_vacc, model_weekly_vacc)[0]
            pvals[subdist] = pearsonr(data_weekly_vacc, model_weekly_vacc)[1]

        # If weighted - return weighted average of correlation
        if weighted:
            # Calculate population proportion by subdist and age
            pop_prop_by_subdist = prep_data['population_by_clinic_age'].reset_index().groupby(['subdist', 'age']).sum().prop_network
            total_corr = sum([(pop_prop_by_subdist.loc[(subdist, 0)] + pop_prop_by_subdist.loc[(subdist, 1)]) * corr
                              for subdist, corr in corrs.items()])
            total_pval = sum([(pop_prop_by_subdist.loc[(subdist, 0)] + pop_prop_by_subdist.loc[(subdist, 1)]) * pv
                              for subdist, pv in pvals.items()])

            return total_corr, total_pval

        else:  # not weighted
            # Return average correlation
            total_corr = sum(list(corrs.values())) / len(list(corrs.values()))
            total_pval = sum(list(pvals.values())) / len(list(pvals.values()))
            return total_corr, total_pval

    # If aggregated correlation fit
    else:
        # Get model weekly cases
        model_weekly_vacc = get_vacc_model_weekly_cases(model_results, season=season).vacc_count.values.copy()

        # Get vaccination data weekly cases
        data_weekly_vacc = vacc_data_all_seasons['total'][vacc_data_all_seasons['total'].season == season].vacc_count.values.copy()

        # Smooth
        if smooth:
            # Smooth model
            model_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), model_weekly_vacc]))
            model_weekly_vacc = model_weekly_vacc.rolling(window).mean()[0].values[window - 1:]
            # Smooth data
            data_weekly_vacc = pd.DataFrame(np.concatenate([[0] * (window - 1), data_weekly_vacc]))
            data_weekly_vacc = data_weekly_vacc.rolling(window).mean()[0].values[window - 1:]

        # Calculate correlation
        # return np.corrcoef(data_weekly_vacc, model_weekly_vacc)[0, 1]
        return pearsonr(data_weekly_vacc, model_weekly_vacc)


###############################
# ---------- Plots ---------- #
###############################
def plot_aggregated_fit(model_results, data_for_fit, age=None, homogenous=False):
    # Get data to plot and model results
    # If children
    if age == 0:
        infected_data_for_plot = data_for_fit['infected_data_agg_children']
        model_weekly_vacc = get_vacc_model_weekly_cases(model_results, age=age)
    # If adult
    elif age == 1:
        infected_data_for_plot = data_for_fit['infected_data_agg_adult']
        model_weekly_vacc = get_vacc_model_weekly_cases(model_results, age=age)
    # If total
    else:
        infected_data_for_plot = data_for_fit['infected_data_agg']
        model_weekly_vacc = get_vacc_model_weekly_cases(model_results)

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(infected_data_for_plot.index, infected_data_for_plot.vacc_count, c='r', label='data')
    plt.plot(model_weekly_vacc.index, model_weekly_vacc.vacc_count, linewidth=3, label='model')
    plt.title(f'Influenza Vaccination - {"total" if age is None else ["children", "adults"][age]}', size=22)
    plt.xlabel('time', size=18)
    plt.ylabel('Number of individuals', size=18)
    plt.legend(fontsize=18)
    plt.show()


def plot_aggregated_fit_with_cloud_vaccination(model_results_list, data_for_fit, prep_data, likelihood_by, age=None):
    # Calculate log-likelihood
    if likelihood_by == 'subdist':
        model_res_likelihood_list = [log_likelihood_agg_by_subdist(res['lambdas'], data_for_fit['data_for_fit_subdist'], prep_data)
                                     for res in model_results_list]
    if likelihood_by == 'age':
        model_res_likelihood_list = [log_likelihood_agg_with_age(res, data_for_fit) for res in model_results_list]

    # Get median realization
    med = np.argsort(np.array(model_res_likelihood_list))[len(model_res_likelihood_list)//2]

    # Create fig
    plt.figure(figsize=(20, 10))

    for i, res in enumerate(model_results_list):
        # Get data to plot and model results
        # If children
        if age == 0:
            infected_data_for_plot = data_for_fit['infected_data_agg_children']
            new_I = res['new_I_by_age'][0]
        # If adult
        elif age == 1:
            infected_data_for_plot = data_for_fit['infected_data_agg_adult']
            new_I = res['new_I_by_age'][1]
        # If total
        else:
            infected_data_for_plot = data_for_fit['infected_data_agg']
            new_I = res['new_I']

        # Get newly infected sizes
        new_I_sizes = np.array([len(st) for st in new_I[1:]])

        # Time steps
        ts = np.arange(len(new_I_sizes))

        # Newly infected
        if i == med:
            med_plot = plt.plot(ts, new_I_sizes, label='model', linewidth=2.5)
        else:
            plt.plot(ts, new_I_sizes, label='model', linewidth=0.5, c='gray', alpha=0.4)

    # Plot data
    data_plt = plt.scatter(ts, infected_data_for_plot, label='data', c='r')

    plt.title(f'Influenza Vaccination - {"total" if age is None else ["children", "adults"][age]}', size=22)


    plt.xlabel('Time (days of season)', size=20)
    plt.ylabel('Number of individuals', size=20)

    plt.xlim([0, len(new_I_sizes)])
    plt.xticks(ts[::50], size=15)

    plt.legend(handles=[med_plot[0], data_plt], fontsize=20)

    plt.show()


def plot_fit_by_subdist_vaccination(model_results, data_for_fit, prep_data):
    # Aggregate lambda by subdist
    # Initialize dict to all arrays of 0s
    infected_by_subdist = {key: np.array([0] * len(prep_data['day_in_season_short'])) for key in
                           prep_data['relevant_subdists_age']}

    # Go over the clinics and age groups and aggregate according to the clinic's subdist
    for (clinic, age), data in model_results['infected_by_clinic_age'].items():
        subdist = prep_data['vaccination_coverage_with_age'].loc[clinic].subdist[0]
        infected_by_subdist[(subdist, age)] = infected_by_subdist[(subdist, age)] + data

    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(15, 30))  # , sharey=True)
    plt.tight_layout(w_pad=3, h_pad=5)

    ages = ['children', 'adults']

    # Time steps
    ts = np.arange(list(infected_by_subdist.values())[0].size)

    for i, (subdist, age) in enumerate(infected_by_subdist):
        # Plot newly infected
        axs[i // 2, age].plot(ts, infected_by_subdist[(subdist, age)], label='model', linewidth=1.5)
        axs[i // 2, age].scatter(ts, data_for_fit['data_for_fit_subdist'][(subdist, age)], label='data', c='r', s=3)

        axs[i // 2, age].set_title(f'subdistrict {subdist} - {ages[age]}', size=12, fontweight='bold')
        axs[i // 2, age].set_xlabel('Time (days of season)', size=10, fontweight='bold')
        axs[i // 2, age].set_ylabel('Number of individuals', size=10, fontweight='bold', labelpad=10)

        axs[i // 2, age].set_xticks(ts[::50])
        axs[i // 2, age].tick_params(labelsize=8)

    plt.show()


def plot_fit_by_subdist_with_cloud_vaccination(model_results_list, data_for_fit, prep_data, likelihood_by):
    # Calculate log-likelihood
    if likelihood_by == 'subdist':
        model_res_likelihood_list = [log_likelihood_agg_by_subdist(res['lambdas'], data_for_fit['data_for_fit_subdist'], prep_data)
                                     for res in model_results_list]
    if likelihood_by == 'age':
        model_res_likelihood_list = [log_likelihood_agg_with_age(res, data_for_fit) for res in model_results_list]

    # Get median realization
    med = np.argsort(np.array(model_res_likelihood_list))[len(model_res_likelihood_list)//2]

    # Create fig
    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(15, 30))  # , sharey=True)
    plt.tight_layout(w_pad=3, h_pad=5)
    ages = ['children', 'adults']

    for j, res in enumerate(model_results_list):
        # Aggregate lambda by subdist
        # Initialize dict to all arrays of 0s
        infected_by_subdist = {key: np.array([0] * len(prep_data['day_in_season_short'])) for key in
                               prep_data['relevant_subdists_age']}

        # Go over the clinics and age groups and aggregate according to the clinic's subdist
        for (clinic, age), data in res['infected_by_clinic_age'].items():
            subdist = prep_data['vaccination_coverage_with_age'].loc[clinic].subdist[0]
            infected_by_subdist[(subdist, age)] = infected_by_subdist[(subdist, age)] + data

        # Time steps
        ts = np.arange(list(infected_by_subdist.values())[0].size)

        for i, (subdist, age) in enumerate(infected_by_subdist):
            if j == med:
                # Plot newly infected - model
                axs[i // 2, age].plot(ts, infected_by_subdist[(subdist, age)], label='model', linewidth=1.5)
            else:
                axs[i // 2, age].plot(ts, infected_by_subdist[(subdist, age)], linewidth=0.3, label='model', c='gray', alpha=0.4)

            # Plot data
            axs[i // 2, age].scatter(ts, data_for_fit['data_for_fit_subdist'][(subdist, age)], label='data', c='r', s=3)

            axs[i // 2, age].set_title(f'subdistrict {subdist} - {ages[age]}', size=12, fontweight='bold')
            axs[i // 2, age].set_xlabel('Time (days of season)', size=10, fontweight='bold')
            axs[i // 2, age].set_ylabel('Number of individuals', size=10, fontweight='bold', labelpad=10)

            axs[i // 2, age].set_xticks(ts[::50])
            axs[i // 2, age].tick_params(labelsize=8)

    plt.show()


def plot_vacc_coverage(model_results, prep_data, homogenous=False):
    if not homogenous:
        vacc_prop_gb_subdist = get_vaccination_coverage(model_results, prep_data)

    # If homogenous model
    else:
        # Get vaccination coverage_data
        data_coverage = pd.read_pickle(vaccination_coverage_with_age_path)

        # Group by subdist and calculate mean
        vacc_prop_gb_subdist = data_coverage.reset_index().groupby(['subdist', 'age']).mean()[['data_coverage']]

        # Only relevant subdists
        vacc_prop_gb_subdist = vacc_prop_gb_subdist.loc[product(prep_data['relevant_subdists'], [0, 1])]

        # model coverage - same coverage for all subdist (only by age)
        vacc_prop_gb_subdist['model_coverage'] = \
            np.tile(((model_results['R'][-1] + model_results['I'][-1]) / prep_data['N']), 7)

    # Plot
    vacc_prop_gb_subdist.plot.bar(figsize=(15, 7))
    plt.title('Vaccination Coverage by Subdistrict', size=20)
    plt.xlabel('\nSubdistrict', size=15)
    plt.ylabel('Vaccination coverage', size=15)
    plt.xticks(np.arange(14), vacc_prop_gb_subdist.index, rotation='horizontal', size=14)
    plt.legend(fontsize=15, labels=['data', 'model'], loc=(1.01, 0.87))
    plt.show()


def plot_vacc_coverage_by_age(model_results, data_for_fit, prep_data):
    # Get infection rates
    vaccination_coverage = get_vaccination_coverage_by_age(model_results, data_for_fit, prep_data)

    # Plot
    vaccination_coverage.plot.bar(figsize=(15, 7))
    plt.title('Vaccination Coverage by Age', size=20)
    plt.xlabel('', size=15)
    plt.ylabel('Vaccination coverage', size=15)
    plt.xticks(np.arange(3), vaccination_coverage.index, rotation='horizontal', size=14)
    plt.legend(fontsize=15, labels=['data', 'model'], loc=(1.01, 0.87))
    plt.show()






