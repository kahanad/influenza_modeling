import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import networkx as nx
from itertools import product
import pickle
from scipy.stats import pearsonr
from .vaccination_model_with_age import *

# Influenza paths
vaccination_coverage_influenza_path = '../../Data/influenza_model/data/vaccination_coverage_influenza.pickle'
influenza_diagnoses_by_age_clinic_path = '../../Data/influenza_model/data/influenza_diagnoses_by_age_clinic.pickle'
influenza_weekly_cases_by_clinic_age_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_clinic_age.pickle'
influenza_weekly_cases_by_age_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_age.pickle'
influenza_weekly_cases_by_subdist_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_subdist.pickle'
total_influenza_weekly_cases_path = '../../Data/influenza_model/data/total_influenza_weekly_cases.pickle'
# Short season
influenza_weekly_cases_by_clinic_age_short_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_clinic_age_short.pickle'
influenza_weekly_cases_by_age_short_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_age_short.pickle'
influenza_weekly_cases_by_subdist_short_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_subdist_short.pickle'
total_influenza_weekly_cases_short_path = '../../Data/influenza_model/data/total_influenza_weekly_cases_short.pickle'

# influenza_weekly_cases_by_clinic_age_short_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_clinic_age_short_10.pickle'
# influenza_weekly_cases_by_age_short_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_age_short_10.pickle'
# influenza_weekly_cases_by_subdist_short_path = '../../Data/influenza_model/data/influenza_weekly_cases_by_subdist_short_10.pickle'
# total_influenza_weekly_cases_short_path = '../../Data/influenza_model/data/total_influenza_weekly_cases_short_10.pickle'

# Relevant seasons
# seasons = [2013, 2014, 2015, 2016, 2017]  # 5 SEASONS
seasons = [2011, 2012, 2013, 2014, 2015, 2016, 2017]


def create_data_for_fit_influenza():
    # Initialize a dict
    data_for_fit = {}

    # Load data for fit
    # By clinic and age
    with open(influenza_weekly_cases_by_clinic_age_path, 'rb') as pickle_in:
        data_for_fit['by_clinic_age'] = pickle.load(pickle_in)

    # By age
    with open(influenza_weekly_cases_by_age_path, 'rb') as pickle_in:
        data_for_fit['by_age'] = pickle.load(pickle_in)

    # By subdist and age
    with open(influenza_weekly_cases_by_subdist_path, 'rb') as pickle_in:
        data_for_fit['by_subdist'] = pickle.load(pickle_in)

    # Total
    data_for_fit['total'] = pd.read_pickle(total_influenza_weekly_cases_path)

    # Short season
    # By clinic and age
    with open(influenza_weekly_cases_by_clinic_age_short_path, 'rb') as pickle_in:
        data_for_fit['short_by_clinic_age'] = pickle.load(pickle_in)

    # By age
    with open(influenza_weekly_cases_by_age_short_path, 'rb') as pickle_in:
        data_for_fit['short_by_age'] = pickle.load(pickle_in)

    # By subdist and age
    with open(influenza_weekly_cases_by_subdist_short_path, 'rb') as pickle_in:
        data_for_fit['short_by_subdist'] = pickle.load(pickle_in)

    # Total
    data_for_fit['short_total'] = pd.read_pickle(total_influenza_weekly_cases_short_path)

    return data_for_fit


def initialize_influenza_model(prep_data, short=False, homogenous=False):
    """For using in the coupled model only!"""
    ##############################
    # ----- Initialization ----- #
    ##############################
    # Get prep data
    network, relevant_clinics_age, node_clinic_age = prep_data['network'], prep_data['relevant_clinics_age'],\
                                                     prep_data['node_clinic_age']
    nodes_by_clinic_age = prep_data['nodes_by_clinic_age']
    # vaccination_coverage_influenza = prep_data['vaccination_coverage_influenza']
    population_by_clinic_age = prep_data['population_by_clinic_age']

    # Set season length
    season_length = 274 if short else 365
    # season_length = 244 if short else 365

    # Influenza vaccination efficacy
    vaccination_efficacy = 0.45

    # Cross reactivity rate - by age group TODO: CHECK!!!
    cross_reactivity = [0.2, 0.3]

    # Susceptible - initialize to all nodes
    S_0 = set(network.nodes)

    # Recovered - initialize according to cross reactivity rate
    R_0 = set()
    # Go over each clinic and age and choose nodes to be immunised due to cross reactivity
    for clinic, age in relevant_clinics_age:
        # Get number of immunised nodes
        cur_num_immunised = int(round(population_by_clinic_age.loc[(clinic, age), 'network_population'] * cross_reactivity[age]))
        # Choose nodes to vaccinate
        nodes_to_immunise = np.random.choice(list(nodes_by_clinic_age[(clinic, age)]), size=cur_num_immunised, replace=False)
        # Add to V_0
        R_0 = R_0.union(nodes_to_immunise)

    # Update susceptible
    S_0 = S_0 - R_0

    # Vaccinated
    V_0 = set()

    # Infected - initialize I_0_size of the population
    Is_0 = set()
    Ia_0 = set()

    # Update susceptible
    S_0 = S_0 - Is_0 - Ia_0

    # Initialize lists to save all the states
    S = [S_0]
    V = [V_0]
    Is = [Is_0]
    Ia = [Ia_0]
    R = [R_0]
    Rs = [set()]

    # Initialize a list to save the newly infected - total and by age group
    new_Is = [set()]
    new_Is_by_age = [[set()], [set()]]

    # Initialize a dictionary lambdas_kt
    lambdas = {key: np.array([0.] * season_length) for key in relevant_clinics_age}

    # Initialize an array for lambda_t (aggregated) - total, children, adult
    lambdas_agg_total = np.array([0.] * season_length)
    lambdas_agg_children = np.array([0.] * season_length)
    lambdas_agg_adult = np.array([0.] * season_length)

    # Initialize a dict for infected by clinic and age
    Is_by_clinic_age = {key: np.array([0.] * season_length) for key in relevant_clinics_age}

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

        return {'S': S, 'V': V, 'Is': Is, 'Ia': Ia, 'R': R, 'Rs': Rs, 'new_Is': new_Is,
                'new_Is_by_age': new_Is_by_age, 'lambdas': lambdas,  'lambdas_agg_total': lambdas_agg_total,
                'lambdas_agg_children': lambdas_agg_children, 'lambdas_agg_adult': lambdas_agg_adult,
                'Is_by_clinic_age': Is_by_clinic_age, 'S_by_age': S_by_age, 'S_by_clinic_age': S_by_clinic_age}

    return {'S': S, 'V': V, 'Is': Is, 'Ia': Ia, 'R': R, 'Rs': Rs, 'new_Is': new_Is, 'new_Is_by_age': new_Is_by_age,
            'lambdas': lambdas, 'lambdas_agg_total': lambdas_agg_total, 'lambdas_agg_children': lambdas_agg_children,
            'lambdas_agg_adult': lambdas_agg_adult, 'Is_by_clinic_age': Is_by_clinic_age}


def run_influenza_model(parameters, prep_data, season, short=False, intervention_nodes=set(), homogenous=False):
    ##############################
    # ----- Initialization ----- #
    ##############################
    # Get model parameters
    beta = parameters[season]['beta']
    delta = [parameters[season]['delta'], 1]
    phi = parameters[season]['phi']
    I_0_size = 0  # parameters[season]['I_0_size']
    epsilon = parameters[season]['epsilon']

    # Get prep data
    network, relevant_clinics_age, node_clinic_age = prep_data['network'], prep_data['relevant_clinics_age'],\
                                                     prep_data['node_clinic_age']
    nodes_by_clinic_age = prep_data['nodes_by_clinic_age']
    vaccination_coverage_influenza = prep_data['vaccination_coverage_influenza']
    population_by_clinic_age = prep_data['population_by_clinic_age']

    # Set season length
    season_length = 274 if short else 365
    # season_length = 244 if short else 365

    # Influenza vaccination efficacy
    vaccination_efficacy = 0.45

    # Asymptomatic fraction
    asymp_frac = 0.191

    # Transmissibility - based on log viral load according to infection type (symptomatic/asymptomatic)
    rho_symp = 5.5
    rho_asym = 4.5

    # Cross reactivity rate - by age group TODO: CHECK!!!
    cross_reactivity = [0.2, 0.3]
    # cross_reactivity = [0, 0]

    # Recovery rate - symptomatic
    gamma_symp = [0.1477929674216146, 1/4.5]
    # Recovery rate - asymptomatic
    gamma_asymp = 1/3.2

    # Susceptible - initialize to all nodes
    S_0 = set(network.nodes)

    # Recovered - initialize according to cross reactivity rate
    R_0 = set()
    # Go over each clinic and age and choose nodes to be immunised due to cross reactivity
    for clinic, age in relevant_clinics_age:
        # Get number of immunised nodes
        cur_num_immunised = int(round(population_by_clinic_age.loc[(clinic, age), 'network_population'] * cross_reactivity[age]))
        # Choose nodes to vaccinate
        nodes_to_immunise = np.random.choice(list(nodes_by_clinic_age[(clinic, age)]), size=cur_num_immunised, replace=False)
        # Add to V_0
        R_0 = R_0.union(nodes_to_immunise)

    # Update susceptible
    S_0 = S_0 - R_0

    # Vaccinated
    V_0 = set()
    # Go over each clinic and age and choose nodes to vaccinate according to the relevant vaccination coverage
    for clinic, age in relevant_clinics_age:
        # Get the relevant vaccination coverage for the clinic, age and season
        cur_vacc_coverage = vaccination_coverage_influenza.loc[(clinic, age, season), 'data_coverage']
        # Get number of vaccinated nodes
        cur_num_vaccinated = int(round(population_by_clinic_age.loc[(clinic, age), 'network_population'] * cur_vacc_coverage))
        # Choose nodes to vaccinate
        nodes_to_vaccinate = np.random.choice(list(nodes_by_clinic_age[(clinic, age)]), size=cur_num_vaccinated, replace=False)
        # Choose effective vaccination nodes
        vaccinated_nodes = np.random.choice(nodes_to_vaccinate, int(round(vaccination_efficacy*len(nodes_to_vaccinate))))
        # Add to V_0
        V_0 = V_0.union(vaccinated_nodes)

    # Add intervention vaccinated_nodes
    if len(intervention_nodes) > 0:
        # Choose effective vaccination nodes
        vaccinated_intervention_nodes = np.random.choice(list(intervention_nodes), int(round(vaccination_efficacy*len(intervention_nodes))))
        # Update V_0
        V_0 = V_0.union(vaccinated_intervention_nodes)

    # Update susceptible
    S_0 = S_0 - V_0

    # Infected - initialize I_0_size of the population
    I_0_num_of_nodes = int(round(len(S_0)*I_0_size))
    Is_0 = set(np.random.choice(list(S_0), replace=False, size=int(round(I_0_num_of_nodes*(1-asymp_frac)))))
    Ia_0 = set(np.random.choice(list(S_0 - Is_0), replace=False, size=int(round(I_0_num_of_nodes*asymp_frac))))

    # Update susceptible
    S_0 = S_0 - Is_0 - Ia_0

    # Initialize lists to save all the states
    S = [S_0]
    # V = [V_0]
    Is = [Is_0]
    Ia = [Ia_0]
    R = [R_0]
    Rs = [set()]

    # Initialize a list to save the newly infected - total and by age group
    new_Is = [set()]
    new_Is_by_age = [[set()], [set()]]

    # Initialize a dictionary lambdas_kt
    lambdas = {key: np.array([0.] * season_length) for key in relevant_clinics_age}

    # Initialize an array for lambda_t (aggregated) - total, children, adult
    lambdas_agg_total = np.array([0.] * season_length)
    lambdas_agg_children = np.array([0.] * season_length)
    lambdas_agg_adult = np.array([0.] * season_length)

    # Initialize a dict for infected by clinic and age
    Is_by_clinic_age = {key: np.array([0.] * season_length) for key in relevant_clinics_age}

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
        new_Ia_t = set()
        new_Is_t = set()
        new_Is_by_age_t = [set(), set()]
        new_R_t = set()
        new_Rs_t = set()

        # --- Infection from friends --- #
        # Go over the infected individuals
        if not homogenous:
            for i, node in enumerate(list(Is[-1]) + list(Ia[-1])):
                for contact in network[node]:
                    # If the contact is susceptible and not exposed in this stage yet
                    if contact in S[-1] and contact not in new_Is_t and contact not in new_Ia_t:
                        # Get age and clinic
                        cur_clinic, cur_age = node_clinic_age[contact]
                        # Get relevant transmissability (rho)
                        rho = rho_symp if i < len(Is[-1]) else rho_asym
                        # Force of infection
                        force_of_infection = beta * delta[cur_age] * rho * (1 + np.cos((2 * np.pi * t) / 365 + phi))
                        # Contact is infected with probability according to the force of infection
                        if np.random.rand() < force_of_infection:
                            # Generate infection type
                            infection_type = 'asymptomatic' if np.random.rand() < asymp_frac else 'symptomatic'
                            if infection_type == 'symptomatic':
                                new_Is_t.add(contact)
                                new_Is_by_age_t[cur_age].add(contact)
                                Is_by_clinic_age[(cur_clinic, cur_age)][t] += 1
                            elif infection_type == 'asymptomatic':
                                new_Ia_t.add(contact)
                        # Update lambda_kt
                        lambdas[(cur_clinic, cur_age)][t] += force_of_infection * (1-asymp_frac)
                        lambdas_agg_total[t] += force_of_infection * (1-asymp_frac)
                        if cur_age == 0:
                            lambdas_agg_children[t] += force_of_infection * (1-asymp_frac)
                        else:
                            lambdas_agg_adult[t] += force_of_infection * (1-asymp_frac)

            # Add "epsilon" - constant infection of 1 random node each day
            if len(S[-1] - new_Is_t - new_Ia_t) <= epsilon:
                epsilon_nodes = S[-1] - new_Is_t - new_Ia_t
            else:
                epsilon_nodes = set(np.random.choice(list(S[-1] - new_Is_t - new_Ia_t), epsilon, replace=False))
            for i, node in enumerate(epsilon_nodes):
                # Get age and clinic
                cur_clinic, cur_age = node_clinic_age[node]
                if i < 4:
                    new_Is_t.add(node)
                    new_Is_by_age_t[cur_age].add(node)
                else:
                    new_Ia_t.add(node)

        else:  # If homogenous
            # Get number of nodes to infect (for each age group
            num_of_infected_s = np.zeros(2)
            num_of_infected_a = np.zeros(2)
            for i, node in enumerate(list(Is[-1]) + list(Ia[-1])):
                # Get age of infected individual
                cur_age = node_clinic_age[node][1]
                # Get relevant transmissability (rho)
                rho = rho_symp if i < len(Is[-1]) else rho_asym
                # Force of infection - for children/adult - based on the contacted node
                force_of_infection_children = beta * delta[0] * rho * (1 + np.cos((2 * np.pi * t) / 365 + phi))
                force_of_infection_adult = beta * delta[1] * rho * (1 + np.cos((2 * np.pi * t) / 365 + phi))

                # Update lambdas
                # Calculate lambdas by clinic and age - by current infected individual
                # to all susceptible individuals (by clinic and age)
                for contact_clinic, contact_age in lambdas:
                    lambdas[(contact_clinic, contact_age)][t] += \
                        C[cur_age, contact_age]*beta*delta[contact_age]*len(S_by_clinic_age[(contact_clinic, contact_age)])

                # Calculate lambda for children and adults (from current infected individual)
                cur_lambda_children = C[cur_age, 0] * force_of_infection_children*len(S_by_age[0]) * (1-asymp_frac)
                cur_lambda_adult = C[cur_age, 1] * force_of_infection_adult *len(S_by_age[1]) * (1-asymp_frac)
                # Update total lambda for current iteration
                lambdas_agg_children[t] += cur_lambda_children
                lambdas_agg_adult[t] += cur_lambda_adult

                # Generate number of infected form current individual from each age group (form poisson with lambda)
                # Symptomatic
                cur_num_of_infected_s = np.array([np.random.poisson(C[cur_age, 0]*force_of_infection_children*len(S_by_age[0])*(1-asymp_frac)),
                                                np.random.poisson(C[cur_age, 1]*force_of_infection_adult*len(S_by_age[1])*(1-asymp_frac))])
                num_of_infected_s += cur_num_of_infected_s
                # Asymptomatic
                cur_num_of_infected_a = np.array([np.random.poisson(C[cur_age, 0]*force_of_infection_children*len(S_by_age[0])*asymp_frac),
                                                np.random.poisson(C[cur_age, 1]*force_of_infection_adult*len(S_by_age[1])*asymp_frac)])
                num_of_infected_a += cur_num_of_infected_a

            # Choose nodes to infect (randomly) - make sure not to sample more the number of susceptible
            # Symptomatic
            if len(S_by_age[0]) == 0:
                new_Is_by_age_t[0] = set()
            else:
                new_Is_by_age_t[0] = set(np.random.choice(list(S_by_age[0]), size=min(len(S_by_age[0]), int(num_of_infected_s[0])), replace=False))

            if len(S_by_age[1]) == 0:
                new_Is_by_age_t[1] = set()
            else:
                new_Is_by_age_t[1] = set(np.random.choice(list(S_by_age[1]), size=min(len(S_by_age[1]), int(num_of_infected_s[1])), replace=False))

            # Asymptomatic
            if len(S_by_age[0]) == 0:
                new_Ia_children = set()
            else:
                new_Ia_children = set(np.random.choice(list(S_by_age[0] - new_Is_by_age_t[0]), size=min(len(S_by_age[0]), int(num_of_infected_a[0])), replace=False))

            if len(S_by_age[1]) == 0:
                new_Ia_adults = set()
            else:
                new_Ia_adults = set(np.random.choice(list(S_by_age[1] - new_Is_by_age_t[1]), size=min(len(S_by_age[1]), int(num_of_infected_a[1])), replace=False))

            # Infect nodes
            new_Is_t = new_Is_by_age_t[0].union(new_Is_by_age_t[1])
            new_Ia_t = new_Ia_children.union(new_Ia_adults)

            # Add "epsilon" - constant infection of 1 random node each day
            if len(S[-1] - new_Is_t - new_Ia_t) <= epsilon:
                epsilon_nodes = S[-1] - new_Is_t - new_Ia_t
            else:
                epsilon_nodes = set(np.random.choice(list(S[-1] - new_Is_t - new_Ia_t), epsilon, replace=False))
            for i, node in enumerate(epsilon_nodes):
                # Get age and clinic
                cur_clinic, cur_age = node_clinic_age[node]
                if i < 4:
                    new_Is_t.add(node)
                    new_Is_by_age_t[cur_age].add(node)
                else:
                    new_Ia_t.add(node)
                    new_Ia_children.add(node) if cur_age == 0 else new_Ia_adults.add(node)

            # # Update infected by clinic and age
            # for node in new_Is_t:
            #     cur_clinic, cur_age = node_clinic_age[node]
            #     Is_by_clinic_age[(cur_clinic, cur_age)][t] += 1

            # Update S_by_age
            S_by_age[0] = S_by_age[0] - new_Is_by_age_t[0] - new_Ia_children
            S_by_age[1] = S_by_age[1] - new_Is_by_age_t[1] - new_Ia_adults

            # Update S_by_clinic_age (remove infected)
            for i, node in enumerate(list(new_Is_t) + list(new_Ia_t)):
                cur_clinic, cur_age = node_clinic_age[node]
                S_by_clinic_age[(cur_clinic, cur_age)].remove(node)
                if i < len(new_Is_t):
                    # Update new infected by clinic and age
                    Is_by_clinic_age[(cur_clinic, cur_age)][t] += 1

        # Transmission from I to R
        for i, node in enumerate(list(Is[-1]) + list(Ia[-1])):
            # Get infection type
            infection_type = 'symptomatic' if i < len(Is[-1]) else 'asymptomatic'
            # Get individual's age
            cur_age = node_clinic_age[node][1]
            # Get relevant gamma according to infection type and individual age
            if infection_type == 'symptomatic':
                gamma = gamma_symp[cur_age]
            elif infection_type == 'asymptomatic':
                gamma = gamma_asymp
            # Individuals transmitted from I to R with probability gamma - according to infection type and age
            if np.random.rand() < gamma:
                new_R_t.add(node)
                if infection_type == 'symptomatic':
                    new_Rs_t.add(node)

        # Update stages
        S.append(S[-1] - new_Is_t - new_Ia_t)
        Is.append(Is[-1].union(new_Is_t) - new_R_t)
        Ia.append(Ia[-1].union(new_Ia_t) - new_R_t)
        R.append(R[-1].union(new_R_t))
        Rs.append(Rs[-1].union(new_Rs_t))

        # Save the newly infected
        new_Is.append(new_Is_t)
        new_Is_by_age[0].append(new_Is_by_age_t[0])
        new_Is_by_age[1].append(new_Is_by_age_t[1])

    # Save results to dictionary
    model_results = {'S': S, 'V_0': V_0, 'Is': Is, 'Ia': Ia, 'R': R, 'Rs': Rs, 'new_Is': new_Is, 'new_Is_by_age': new_Is_by_age, 'lambdas': lambdas,
                     'lambdas_agg_total': lambdas_agg_total, 'lambdas_agg_children': lambdas_agg_children, 'lambdas_agg_adult': lambdas_agg_adult,
                     'parameters': parameters, 'N': network.number_of_nodes(),  'Is_by_clinic_age': Is_by_clinic_age}

    return model_results


############################
# --- Helper Functions --- #
############################
def get_model_weekly_cases(model_results, season, short=False, age=None, by_subdist=False, by_subdist_age=False):
    """Receives model results dictionary and returns the number of weekly cases aggregated weekly,
    by age group - for a specific season"""
    # Relevant dates
    season_length = 274 if short else 365
    # season_length = 244 if short else 365
    start_month = 9 if short else 6
    # start_month = 10 if short else 6
    dates = [pd.Timestamp(season - 1, start_month, 1) + pd.Timedelta(i, unit='d') for i in range(season_length)]

    if not by_subdist and not by_subdist_age:
        # Get the model new symptomatic cases by season according to age
        if age == 0:
            model_cases_nodes = model_results['new_Is_by_age'][0]
        elif age == 1:
            model_cases_nodes = model_results['new_Is_by_age'][1]
        else:  # Total
            model_cases_nodes = model_results['new_Is']

        # Get number of cases by day
        model_cases_by_day = np.array([len(st) for st in model_cases_nodes[1:]])

    else:  # by subdist
        model_cases_by_day = model_results.copy()

    # Create a DF of the cases with the dates as index
    model_cases_df = pd.DataFrame(model_cases_by_day, index=np.array(dates), columns=['cases'])

    # Aggregate weekly
    model_weekly_cases = model_cases_df.resample('W').sum().fillna(0).copy()

    # return model_weekly_cases[:-1]  # TODO: CHECK IF [:-1] IS NEEDED
    return model_weekly_cases[:-1] if season != 2015 else model_weekly_cases


def get_model_weekly_cases_all_seasons(model_results_list, prep_data, age=None, by_subdist=False, by_subdist_age=False):
    """Receives model results dictionary and returns the number of weekly cases aggregated weekly,
    by age group - for all seasons"""
    # Relevant dates
    season_length = 365
    start_month = 6

    if not by_subdist and not by_subdist_age:
        # Initialize a list for weekly cases for all seasons
        model_weekly_cases_list = []
        # Go over all seasons
        for s, model_results in enumerate(model_results_list):
            # Get the model new symptomatic cases by season according to age
            if age == 0:
                model_cases_nodes = model_results['new_Is_by_age'][0]
            elif age == 1:
                model_cases_nodes = model_results['new_Is_by_age'][1]
            else:  # Total
                model_cases_nodes = model_results['new_Is']

            # Get number of cases by day
            model_cases_by_day = np.array([len(st) for st in model_cases_nodes[1:]])

            # Get relevant dates
            dates = [pd.Timestamp(seasons[s] - 1, start_month, 1) + pd.Timedelta(i, unit='d') for i in range(season_length)]

            # Create a DF of the cases with the dates as index
            model_cases_df = pd.DataFrame(model_cases_by_day, index=np.array(dates), columns=['cases'])

            # Aggregate weekly
            model_weekly_cases = model_cases_df.resample('W').sum().fillna(0).copy()

            # Add to the list
            model_weekly_cases = model_weekly_cases[:-1] if seasons[s] != 2015 else model_weekly_cases
            model_weekly_cases_list.append(model_weekly_cases)

        return pd.concat(model_weekly_cases_list)

    elif by_subdist:  # by subdist
        # Initialize a list for weekly cases for all seasons
        model_weekly_cases_by_subdist_lists = {subdist: [] for subdist in prep_data['relevant_subdists']}

        # Go over all seasons
        for s, model_results in enumerate(model_results_list):
            Is_by_subdist = {subdist: np.array([0] * season_length) for subdist in prep_data['relevant_subdists']}
            for (clinic, age), data in model_results['Is_by_clinic_age'].items():
                subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
                Is_by_subdist[subdist] = Is_by_subdist[subdist] + data

            # Get relevant dates
            dates = [pd.Timestamp(seasons[s] - 1, start_month, 1) + pd.Timedelta(i, unit='d') for i in range(season_length)]

            # Go over subdists, aggregate weekly and save to the list of all seasons
            for subdist, subdist_cases_by_day in Is_by_subdist.items():
                # Create a df of the cases with the dates as index
                model_cases_df = pd.DataFrame(subdist_cases_by_day, index=np.array(dates), columns=['cases'])

                # Aggregate weekly
                model_weekly_cases = model_cases_df.resample('W').sum().fillna(0).copy()

                # Add to the list
                model_weekly_cases = model_weekly_cases[:-1] if seasons[s] != 2015 else model_weekly_cases
                model_weekly_cases_by_subdist_lists[subdist].append(model_weekly_cases)

        # Concat lists and return
        return {subdist: pd.concat(lst) for subdist, lst in model_weekly_cases_by_subdist_lists.items()}

    else:  # by subdist_age
        # Initialize a list for weekly cases for all seasons
        model_weekly_cases_by_subdist_age_lists = {(subdist, age): [] for subdist, age in prep_data['relevant_subdists_age']}

        # Go over all seasons
        for s, model_results in enumerate(model_results_list):
            Is_by_subdist_age = {(subdist, age): np.array([0] * season_length) for subdist, age in prep_data['relevant_subdists_age']}
            for (clinic, age), data in model_results['Is_by_clinic_age'].items():
                subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
                Is_by_subdist_age[(subdist, age)] = Is_by_subdist_age[(subdist, age)] + data

            # Get relevant dates
            dates = [pd.Timestamp(seasons[s] - 1, start_month, 1) + pd.Timedelta(i, unit='d') for i in range(season_length)]

            # Go over subdists, aggregate weekly and save to the list of all seasons
            for (subdist, age), subdist_age_cases_by_day in Is_by_subdist_age.items():
                # Create a df of the cases with the dates as index
                model_cases_df = pd.DataFrame(subdist_age_cases_by_day, index=np.array(dates), columns=['cases'])

                # Aggregate weekly
                model_weekly_cases = model_cases_df.resample('W').sum().fillna(0).copy()

                # Add to the list
                model_weekly_cases = model_weekly_cases[:-1] if seasons[s] != 2015 else model_weekly_cases
                model_weekly_cases_by_subdist_age_lists[(subdist, age)].append(model_weekly_cases)

        # Concat lists and return
        return {(subdist, age): pd.concat(lst) for (subdist,age), lst in model_weekly_cases_by_subdist_age_lists.items()}


def get_infection_rate_by_clinic(model_results, prep_data):
    # Get prep data
    network, relevant_clinics_age, population_by_clinic_age = prep_data['network'], prep_data['relevant_clinics_age'],\
                                                              prep_data['population_by_clinic_age']

    # Initialize a dictionary to save the vaccination coverage by clinic
    infection_rate_by_clinic_age = dict.fromkeys(relevant_clinics_age, 0)

    # Get infected nodes
    infected_nodes = model_results['Rs'][-1]

    # Go over clinics and count the number of infected
    for node in infected_nodes:
        node_clinic = network.nodes[node]['clinic']
        node_age = network.nodes[node]['age']
        infection_rate_by_clinic_age[(node_clinic, node_age)] += 1

    # Normalize by clinic network population to receive the coverage %
    for clinic, age in infection_rate_by_clinic_age:
        infection_rate_by_clinic_age[(clinic, age)] /= population_by_clinic_age.loc[(clinic, age)].network_population

    return infection_rate_by_clinic_age


def get_infection_rate_by_age(model_results, data_for_fit, prep_data, season):
    # Get prep data
    network, relevant_clinics_age, population_by_clinic_age = prep_data['network'], prep_data['relevant_clinics_age'], \
                                                              prep_data['population_by_clinic_age']

    # Initialize a dictionary to save the vaccination coverage by clinic
    model_infection_rates = dict.fromkeys([0, 1], 0)

    # Get symptomatically infected nodes
    infected_nodes = model_results['Rs'][-1].union(model_results['Is'][-1])

    # Go over clinics and count the number of infected
    for node in infected_nodes:
        node_age = network.nodes[node]['age']
        model_infection_rates[node_age] += 1

    # Add total
    model_infection_rates['total'] = model_infection_rates[0] + model_infection_rates[1]

    # Normalize by clinic network population to receive the coverage %
    for age in [0, 1]:
        model_infection_rates[age] /= population_by_clinic_age.loc[pd.IndexSlice[:, age], 'network_population'].sum()

    # Normalize total
    model_infection_rates['total'] /= prep_data['N']

    # Data infection rate
    chidren_infection_rate = data_for_fit['by_age'][0][data_for_fit['by_age'][0].season == season].cases.sum() / \
                             prep_data['population_by_clinic_age'].loc[pd.IndexSlice[:, 0], 'network_population'].sum()
    adult_infection_rate = data_for_fit['by_age'][1][data_for_fit['by_age'][1].season == season].cases.sum() / \
                           prep_data['population_by_clinic_age'].loc[pd.IndexSlice[:, 1], 'network_population'].sum()
    total_infection_rate = data_for_fit['total'][data_for_fit['total'].season == season].cases.sum() / prep_data['N']

    data_infection_rates = [chidren_infection_rate, adult_infection_rate, total_infection_rate]

    # Save to df
    infection_rates = pd.DataFrame(columns=['data', 'model'])
    infection_rates['data'] = data_infection_rates
    infection_rates['model'] = [model_infection_rates[0], model_infection_rates[1], model_infection_rates['total']]
    infection_rates['age'] = ['children', 'adults', 'total']
    infection_rates.set_index('age', inplace=True)

    return infection_rates


def calc_infection_rates(model_results, prep_data, season, by_clinic=False):
    # Get model infection rate by clinic and age
    model_infection_rate = get_infection_rate_by_clinic(model_results, prep_data)
    model_infection_rate = pd.DataFrame(pd.Series(model_infection_rate), columns=['model_infection_rate'])
    model_infection_rate['clinic_code'] = model_infection_rate.index.map(lambda x: x[0])
    model_infection_rate['age'] = model_infection_rate.index.map(lambda x: x[1])
    model_infection_rate.set_index(['clinic_code', 'age'], inplace=True)

    # Get influenza infection diagnoses data and merge with model data
    data_infection_rate = prep_data['infection_rates'][season].copy()
    data_infection_rate = data_infection_rate.merge(model_infection_rate, left_index=True, right_index=True)

    # If by_clinic is True, return infection rates in model and data by clinic
    if by_clinic:
        data_infection_rate = data_infection_rate[['infection_rate', 'model_infection_rate']].copy()
        data_infection_rate.columns = ['data_infection_rate', 'model_infection_rate']
        return data_infection_rate

    # Multiply vaccination coverage by proportion out of subdist for weighted average calculation
    data_infection_rate['data_mul'] = data_infection_rate['infection_rate'] * data_infection_rate['pop_prop']
    data_infection_rate['net_mul'] = data_infection_rate['model_infection_rate'] * data_infection_rate['pop_prop']

    # Group by subdist and calculate mean
    infection_gb_subdist = data_infection_rate.reset_index().groupby(['subdist', 'age']).sum()[
        ['data_mul', 'net_mul']]
    infection_gb_subdist.columns = ['data_infection_rate', 'model_infection_rate']

    return infection_gb_subdist


#############################
# ---------- Fit ---------- #
#############################
def calc_weekly_lambdas(lambdas, season):
    # Get season length
    short = len(lambdas) < 365
    # Relevant dates
    season_length = 274 if short else 365
    # season_length = 244 if short else 365
    start_month = 9 if short else 6
    # start_month = 10 if short else 6
    dates = [pd.Timestamp(season - 1, start_month, 1) + pd.Timedelta(i, unit='d') for i in range(season_length)]

    # Create a DF of the lambdas with the dates as index
    lambdas_df = pd.DataFrame(lambdas, index=np.array(dates), columns=['lambdas'])

    # Aggregate weekly
    weekly_lambdas = lambdas_df.resample('W').sum().fillna(0).copy()
    return weekly_lambdas.values.flatten()[:-1] if season != 2015 else weekly_lambdas.values.flatten()


def log_likelihood_influenza(lambdas, data_for_fit, season):
    # Initialize a variable to sum the log-likelihood
    log_like = 0

    # Go over the clinics
    for (clinic, age), cur_lambdas in lambdas.items():
        # Get weekly lambdas
        weekly_lambdas = calc_weekly_lambdas(cur_lambdas, season)
        # Get relevant data for fit
        cur_data_for_fit = data_for_fit[(clinic, age)][data_for_fit[(clinic, age)].season == season].cases.values
        # Sum the log-likelihood for each stage
        log_like += np.sum(-weekly_lambdas + 1e-10 + cur_data_for_fit * np.log(weekly_lambdas + 1e-10))

    return log_like


def log_likelihood_agg_by_subdist_influenza(lambdas, data_for_fit, season, prep_data):
    relevant_subdists_age = prep_data['relevant_subdists_age']
    vaccination_coverage_with_age = prep_data['vaccination_coverage_with_age']

    # Get season length
    short = len(list(lambdas.values())[0]) < 365

    # Aggregate lambda and by subdist
    # Initialize dict to all arrays of 0s
    season_len_weeks = list(data_for_fit.values())[0][list(data_for_fit.values())[0].season == season].shape[0]
    lambdas_subdist = {key: np.array([0] * season_len_weeks) for key in relevant_subdists_age}

    # Go over the clinics and age groups and aggregate according to the clinic's subdist
    for (clinic, age), cur_data in lambdas.items():
        # Get current subdist
        subdist = vaccination_coverage_with_age.loc[clinic].subdist[0]
        # Get weekly lambdas
        weekly_lambdas = calc_weekly_lambdas(cur_data, season)
        # Sum
        lambdas_subdist[(subdist, age)] = lambdas_subdist[(subdist, age)] + weekly_lambdas

    # Initialize a variable to sum the log-likelihood
    log_like = 0

    # Go over the clinics
    for subdist, age in lambdas_subdist:
        # Get relevant data for fit
        cur_data_for_fit = data_for_fit[(subdist, age)][data_for_fit[(subdist, age)].season == season].cases.values
        # Sum the log-likelihood for each stage
        log_like += np.sum(-lambdas_subdist[(subdist, age)] + 1e-10 + cur_data_for_fit * np.log(lambdas_subdist[(subdist, age)] + 1e-10))

    return log_like


def log_likelihood_agg_influenza(lambdas_agg, data_for_fit_agg, season):
    """Receives aggregated lambdas (total/adult/children) and returns the log likelihood"""
    # Get season length
    short = len(lambdas_agg) < 365

    # Calc weekly lambdas
    weekly_lambdas = calc_weekly_lambdas(lambdas_agg, season)

    # Get relevant data for fit
    cur_data_for_fit = data_for_fit_agg[data_for_fit_agg.season == season].cases.values

    # Sum the log-likelihood for each stage
    log_like = np.sum(-weekly_lambdas + 1e-10 + cur_data_for_fit * np.log(weekly_lambdas + 1e-10))

    return log_like


def log_likelihood_agg_age_influenza(model_results, data_for_fit, season):
    """Receives model results and data for fit, calculates the log likelihood for children and adult separately
    and reutrns the sum of these log likelihood"""
    log_like_agg_children = log_likelihood_agg_influenza(model_results['lambdas_agg_children'],
                                                         data_for_fit['by_age'][0], season)
    log_like_agg_adult = log_likelihood_agg_influenza(model_results['lambdas_agg_adult'],
                                                      data_for_fit['by_age'][1], season)

    return log_like_agg_children + log_like_agg_adult


def calc_correlation_fit_flu(model_results, data_for_fit, prep_data, season, by_subdist=False, by_subdist_age=False, weighted=False, smooth=False,
                             window=None):
    # If aggregated correlation fit
    if by_subdist_age:
        # Initialize dict for weekly cases by subdist - all arrays of 0s
        Is_by_subdist = {subdist: np.array([0] * 365) for subdist in prep_data['relevant_subdists_age']}

        # Go over clinics and aggregate by subdists
        for (clinic, age), data in model_results['Is_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            Is_by_subdist[(subdist, age)] = Is_by_subdist[(subdist, age)] + data

        # Get weekly cases for each subdist
        model_weekly_cases_by_subdist = {}
        for subdist, data in Is_by_subdist.items():
            model_weekly_cases_by_subdist[subdist] = get_model_weekly_cases(data, season, short=False, by_subdist=True)

        # Go over subdists and age groups and calculate correlation
        corrs = {}
        pvals = {}
        for subdist, age in prep_data['relevant_subdists_age']:
            # Model weekly vaccination for current subdist and age
            model_weekly_cases = model_weekly_cases_by_subdist[(subdist, age)].cases.values

            # Data weekly vaccination for current subdsit and age
            data_weekly_cases = data_for_fit['by_subdist'][(subdist, age)]

            # Get only relevant season
            data_weekly_cases = data_weekly_cases[data_weekly_cases.season == season].cases.values.copy()

            # Smooth
            if smooth:
                # Smooth model
                model_weekly_cases = pd.DataFrame(np.concatenate([[0] * (window - 1), model_weekly_cases]))
                model_weekly_cases = model_weekly_cases.rolling(window).mean()[0].values[window - 1:]
                # Smooth data
                data_weekly_cases = pd.DataFrame(np.concatenate([[0] * (window - 1), data_weekly_cases]))
                data_weekly_cases = data_weekly_cases.rolling(window).mean()[0].values[window - 1:]

            # Calculate correlation for current subdist
            # corrs[(subdist, age)] = np.corrcoef(data_weekly_cases, model_weekly_cases)[0, 1]
            corrs[(subdist, age)] = pearsonr(data_weekly_cases, model_weekly_cases)[0]
            pvals[(subdist, age)] = pearsonr(data_weekly_cases, model_weekly_cases)[1]

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
        # Initialize dict for weekly cases by subdist - all arrays of 0s
        Is_by_subdist = {subdist: np.array([0] * 365) for subdist in prep_data['relevant_subdists']}

        # Go over clinics and aggregate by subdists
        for (clinic, age), data in model_results['Is_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            Is_by_subdist[subdist] = Is_by_subdist[subdist] + data

        # Get weekly cases for each subdist
        model_weekly_cases_by_subdist = {}
        for subdist, data in Is_by_subdist.items():
            model_weekly_cases_by_subdist[subdist] = get_model_weekly_cases(data, season, short=False, by_subdist=True)

        # Go over subdists and age groups and calculate correlation
        corrs = {}
        pvals = {}
        for subdist, age in prep_data['relevant_subdists_age']:
            # Model weekly vaccination for current subdist and age
            model_weekly_cases = model_weekly_cases_by_subdist[subdist].cases.values

            # Data weekly vaccination for current subdsit and age
            data_weekly_cases = data_for_fit['by_subdist'][(subdist, 0)].copy()
            data_weekly_cases.cases += data_for_fit['by_subdist'][(subdist, 1)].cases

            # Get only relevant season
            data_weekly_cases = data_weekly_cases[data_weekly_cases.season == season].cases.values.copy()

            # Smooth
            if smooth:
                # Smooth model
                model_weekly_cases = pd.DataFrame(np.concatenate([[0] * (window - 1), model_weekly_cases]))
                model_weekly_cases = model_weekly_cases.rolling(window).mean()[0].values[window - 1:]
                # Smooth data
                data_weekly_cases = pd.DataFrame(np.concatenate([[0] * (window - 1), data_weekly_cases]))
                data_weekly_cases = data_weekly_cases.rolling(window).mean()[0].values[window - 1:]

            # Calculate correlation for current subdist
            # corrs[subdist] = np.corrcoef(data_weekly_cases, model_weekly_cases)[0, 1]
            corrs[subdist] = pearsonr(data_weekly_cases, model_weekly_cases)[0]
            pvals[subdist] = pearsonr(data_weekly_cases, model_weekly_cases)[1]

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
        model_weekly_cases = get_model_weekly_cases(model_results, season).cases.values

        # Get influenza data weekly cases
        data_weekly_cases = data_for_fit['total'][data_for_fit['total'].season == season].cases.values.copy()

        # Smooth
        if smooth:
            # Smooth model
            model_weekly_cases = pd.DataFrame(np.concatenate([[0] * (window - 1), model_weekly_cases]))
            model_weekly_cases = model_weekly_cases.rolling(window).mean()[0].values[window - 1:]
            # Smooth data
            data_weekly_cases = pd.DataFrame(np.concatenate([[0] * (window - 1), data_weekly_cases]))
            data_weekly_cases = data_weekly_cases.rolling(window).mean()[0].values[window - 1:]

        # Calculate correlation
        return pearsonr(model_weekly_cases, data_weekly_cases)


def infection_rate_error(model_results, prep_data, data_for_fit, season, mse=True, weighted=False, by_clinic=False, by_age=False):
    # Get infection rates by subdist, by age or by clinic
    if by_clinic:
        infection_rates = calc_infection_rates(model_results, prep_data, season, by_clinic=True)

    elif by_age:
        infection_rates = get_infection_rate_by_age(model_results, data_for_fit, prep_data, season)
        infection_rates = infection_rates.iloc[:2].copy()
        infection_rates.columns = ['data_infection_rate', 'model_infection_rate']

    else:  # By subdist
        infection_rates = calc_infection_rates(model_results, prep_data, season)

    # Calculate the model's error
    infection_rates['error'] = infection_rates.data_infection_rate - infection_rates.model_infection_rate

    if weighted:  # Calculate weighted MSE of data and model infection rates
        # Add population proportion
        if by_clinic:
            infection_rates['pop_prop'] = infection_rates.apply(lambda row:
                                                                prep_data['population_by_clinic_age'].loc[row.name[0]].prop_network.sum(), axis=1)

        elif by_age:
            infection_rates['pop_prop'] = [prep_data['population_by_clinic_age'].loc[pd.IndexSlice[:, age], 'network_population'].sum() / prep_data['N']
                                           for age in [0, 1]]

        else:  # If by subdist
            pop_prop_by_subdist = prep_data['population_by_clinic_age'].reset_index().groupby(['subdist', 'age']).sum().prop_network
            infection_rates['pop_prop'] = infection_rates.index.map(lambda x: pop_prop_by_subdist.loc[x])

        # Calculate weighted MSE or MAD
        if mse:
            infection_weighted_mse = ((infection_rates.error ** 2) * infection_rates.pop_prop).mean()
            return infection_weighted_mse

        else:  # MAD
            infection_weighted_mad = ((infection_rates.error.abs()) * infection_rates.pop_prop).mean()
            return infection_weighted_mad

    else:  # Calculate regular MSE or MAD
        # Calculate MSE of data and model infection rates
        if mse:
            infection_mse = (infection_rates.error ** 2).mean()
            return infection_mse

        else:  # MAD
            infection_mad = (infection_rates.error.abs()).mean()
            return infection_mad


def model_comparison_influenza(results, data_for_fit, prep_data, season, measures):
    # Create a df for the errors
    errors = pd.DataFrame(index=['network', 'homogenous'])

    if 'MSE' in measures:
        # Calculate errors
        # MSE
        errors['MSE by age'] = [infection_rate_error(res, prep_data, data_for_fit, season=season, by_age=True)
                                for res in results]
        errors['MSE by subdist'] = [infection_rate_error(res, prep_data, data_for_fit, season=season) for res in results]
        errors['MSE by clinic'] = [infection_rate_error(res, prep_data, data_for_fit, season=season, by_clinic=True)
                                   for res in results]
    # Weighted MSE
    if 'Weighted MSE' in measures:
        errors['Weighted MSE by age'] = [infection_rate_error(res, prep_data, data_for_fit, season=season, weighted=True, by_age=True)
                                         for res in results]
        errors['Weighted MSE by subdist'] = [infection_rate_error(res, prep_data, data_for_fit, season=season, weighted=True)
                                             for res in results]
        errors['Weighted MSE by clinic'] = [infection_rate_error(res, prep_data, data_for_fit, season=season, weighted=True, by_clinic=True)
                                            for res in results]

    # MAD
    if 'MAD' in measures:
        errors['MAD by age'] = [infection_rate_error(res, prep_data, data_for_fit, season=season, mse=False, by_age=True)
                                for res in results]
        errors['MAD by subdist'] = [infection_rate_error(res, prep_data, data_for_fit, mse=False, season=season)
                                    for res in results]
        errors['MAD by clinic'] = [infection_rate_error(res, prep_data, data_for_fit, season=season, mse=False, by_clinic=True)
                                   for res in results]

    # Weighted MAD
    if 'Weighted MAD' in measures:
        errors['Weighted MAD by age'] = [infection_rate_error(res, prep_data, data_for_fit, season=season, mse=False, weighted=True, by_age=True)
                                     for res in results]
        errors['Weighted MAD by subdist'] = [infection_rate_error(res, prep_data, data_for_fit, season=season, mse=False, weighted=True)
                                             for res in results]
        errors['Weighted MAD by clinic'] = [infection_rate_error(res, prep_data, data_for_fit, season=season, mse=False, weighted=True, by_clinic=True)
                                            for res in results]

    return errors.T


############################
# ----- Intervention ----- #
############################
def infection_rate_intervention_influenza(intervention_nodes, parameters, prep_data, season, num_of_simulations=5):
    # Run the model num_of_simulations times
    model_results_list = [run_influenza_model(parameters, prep_data, season, intervention_nodes=intervention_nodes)
                          for i in range(num_of_simulations)]

    # Calculate infection rate
    model_res_infection_rate_list = [len(res['R'][-1] - res['R'][0]) / res['N']
                                     for res in model_results_list]

    # Get median realization
    med = np.argsort(np.array(model_res_infection_rate_list))[len(model_res_infection_rate_list) // 2]

    return model_results_list[med], model_res_infection_rate_list[med]


###############################
# ---------- Plots ---------- #
###############################
def plot_aggregated_fit_influenza(model_results, data_for_fit, season, age=None):
    # Get season length
    short = len(model_results['new_Is']) < 366

    # Get data to plot and model results
    if season != 'all':
        # If children
        if age == 0:
            infected_data_for_plot = data_for_fit['by_age' if not short else 'short_by_age'][0]
            model_weekly_cases = get_model_weekly_cases(model_results, season, short=short, age=0)
        # If adult
        elif age == 1:
            infected_data_for_plot = data_for_fit['by_age' if not short else 'short_by_age'][1]
            model_weekly_cases = get_model_weekly_cases(model_results, season, short=short, age=1)
        # If total
        else:
            infected_data_for_plot = data_for_fit['total' if not short else 'short_total']
            model_weekly_cases = get_model_weekly_cases(model_results, season, short=short)

        # Get only relevant season form the data
        infected_data_for_plot = infected_data_for_plot[infected_data_for_plot.season == season]

    else:  # plot all seasons
        # If children
        if age == 0:
            infected_data_for_plot = data_for_fit['by_age' if not short else 'short_by_age'][0]
            model_weekly_cases = pd.concat([get_model_weekly_cases(model_results[i], s, age=0, short=short) for i, s in enumerate(seasons)])
        # If adult
        elif age == 1:
            infected_data_for_plot = data_for_fit['by_age' if not short else 'short_by_age'][1]
            model_weekly_cases = pd.concat([get_model_weekly_cases(model_results[i], s, age=1, short=short) for i, s in enumerate(seasons)])
        # If total
        else:
            infected_data_for_plot = data_for_fit['total' if not short else 'short_total']
            model_weekly_cases = pd.concat([get_model_weekly_cases(model_results[i], s, short=short) for i, s in enumerate(seasons)])

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(infected_data_for_plot.index, infected_data_for_plot.cases, c='r', label='data')
    plt.plot(model_weekly_cases.index, model_weekly_cases.cases, linewidth=3, label='model')
    plt.title(f'New Influenza Cases - {"total" if age is None else ["children", "adults"][age]}', size=22)
    plt.xlabel('time', size=18)
    plt.ylabel('Number of cases', size=18)
    plt.legend(fontsize=18)
    plt.show()


def plot_aggregated_fit_with_cloud_influenza(model_results_list, data_for_fit, season, prep_data, likelihood_by, age=None):
    # Calculate log-likelihood
    if likelihood_by == 'subdist':
        model_res_likelihood_list = [log_likelihood_agg_by_subdist_influenza(res['lambdas'], data_for_fit['by_subdist'], season, prep_data)
                                     for res in model_results_list]
    if likelihood_by == 'age':
        model_res_likelihood_list = [log_likelihood_agg_age_influenza(res, data_for_fit, season) for res in model_results_list]

    # Get median realization
    med = np.argsort(np.array(model_res_likelihood_list))[len(model_res_likelihood_list)//2]

    # Create fig
    plt.figure(figsize=(20, 10))

    for i, res in enumerate(model_results_list):
        # Get data to plot and model results
        # If children
        if age == 0:
            infected_data_for_plot = data_for_fit['by_age'][0]
            model_weekly_cases = get_model_weekly_cases(res, season, short=False, age=0)
        # If adult
        elif age == 1:
            infected_data_for_plot = data_for_fit['by_age'][1]
            model_weekly_cases = get_model_weekly_cases(res, season, short=False, age=1)
        # If total
        else:
            infected_data_for_plot = data_for_fit['total']
            model_weekly_cases = get_model_weekly_cases(res, season, short=False)

        # Newly infected
        if i == med:
            med_plot = plt.plot(model_weekly_cases.index, model_weekly_cases.cases, linewidth=3, label='model')
        else:
            plt.plot(model_weekly_cases.index, model_weekly_cases.cases, linewidth=0.5, label='model', c='gray', alpha=0.4)

    # Plot data
    # Get only relevant season form the data
    infected_data_for_plot = infected_data_for_plot[infected_data_for_plot.season == season]
    data_plt = plt.scatter(infected_data_for_plot.index, infected_data_for_plot.cases, c='r', label='data')

    plt.title(f'New Influenza Cases - {"total" if age is None else ["children", "adults"][age]}', size=22)
    plt.xlabel('time', size=18)
    plt.ylabel('Number of cases', size=18)
    plt.legend(handles=[med_plot[0], data_plt], fontsize=18)

    plt.show()


def plot_fit_by_subdist_influenza(model_results, data_for_fit, prep_data, season):
    # Season length
    short = len(model_results['new_Is']) < 366
    season_length = 274 if short else 365

    # Aggregate lambda by subdist
    # Initialize dict to all arrays of 0s
    infected_by_subdist = {key: np.array([0] * season_length) for key in prep_data['relevant_subdists_age']}

    # Go over the clinics and age groups and aggregate according to the clinic's subdist
    for (clinic, age), data in model_results['Is_by_clinic_age'].items():
        subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
        infected_by_subdist[(subdist, age)] = infected_by_subdist[(subdist, age)] + data

    # Get weekly cases for each subdist and age
    for (subdist, age), data in infected_by_subdist.items():
        infected_by_subdist[(subdist, age)] = get_model_weekly_cases(data, season, short, by_subdist=True)

    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(15, 30))  # , sharey=True)
    plt.tight_layout(w_pad=3, h_pad=5)

    ages = ['children', 'adults']

    for i, (subdist, age) in enumerate(infected_by_subdist):
        # Plot newly infected - model
        axs[i // 2, age].plot(infected_by_subdist[(subdist, age)].index, infected_by_subdist[(subdist, age)].cases, label='model', linewidth=1.5)
        # Plot data
        data = data_for_fit[(subdist, age)][data_for_fit[(subdist, age)].season == season]
        axs[i // 2, age].scatter(data.index, data.cases, label='data', c='r', s=3)
        #         axs[i // 2, age].plot(data.index, data.cases, label='model', linewidth=1.5)

        axs[i // 2, age].set_title(f'subdistrict {subdist} - {ages[age]}', size=12, fontweight='bold')
        axs[i // 2, age].set_xlabel('Time (days of season)', size=10, fontweight='bold')
        axs[i // 2, age].set_ylabel('Number of individuals', size=10, fontweight='bold', labelpad=10)

        #         axs[i // 2, age].set_xticks(ts[::50])
        axs[i // 2, age].tick_params(labelsize=8)

    plt.show()


def plot_fit_by_subdist_with_cloud_influenza(model_results_list, data_for_fit, season, prep_data, likelihood_by):
    # Calculate log-likelihood
    if likelihood_by == 'subdist':
        model_res_likelihood_list = [log_likelihood_agg_by_subdist_influenza(res['lambdas'], data_for_fit['by_subdist'], season, prep_data)
                                     for res in model_results_list]
    if likelihood_by == 'age':
        model_res_likelihood_list = [log_likelihood_agg_age_influenza(res, data_for_fit, season) for res in model_results_list]

    # Get median realization
    med = np.argsort(np.array(model_res_likelihood_list))[len(model_res_likelihood_list)//2]

    # Create fig
    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(15, 30))  # , sharey=True)
    plt.tight_layout(w_pad=3, h_pad=5)
    ages = ['children', 'adults']

    for j, res in enumerate(model_results_list):
        # Aggregate lambda by subdist
        # Initialize dict to all arrays of 0s
        infected_by_subdist = {key: np.array([0] * 365) for key in prep_data['relevant_subdists_age']}

        # Go over the clinics and age groups and aggregate according to the clinic's subdist
        for (clinic, age), data in res['Is_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            infected_by_subdist[(subdist, age)] = infected_by_subdist[(subdist, age)] + data

        # Get weekly cases for each subdist and age
        for (subdist, age), data in infected_by_subdist.items():
            infected_by_subdist[(subdist, age)] = get_model_weekly_cases(data, season, short=False, by_subdist=True)

        for i, (subdist, age) in enumerate(infected_by_subdist):
            if j == med:
                # Plot newly infected - model
                axs[i // 2, age].plot(infected_by_subdist[(subdist, age)].index, infected_by_subdist[(subdist, age)].cases,
                                      label='model', linewidth=1.5)
            else:
                axs[i // 2, age].plot(infected_by_subdist[(subdist, age)].index, infected_by_subdist[(subdist, age)].cases,
                                      linewidth=0.3, label='model', c='gray', alpha=0.4)
            # Plot data
            data = data_for_fit['by_subdist'][(subdist, age)][data_for_fit['by_subdist'][(subdist, age)].season == season]
            axs[i // 2, age].scatter(data.index, data.cases, label='data', c='r', s=3)
            #         axs[i // 2, age].plot(data.index, data.cases, label='model', linewidth=1.5)

            axs[i // 2, age].set_title(f'subdistrict {subdist} - {ages[age]}', size=12, fontweight='bold')
            axs[i // 2, age].set_xlabel('Time (days of season)', size=10, fontweight='bold')
            axs[i // 2, age].set_ylabel('Number of individuals', size=10, fontweight='bold', labelpad=10)

            #         axs[i // 2, age].set_xticks(ts[::50])
            axs[i // 2, age].tick_params(labelsize=8)

    plt.show()


def plot_infection_rate_influenza(model_results, prep_data, season):
    # Get model and data infection rates by subdist
    infection_gb_subdist = calc_infection_rates(model_results, prep_data, season)

    # Plot
    infection_gb_subdist.plot.bar(figsize=(15, 7))
    plt.title('Infection Rate by Subdistrict', size=20)
    plt.xlabel('\nSubdistrict', size=15)
    plt.ylabel('Infection rate', size=15)
    plt.xticks(np.arange(14), infection_gb_subdist.index, rotation='horizontal', size=14)
    plt.legend(fontsize=15, labels=['data', 'model'], loc=(1.01, 0.87))
    plt.show()


def plot_infection_rate_by_age_influenza(model_results, data_for_fit, prep_data, season):
    # Get infection rates
    infection_rates = get_infection_rate_by_age(model_results, data_for_fit, prep_data, season)

    # Plot
    infection_rates.plot.bar(figsize=(15, 7))
    plt.title('Infection Rate by Age', size=20)
    plt.xlabel('', size=15)
    plt.ylabel('Infection rate', size=15)
    plt.xticks(np.arange(3), infection_rates.index, rotation='horizontal', size=14)
    plt.legend(fontsize=15, labels=['data', 'model'], loc=(1.01, 0.87))
    plt.show()


def plot_model_comparison_influenza(results, data_for_fit, prep_data, season, measure):
    # Get error rates
    errors = model_comparison_influenza(results, data_for_fit, prep_data, season, [measure])

    errors.plot.bar(figsize=(15, 7))
    plt.title('Model Comparison', size=20)
    plt.xlabel('', size=15)
    plt.ylabel(measure, size=15)
    plt.xticks(np.arange(3), errors.index, rotation='horizontal', size=14)
    plt.legend(fontsize=15)
    plt.show()



