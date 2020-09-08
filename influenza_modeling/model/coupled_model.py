import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import networkx as nx
from itertools import product
import pickle
from .vaccination_model_with_age import *
from.influenza_model import *


def run_coupled_model(parameters_i, parameters_v, prep_data, season, short=False, vacc_start=92, homogenous=False):
    ##############################
    # ----- Initialization ----- #
    ##############################
    # Get influenza model parameters
    beta_i = parameters_i[season]['beta']
    delta_i = [parameters_i[season]['delta'], 1]
    phi = parameters_i[season]['phi']
    I_0_size_i = 0  # parameters[season]['I_0_size']
    epsilon = parameters_i[season]['epsilon']

    # Get vaccination model parameters
    beta_v = parameters_v['beta']
    delta_v = [parameters_v['delta'], 1]
    gamma_v = parameters_v['gamma']
    I_0_size_v = parameters_v['I_0_size']

    # Get prep data
    network, relevant_clinics_age, node_clinic_age = prep_data['network'], prep_data['relevant_clinics_age'], prep_data['node_clinic_age']
    # nodes_by_clinic_age = prep_data['nodes_by_clinic_age']
    # vaccination_coverage_influenza = prep_data['vaccination_coverage_influenza']
    # population_by_clinic_age = prep_data['population_by_clinic_age']

    # Set season length
    season_length = 274 if short else 365
    # season_length = 244 if short else 365

    # Influenza vaccination efficacy
    vaccination_efficacy = 0.45

    # Influenza vaccination delay (in days)
    vacc_delay = 14

    # Vaccination waning (111 days)
    omega = 1 / (111-vacc_delay)

    # Asymptomatic fraction
    asymp_frac = 0.191

    # Transmissibility - based on log viral load according to infection type (symptomatic/asymptomatic)
    rho_symp = 5.5
    rho_asym = 4.5

    # Recovery rate - symptomatic
    gamma_symp = [0.1477929674216146, 1 / 4.5]
    # Recovery rate - asymptomatic
    gamma_asymp = 1 / 3.2

    # Initialize vaccination model
    vacc_model = initialize_vaccination_model(parameters_v, prep_data, homogenous=homogenous)
    # Vaccination model states
    Sv, Iv, Rv, Vv, Vn = vacc_model['S'], vacc_model['I'], vacc_model['R'], [set()], [set()]
    # Newly infected
    new_Iv, new_Iv_by_age, Iv_by_clinic_age = vacc_model['new_I'], vacc_model['new_I_by_age'], vacc_model['infected_by_clinic_age']
    # Lambdas
    lambdas_v, lambdas_agg_total_v, lambdas_agg_children_v, lambdas_agg_adult_v = vacc_model['lambdas'], vacc_model['lambdas_agg_total'], \
                                                                                  vacc_model['lambdas_agg_children'], vacc_model['lambdas_agg_adult']
    # If homogenous model
    if homogenous:
        Sv_by_age = vacc_model['S_by_age']
        C = vacc_model['C']

    # Initialize influenza model
    flu_model = initialize_influenza_model(prep_data, homogenous=homogenous)
    # Influenza model states
    Si, Vi, Is, Ia, Ri, Rs = flu_model['S'], flu_model['V'], flu_model['Is'], flu_model['Ia'], flu_model['R'], flu_model['Rs']
    # Newly infected
    new_Is, new_Is_by_age, Is_by_clinic_age = flu_model['new_Is'], flu_model['new_Is_by_age'], flu_model['Is_by_clinic_age']
    # Lambdas
    lambdas, lambdas_agg_total, lambdas_agg_children, lambdas_agg_adult = flu_model['lambdas'], flu_model['lambdas_agg_total'], \
                                                                          flu_model['lambdas_agg_children'], flu_model['lambdas_agg_adult']
    # If homogenous model
    if homogenous:
        Si_by_age = flu_model['S_by_age']
        Si_by_clinic_age = flu_model['S_by_clinic_age']

    #############################
    # ------- Run Model ------- #
    #############################

    # for t in tqdm(range(season_length)):
    for t in (range(season_length)):
        # Vaccination model newly infected
        new_Iv_t = set()
        new_Rv_t = set()
        new_V_t = set()
        new_Vn_t = set()
        new_Iv_by_age_t = [set(), set()]

        # Influenza model newly infected
        new_Ia_t = set()
        new_Is_t = set()
        new_Is_by_age_t = [set(), set()]
        new_R_t = set()
        new_Rs_t = set()
        new_Si_t = set()

        # --- Vaccination "infection" --- #
        # Only if within vaccination season (after the beginning)
        if vacc_start <= t < vacc_start + 181:  # 92 = from September 1st # 181 = vacc season len
            # If network model
            if not homogenous:
                # Go over the infected (vaccinated) nodes
                for node in Iv[-1]:
                    for contact in network[node]:
                        # If the contact is susceptible and not exposed in this stage yet
                        if contact in Sv[-1] and contact not in new_Iv_t:
                            # Get age and clinic
                            cur_clinic, cur_age = node_clinic_age[contact]
                            # Contact is exposed with probability beta_2 * delta
                            if np.random.rand() < beta_v * delta_v[cur_age]:
                                new_Iv_t.add(contact)
                                new_Iv_by_age_t[cur_age].add(contact)
                                Iv_by_clinic_age[(cur_clinic, cur_age)][t-vacc_start] += 1
                            # # Update lambda_v
                            # lambdas_v[(cur_clinic, cur_age)][t-vacc_start] += beta_v * delta_v[cur_age]
                            # lambdas_agg_total_v[t-vacc_start] += beta_v * delta_v[cur_age]
                            # if cur_age == 0:
                            #     lambdas_agg_children_v[t-vacc_start] += beta_v * delta_v[cur_age]
                            # else:
                            #     lambdas_agg_adult_v[t-vacc_start] += beta_v * delta_v[cur_age]

                # Add "epsilon" - constant infection of 1 random node each day
                if len(Sv[-1] - new_Iv_t) <= epsilon:
                    epsilon_nodes = Sv[-1] - new_Iv_t
                else:
                    epsilon_nodes = set(np.random.choice(list(Sv[-1] - new_Iv_t), epsilon, replace=False))
                for i, node in enumerate(epsilon_nodes):
                    # Get age and clinic
                    cur_clinic, cur_age = node_clinic_age[node]
                    new_Iv_t.add(node)
                    new_Iv_by_age_t[cur_age].add(node)
                    Iv_by_clinic_age[(cur_clinic, cur_age)][t-vacc_start] += 1

            # If homogenous model
            else:
                # Get number of nodes to infect (for each age group
                num_of_infected = np.zeros(2)
                for node in Iv[-1]:
                    # Get age of infected individual
                    cur_age = node_clinic_age[node][1]
                    # Generate number of infected form current individual from each age group (form poisson with lambda)
                    cur_num_of_infected = np.array([np.random.poisson(C[cur_age, 0] * beta_v * delta_v[0] * len(Sv_by_age[0])),
                                                    np.random.poisson(C[cur_age, 1] * beta_v * delta_v[1] * len(Sv_by_age[1]))])
                    num_of_infected += cur_num_of_infected

                # Choose nodes to infect (randomly)
                if num_of_infected[0] < (len(Sv_by_age[0])):
                    new_Iv_by_age_t[0] = set(np.random.choice(list(Sv_by_age[0]), size=int(num_of_infected[0]), replace=False))
                else:
                    new_Iv_by_age_t[0] = Sv_by_age[0]

                if num_of_infected[1] < (len(Sv_by_age[1])):
                    new_Iv_by_age_t[1] = set(np.random.choice(list(Sv_by_age[1]), size=int(num_of_infected[1]), replace=False))
                else:
                    new_Iv_by_age_t[1] = Sv_by_age[1]

                # Add "epsilon" - constant infection of 1 random node each day
                if len(Sv[-1] - new_Iv_t) <= epsilon:
                    epsilon_nodes = Sv[-1] - new_Iv_t
                else:
                    epsilon_nodes = set(np.random.choice(list(Sv[-1] - new_Iv_t), epsilon, replace=False))
                for i, node in enumerate(epsilon_nodes):
                    # Get age and clinic
                    cur_clinic, cur_age = node_clinic_age[node]
                    new_Iv_t.add(node)
                    new_Iv_by_age_t[cur_age].add(node)

                # Infect nodes
                new_Iv_t = new_Iv_by_age_t[0].union(new_Iv_by_age_t[1])

                # Update infected by clinic and age and S_by_clinic_age
                for node in new_Iv_t:
                    cur_clinic, cur_age = node_clinic_age[node]
                    Iv_by_clinic_age[(cur_clinic, cur_age)][t-vacc_start] += 1

                # Update S_by_age
                Sv_by_age[0] = Sv_by_age[0] - new_Iv_by_age_t[0]
                Sv_by_age[1] = Sv_by_age[1] - new_Iv_by_age_t[1]

            # Transmission from Iv to Rv
            for node in Iv[-1]:
                # Individuals transmitted from Iv to Rv with probability gamma
                new_Rv_t.add(node) if np.random.rand() < gamma_v else None

            # "Transmission" from Iv to Vv or Vn
            # Individuals transmitted from Iv to V with probability 1/vaccine_delay
            for node in Iv[-1]:
                if np.random.rand() < 1/vacc_delay:
                    # If the node is not already in Vv or Vn
                    if node not in Vv[-1] and node not in Vn[-1]:
                        # Individuals are transmitted to V only if the vaccine is effective (with probability of vaccination_efficacy)
                        if np.random.rand() < vaccination_efficacy:
                            new_V_t.add(node)
                        else:
                            new_Vn_t.add(node)

            # "Transmission" from Rv to Vv or Vn
            for node in Rv[-1]:
                # Individuals transmitted from Rv to V with probability 1/vaccine_delay
                # If the node is not already in Vv or Vn
                if node not in Vv[-1] and node not in Vn[-1]:
                    if np.random.rand() < 1/vacc_delay:
                        # Individuals are transmitted to V only if the vaccine is effective (with probability of vaccination_efficacy)
                        if np.random.rand() < vaccination_efficacy:
                            new_V_t.add(node)
                        else:
                            new_Vn_t.add(node)

            # Update stages
            Sv.append(Sv[-1] - new_Iv_t)
            Iv.append(Iv[-1].union(new_Iv_t) - new_Rv_t)
            Rv.append(Rv[-1].union(new_Rv_t))  # - new_V_t - new_Vn_t)
            Vv.append(Vv[-1].union(new_V_t))
            Vn.append(Vn[-1].union(new_Vn_t))

            # Save the newly infected
            new_Iv.append(new_Iv_t)
            new_Iv_by_age[0].append(new_Iv_by_age_t[0])
            new_Iv_by_age[1].append(new_Iv_by_age_t[1])

        # --- Influenza infection --- #
        # Go over the infected individuals
        # If network model
        if not homogenous:
            for i, node in enumerate(list(Is[-1]) + list(Ia[-1])):
                for contact in network[node]:
                    # If the contact is susceptible and not exposed in this stage yet
                    if contact in Si[-1] and contact not in new_Is_t and contact not in new_Ia_t:
                        # Get age and clinic
                        cur_clinic, cur_age = node_clinic_age[contact]
                        # Get relevant transmissability (rho)
                        rho = rho_symp if i < len(Is[-1]) else rho_asym
                        # Force of infection
                        force_of_infection = beta_i * delta_i[cur_age] * rho * (1 + np.cos((2 * np.pi * t) / 365 + phi))
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
                        lambdas[(cur_clinic, cur_age)][t] += force_of_infection * (1 - asymp_frac)
                        lambdas_agg_total[t] += force_of_infection * (1 - asymp_frac)
                        if cur_age == 0:
                            lambdas_agg_children[t] += force_of_infection * (1 - asymp_frac)
                        else:
                            lambdas_agg_adult[t] += force_of_infection * (1 - asymp_frac)

            # Add "epsilon" - constant infection of 5 random nodes each day
            if len(Si[-1] - new_Is_t - new_Ia_t) <= epsilon:
                epsilon_nodes = Si[-1] - new_Is_t - new_Ia_t - new_V_t
            else:
                epsilon_nodes = set(np.random.choice(list(Si[-1] - new_Is_t - new_Ia_t - new_V_t), epsilon, replace=False))
            for i, node in enumerate(epsilon_nodes):
                # Get age and clinic
                cur_clinic, cur_age = node_clinic_age[node]
                if i < 4:
                    new_Is_t.add(node)
                    new_Is_by_age_t[cur_age].add(node)
                else:
                    new_Ia_t.add(node)

        # If homogenous model
        else:
            # Get number of nodes to infect (for each age group
            num_of_infected_s = np.zeros(2)
            num_of_infected_a = np.zeros(2)
            for i, node in enumerate(list(Is[-1]) + list(Ia[-1])):
                # Get age of infected individual
                cur_age = node_clinic_age[node][1]
                # Get relevant transmissability (rho)
                rho = rho_symp if i < len(Is[-1]) else rho_asym
                # Force of infection - for children/adult - based on the contacted node
                force_of_infection_children = beta_i * delta_i[0] * rho * (1 + np.cos((2 * np.pi * t) / 365 + phi))
                force_of_infection_adult = beta_i * delta_i[1] * rho * (1 + np.cos((2 * np.pi * t) / 365 + phi))

                # Update lambdas
                # Calculate lambdas by clinic and age - by current infected individual
                # to all susceptible individuals (by clinic and age)
                for contact_clinic, contact_age in lambdas:
                    lambdas[(contact_clinic, contact_age)][t] += \
                        C[cur_age, contact_age] * beta_i * delta_i[contact_age] * len(Si_by_clinic_age[(contact_clinic, contact_age)])

                # Calculate lambda for children and adults (from current infected individual)
                cur_lambda_children = C[cur_age, 0] * force_of_infection_children * len(Si_by_age[0]) * (1 - asymp_frac)
                cur_lambda_adult = C[cur_age, 1] * force_of_infection_adult * len(Si_by_age[1]) * (1 - asymp_frac)
                # Update total lambda for current iteration
                lambdas_agg_children[t] += cur_lambda_children
                lambdas_agg_adult[t] += cur_lambda_adult

                # Generate number of infected form current individual from each age group (form poisson with lambda)
                # Symptomatic
                cur_num_of_infected_s = np.array(
                    [np.random.poisson(C[cur_age, 0] * force_of_infection_children * len(Si_by_age[0]) * (1 - asymp_frac)),
                     np.random.poisson(C[cur_age, 1] * force_of_infection_adult * len(Si_by_age[1]) * (1 - asymp_frac))])
                num_of_infected_s += cur_num_of_infected_s
                # Asymptomatic
                cur_num_of_infected_a = np.array([np.random.poisson(C[cur_age, 0] * force_of_infection_children * len(Si_by_age[0]) * asymp_frac),
                                                  np.random.poisson(C[cur_age, 1] * force_of_infection_adult * len(Si_by_age[1]) * asymp_frac)])
                num_of_infected_a += cur_num_of_infected_a

            # Choose nodes to infect (randomly) - make sure not to sample more the number of susceptible
            # Symptomatic
            if len(Si_by_age[0] - new_V_t) <= int(num_of_infected_s[0]):
                new_Is_by_age_t[0] = Si_by_age[0] - new_V_t
            else:
                new_Is_by_age_t[0] = set(
                    np.random.choice(list(Si_by_age[0] - new_V_t), size=min(len(Si_by_age[0]), int(num_of_infected_s[0])), replace=False))

            if len(Si_by_age[1] - new_V_t) <= int(num_of_infected_s[1]):
                new_Is_by_age_t[1] = Si_by_age[1] - new_V_t
            else:
                new_Is_by_age_t[1] = set(
                    np.random.choice(list(Si_by_age[1] - new_V_t), size=min(len(Si_by_age[1]), int(num_of_infected_s[1])), replace=False))

            # Asymptomatic
            if len(Si_by_age[0] - new_V_t - new_Is_by_age_t[0]) <= int(num_of_infected_a[0]):
                new_Ia_children = Si_by_age[0] - new_V_t - new_Is_by_age_t[0]
            else:
                new_Ia_children = set(
                    np.random.choice(list(Si_by_age[0] - new_V_t - new_Is_by_age_t[0]), size=min(len(Si_by_age[0]), int(num_of_infected_a[0])),
                                     replace=False))

            if len(Si_by_age[1] - new_V_t - new_Is_by_age_t[1]) <= int(num_of_infected_a[1]):
                new_Ia_adults = Si_by_age[1] - new_V_t - new_Is_by_age_t[1]
            else:
                new_Ia_adults = set(
                    np.random.choice(list(Si_by_age[1] - new_V_t - new_Is_by_age_t[1]), size=min(len(Si_by_age[1]), int(num_of_infected_a[1])),
                                     replace=False))

            # Infect nodes
            new_Is_t = new_Is_by_age_t[0].union(new_Is_by_age_t[1])
            new_Ia_t = new_Ia_children.union(new_Ia_adults)

            # Add "epsilon" - constant infection of 5 random nodes each day
            if len(Si[-1] - new_Is_t - new_Ia_t) <= epsilon:
                epsilon_nodes = Si[-1] - new_Is_t - new_Ia_t - new_V_t
            else:
                epsilon_nodes = set(np.random.choice(list(Si[-1] - new_Is_t - new_Ia_t - new_V_t), epsilon, replace=False))
            for i, node in enumerate(epsilon_nodes):
                # Get age and clinic
                cur_clinic, cur_age = node_clinic_age[node]
                if i < 4:
                    new_Is_t.add(node)
                    new_Is_by_age_t[cur_age].add(node)
                else:
                    new_Ia_t.add(node)
                    new_Ia_children.add(node) if cur_age == 0 else new_Ia_adults.add(node)

            # Update infected by clinic and age
            for node in new_Is_t:
                cur_clinic, cur_age = node_clinic_age[node]
                Is_by_clinic_age[(cur_clinic, cur_age)][t] += 1

            # Update Si_by_clinic_age (remove infected)
            for i, node in enumerate(list(new_Is_t) + list(new_Ia_t)):
                cur_clinic, cur_age = node_clinic_age[node]
                Si_by_clinic_age[(cur_clinic, cur_age)].remove(node)
                if i < len(new_Is_t):
                    # Update new infected by clinic and age
                    Is_by_clinic_age[(cur_clinic, cur_age)][t] += 1

            # Update S_by_age (remove infected)
            Si_by_age[0] = Si_by_age[0] - new_Is_by_age_t[0] - new_Ia_children
            Si_by_age[1] = Si_by_age[1] - new_Is_by_age_t[1] - new_Ia_adults

            # Update Si_by_age and Si_by_clinic_age (remove vaccinated)
            for node in new_V_t:
                cur_clinic, cur_age = node_clinic_age[node]
                if node in Si[-1]:
                    Si_by_age[cur_age].remove(node)
                    Si_by_clinic_age[(cur_clinic, cur_age)].remove(node)

        # Transmission from I to R
        for i, node in enumerate(list(Is[-1]) + list(Ia[-1])):
            # Get infection type
            infection_type = 'symptomatic' if i < len(Is[-1]) else 'asymptomatic'
            # Get individual's age
            cur_age = node_clinic_age[node][1]
            # Get relevant gamma according to infection type and individual age
            if infection_type == 'symptomatic':
                gamma_i = gamma_symp[cur_age]
            elif infection_type == 'asymptomatic':
                gamma_i = gamma_asymp
            # Individuals transmitted from I to R with probability gamma - according to infection type and age
            if np.random.rand() < gamma_i:
                new_R_t.add(node)
                if infection_type == 'symptomatic':
                    new_Rs_t.add(node)

        # Transmission form V to S
        for node in Vi[-1]:
            # Individuals transmitted from Iv to Rv with probability gamma
            new_Si_t.add(node) if np.random.rand() < omega else None

        # If homogenous model add back to S_by_age
        if homogenous:
            for node in new_Si_t:
                # Get nodes clinic and age
                cur_clinic, cur_age = node_clinic_age[node]
                Si_by_age[cur_age].add(node)
                Si_by_clinic_age[(cur_clinic, cur_age)].add(node)

        # Update stages
        Si.append(Si[-1].union(new_Si_t) - new_Is_t - new_Ia_t - new_V_t)
        Vi.append(Vi[-1].union(new_V_t - Ri[-1]) - new_R_t - new_Si_t)  # Do not add immuned nodes
        Is.append(Is[-1].union(new_Is_t) - new_R_t)
        Ia.append(Ia[-1].union(new_Ia_t) - new_R_t)
        Ri.append(Ri[-1].union(new_R_t))
        Rs.append(Rs[-1].union(new_Rs_t))

        # Save the newly infected
        new_Is.append(new_Is_t)
        new_Is_by_age[0].append(new_Is_by_age_t[0])
        new_Is_by_age[1].append(new_Is_by_age_t[1])

    # Save results to dictionary - lean version
    model_results = {'Rs': Rs, 'Is': Is, 'Iv': Iv,
                     'new_Iv': new_Iv, 'new_Iv_by_age': new_Iv_by_age, 'Iv_by_clinic_age': Iv_by_clinic_age,  # Vaccination model newly infected
                     'new_Is': new_Is, 'new_Is_by_age': new_Is_by_age, 'Is_by_clinic_age': Is_by_clinic_age,  # Influenza model newly infected
                     'lambdas': lambdas, 'lambdas_agg_total': lambdas_agg_total, 'lambdas_agg_children': lambdas_agg_children,  # lambdas flu
                     'lambdas_agg_adult': lambdas_agg_adult}

    # # Save results to dictionary - full version
    # model_results = {'Sv': Sv, 'Iv': Iv, 'Rv': Rv, 'Vv': Vv, 'Vn': Vn,  # Vaccination model states
    #                  'new_Iv': new_Iv, 'new_Iv_by_age': new_Iv_by_age, 'Iv_by_clinic_age': Iv_by_clinic_age,  # Vaccination model newly infected
    #                  'Si': Si, 'Vi': Vi, 'Is': Is, 'Ia': Ia, 'Ri': Ri, 'Rs': Rs,  # Influenza model states
    #                  'new_Is': new_Is, 'new_Is_by_age': new_Is_by_age, 'Is_by_clinic_age': Is_by_clinic_age,  # Influenza model newly infected
    #                  'lambdas': lambdas, 'lambdas_agg_total': lambdas_agg_total, 'lambdas_agg_children': lambdas_agg_children,  # lambdas flu
    #                  'lambdas_agg_adult': lambdas_agg_adult}

                     # ,'lambdas_v': lambdas_v, 'lambdas_agg_total_v': lambdas_agg_total_v,
                     # 'lambdas_agg_children_v': lambdas_agg_children_v, 'lambdas_agg_adult_v': lambdas_agg_adult_v}  # lambdas vaccination

    return model_results


############################
# ----- Intervention ----- #
############################
def start_date_analysis_coupled_model(parameters_i, parameters_v, prep_data, data_for_fit_i, vacc_start, num_of_simulations):
    # Initialize lists for infection rate and vaccination coverage
    infection_rate_list = []

    for season in tqdm(seasons):
        for i in range(num_of_simulations):
            # Run the model
            model_res = run_coupled_model(parameters_i, parameters_v, prep_data, season, vacc_start=vacc_start)

            # Calculate infection rate and add to the list
            infection_rate = get_infection_rate_by_age(model_res, data_for_fit_i, prep_data, season).model
            # infection_rate -= [baseline['attack_rates'][age] for age in ['children', 'adult_rate', 'total']]
            infection_rate_list.append(infection_rate)

    # Create arrays
    total_attack_rates = np.array([r['total'] for r in infection_rate_list])
    children_attack_rates = np.array([r['children'] for r in infection_rate_list])
    adult_attack_rates = np.array([r['adults'] for r in infection_rate_list])

    # Calculate mean and std
    attack_rates = {'total': total_attack_rates.mean(), 'total_std': total_attack_rates.std(),
                    'children': children_attack_rates.mean(), 'children_std': children_attack_rates.std(),
                    'adult_rate': adult_attack_rates.mean(), 'adult_std': adult_attack_rates.std()}

    return {'attack_rates': attack_rates}


############################
# --- Helper Functions --- #
############################
def get_coupled_model_weekly_vacc(model_results, prep_data, age=None, by_subdist=False, by_subdist_age=False):
    """Receives model results dictionary and returns the number of weekly cases aggregated weekly, by age group"""
    # Get dates for aggregation
    dates = [pd.Timestamp(2016, 9, 1) + pd.Timedelta(i, unit='d') for i in range(181)]

    if not by_subdist and not by_subdist_age:
        # Get the model new symptomatic cases by season according to age
        if age == 0:
            model_vacc_nodes = model_results['new_Iv_by_age'][0][:182]
        elif age == 1:
            model_vacc_nodes = model_results['new_Iv_by_age'][1][:182]
        else:  # Total
            model_vacc_nodes = model_results['new_Iv'][:182]

        # Get number of cases by day
        model_vacc_by_day = np.array([len(st) for st in model_vacc_nodes[1:]])

        # Create a DF of the cases with the dates as index
        model_vacc_df = pd.DataFrame(model_vacc_by_day, index=np.array(dates), columns=['vacc_count'])

        # Aggregate weekly
        model_weekly_vacc = model_vacc_df.resample('W').sum().fillna(0).copy()

        return model_weekly_vacc.vacc_count.values

    elif by_subdist:  # by subdist
        # Initialize a dict for infected by subdist
        Iv_by_subdist = {subdist: np.array([0] * 181) for subdist in prep_data['relevant_subdists']}

        # Go over clinic and sum the data to subdist leve
        for (clinic, age), data in model_results['Iv_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            Iv_by_subdist[subdist] = Iv_by_subdist[subdist] + data

        # Go over subdist and aggregate weekly
        model_weekly_vacc_by_subdist = {}
        for subdist in prep_data['relevant_subdists']:
            # Create a DF of the cases with the dates as index
            model_vacc_df = pd.DataFrame(Iv_by_subdist[subdist], index=np.array(dates), columns=['vacc_count'])
            # Aggregate weekly
            model_weekly_vacc = model_vacc_df.resample('W').sum().fillna(0).copy()
            # Save to dict
            model_weekly_vacc_by_subdist[subdist] = model_weekly_vacc.vacc_count.values

        return model_weekly_vacc_by_subdist

    else:  # by subdist_age
        # Initialize a dict for infected by subdist
        Iv_by_subdist_age = {(subdist, age): np.array([0] * 181) for subdist, age in prep_data['relevant_subdists_age']}

        # Go over clinic and sum the data to subdist leve
        for (clinic, age), data in model_results['Iv_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            Iv_by_subdist_age[(subdist, age)] = Iv_by_subdist_age[(subdist, age)] + data

        # Go over subdist and aggregate weekly
        model_weekly_vacc_by_subdist_age = {}
        for subdist, age in prep_data['relevant_subdists_age']:
            # Create a DF of the cases with the dates as index
            model_vacc_df = pd.DataFrame(Iv_by_subdist_age[(subdist, age)], index=np.array(dates), columns=['vacc_count'])
            # Aggregate weekly
            model_weekly_vacc = model_vacc_df.resample('W').sum().fillna(0).copy()
            # Save to dict
            model_weekly_vacc_by_subdist_age[(subdist, age)] = model_weekly_vacc.vacc_count.values

        return model_weekly_vacc_by_subdist_age


def get_coupled_model_weekly_vacc_all_seasons(model_results_list, prep_data, age=None, by_subdist=False, by_subdist_age=False):
    if not by_subdist and not by_subdist_age:
        return np.concatenate([np.concatenate([[0], get_coupled_model_weekly_vacc(model_results_list[i], prep_data, age=age)])
                               for i in range(len(seasons))])

    elif by_subdist:  # by subdist
        return {subdist: np.concatenate([np.concatenate([[0], get_coupled_model_weekly_vacc(model_results_list[i], prep_data, by_subdist=True)[subdist]])
                                         for i in range(len(seasons))])
                for subdist in prep_data['relevant_subdists']}

    else:  # by subdist_age
        return {(subdist, age): np.concatenate([np.concatenate([[0], get_coupled_model_weekly_vacc(model_results_list[i], prep_data, by_subdist_age=True)[(subdist, age)]])
                                         for i in range(len(seasons))])
                for (subdist, age) in prep_data['relevant_subdists_age']}


def calculate_likelihood_lists_all_seasons(model_results_list, data_for_fit_i, prep_data, single=False):
    # If multiple realizations
    if not single:
        # Initialize a dict for likelihood list for each season
        likelihood_lists = {season: [] for season in seasons}

        # Go over results
        for model_res_all_seasons in model_results_list:
            # Go over seasons
            for s, season in enumerate(seasons):
                # Calculate the sum of the likelihoods over all seasons
                season_likelihood = log_likelihood_agg_by_subdist_influenza(model_res_all_seasons[s]["lambdas"], data_for_fit_i["by_subdist"],
                                                                            season, prep_data)
                # Add to the relevant list
                likelihood_lists[season].append(season_likelihood)

        return likelihood_lists

    else:  # single realization
        # Initialize a dict for likelihood list for each season
        likelihood_by_season = {}

        # Go over results
        for s, season in enumerate(seasons):
            # Calculate the sum of the likelihoods over all seasons
            season_likelihood = log_likelihood_agg_by_subdist_influenza(model_results_list[s]["lambdas"], data_for_fit_i["by_subdist"],
                                                                        season, prep_data)
            # Add to the relevant list
            likelihood_by_season[season] = season_likelihood

        return likelihood_by_season


###############################
# ---------- Plots ---------- #
###############################
def plot_aggregated_fit_coupled(model_results, data_for_fit_i, data_for_fit_v, season, prep_data, age=None, vacc_start=92):
    # Influenza model
    # Get season length
    short = len(model_results['new_Is']) < 366

    # Get data to plot and model results
    # If children
    if age == 0:
        infected_data_for_plot = data_for_fit_i['by_age' if not short else 'short_by_age'][0]
        model_weekly_cases = get_model_weekly_cases(model_results, season, short=short, age=0)
    # If adult
    elif age == 1:
        infected_data_for_plot = data_for_fit_i['by_age' if not short else 'short_by_age'][1]
        model_weekly_cases = get_model_weekly_cases(model_results, season, short=short, age=1)
    # If total
    else:
        infected_data_for_plot = data_for_fit_i['total' if not short else 'short_total']
        model_weekly_cases = get_model_weekly_cases(model_results, season, short=short)

    # Get only relevant season form the data
    infected_data_for_plot = infected_data_for_plot[infected_data_for_plot.season == season]

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(infected_data_for_plot.index, infected_data_for_plot.cases, c='b', label='influenza data')
    plt.plot(model_weekly_cases.index, model_weekly_cases.cases, linewidth=3, label='influenza model')

    # Vaccination model
    # Get data to plot and model results
    # If children
    if age == 0:
        vacc_data_for_plot = data_for_fit_v['infected_data_agg_children']
        model_weekly_vacc = get_coupled_model_weekly_vacc(model_results, prep_data, age=age)
    # If adult
    elif age == 1:
        vacc_data_for_plot = data_for_fit_v['infected_data_agg_adult']
        model_weekly_vacc = get_coupled_model_weekly_vacc(model_results, prep_data, age=age)
    # If total
    else:
        vacc_data_for_plot = data_for_fit_v['infected_data_agg']
        model_weekly_vacc = get_coupled_model_weekly_vacc(model_results, prep_data)

    # Time steps
    # Get vaccination season start date
    start_date = pd.Timestamp(season - 1, 6, 1) + pd.Timedelta(days=vacc_start - 1)
    # Get the season range - form start date
    dates = [start_date + pd.Timedelta(days=7) * i for i in range(27)]

    # Plot vaccinated
    plt.scatter(dates, vacc_data_for_plot.vacc_count, c='r', label='vaccination data')
    plt.plot(dates, model_weekly_vacc, linewidth=3, label='vaccination model')

    plt.title(f'Coupled Model - {"total" if age is None else ["children", "adults"][age]}', size=22)
    plt.xlabel('Time', size=20)
    plt.ylabel('Number of individuals', size=20)

    plt.legend(fontsize=20)

    plt.show()


def plot_aggregated_fit_coupled_all_seasons(model_results_list, data_for_fit_i, data_for_fit_v, prep_data, age=None, vacc_start=92):
    # Influenza model
    # Get season length
    short = len(model_results_list[0]['new_Is']) < 366

    # Get data to plot and model results
    # If children
    if age == 0:
        infected_data_for_plot = data_for_fit_i['by_age' if not short else 'short_by_age'][0]
        model_weekly_cases = get_model_weekly_cases_all_seasons(model_results_list, prep_data, age=0)
    # If adult
    elif age == 1:
        infected_data_for_plot = data_for_fit_i['by_age' if not short else 'short_by_age'][1]
        model_weekly_cases = get_model_weekly_cases_all_seasons(model_results_list, prep_data, age=1)
    # If total
    else:
        infected_data_for_plot = data_for_fit_i['total' if not short else 'short_total']
        model_weekly_cases = get_model_weekly_cases_all_seasons(model_results_list, prep_data)

    # Plot flu model and data
    fig = plt.figure(figsize=(30, 10))
    plt.scatter(infected_data_for_plot.index, infected_data_for_plot.cases, c='b', label='influenza data')
    plt.plot(model_weekly_cases.index, model_weekly_cases.cases, linewidth=3, label='influenza model')

    # Vaccination model
    # Get data to plot and model results
    # If children
    if age == 0:
        vacc_data_for_plot = data_for_fit_v['infected_data_agg_children']
        model_weekly_vacc = get_coupled_model_weekly_vacc_all_seasons(model_results_list, prep_data, age=age)
    # If adult
    elif age == 1:
        vacc_data_for_plot = data_for_fit_v['infected_data_agg_adult']
        model_weekly_vacc = get_coupled_model_weekly_vacc_all_seasons(model_results_list, prep_data, age=age)
    # If total
    else:
        vacc_data_for_plot = data_for_fit_v['infected_data_agg']
        model_weekly_vacc = get_coupled_model_weekly_vacc_all_seasons(model_results_list, prep_data, age=age)

    all_dates = []
    for season in seasons:
        # Get vaccination season start date (minus 1 week)
        start_date = pd.Timestamp(season - 1, 6, 1) + pd.Timedelta(days=vacc_start - 7)
        # Get the season range - form start date (minus week)
        dates = [start_date + pd.Timedelta(days=7) * i for i in range(27 + 1)]
        # Add to the list
        all_dates.extend(dates)

    # Add 0 before the first point - for plotting
    vacc_data_for_plot = np.concatenate([[0], vacc_data_for_plot.vacc_count])

    # Plot vaccinated
    plt.scatter(all_dates, np.tile(vacc_data_for_plot, reps=len(seasons)), c='r', label='vaccination data')
    plt.plot(all_dates, model_weekly_vacc, linewidth=3, label='vaccination model')

    plt.title(f'Coupled Model - {"total" if age is None else ["children", "adults"][age]}', size=22)
    plt.xlabel('time', size=24, labelpad=5)
    plt.ylabel('number of individuals', size=24, labelpad=5)

    plt.tick_params(labelsize=22)

    plt.legend(fontsize=20, loc='upper right')

    plt.show()


def plot_fit_by_subdist_coupled(model_results_list, data_for_fit_i, data_for_fit_v, prep_data, vacc_start=92):
    # Create figure
    fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(10, 30))
    plt.tight_layout(w_pad=3, h_pad=5)

    # Get subdists names
    subdists_names = {11.0: 'Jerusalem', 41.0: 'Sharon', 42.0: 'Petah-Tikva', 43.0: 'Ramla', 44.0: 'Rehovot', 51.0: 'Tel-Aviv',
                      61.0: 'Ashkelon', 77.0: 'Judea & Samaria'}

    # Influenza model
    # Get weekly cases for each subdist
    model_weekly_cases_by_subdist = get_model_weekly_cases_all_seasons(model_results_list, prep_data, by_subdist=True)

    for i, subdist in enumerate(model_weekly_cases_by_subdist):
        # Plot newly infected - model
        axs[i].plot(model_weekly_cases_by_subdist[subdist].index, model_weekly_cases_by_subdist[subdist].cases,
                    label='influenza model', linewidth=1.5)
        # Plot data
        data = data_for_fit_i['by_subdist'][(subdist, 0)].copy()
        data.cases += data_for_fit_i['by_subdist'][(subdist, 1)].cases
        axs[i].scatter(data.index, data.cases, label='influenza data', c='b', s=3)

        axs[i].set_title(f'{subdists_names[subdist]} subdistrict', size=12, fontweight='bold')
        axs[i].set_xlabel('time', size=10, fontweight='bold')
        axs[i].set_ylabel('number of individuals', size=10, fontweight='bold', labelpad=10)

        axs[i].tick_params(labelsize=8)

    # Vaccination model
    # Get weekly vaccination for each subdist
    model_weekly_vacc_by_subdist = get_coupled_model_weekly_vacc_all_seasons(model_results_list, prep_data, by_subdist=True)

    # Get relevant dates
    all_dates = []
    for season in seasons:
        # Get vaccination season start date (minus 1 week)
        start_date = pd.Timestamp(season - 1, 6, 1) + pd.Timedelta(days=vacc_start - 7)
        # Get the season range - form start date (minus week)
        dates = [start_date + pd.Timedelta(days=7) * i for i in range(27 + 1)]
        # Add to the list
        all_dates.extend(dates)

    for i, subdist in enumerate(model_weekly_vacc_by_subdist):
        # Plot newly infected - model
        axs[i].plot(all_dates, model_weekly_vacc_by_subdist[subdist],
                    label='vaccination model', linewidth=1.5)

        # Plot data
        vacc_data = data_for_fit_v['data_for_fit_subdist'][(subdist, 0)].copy()
        vacc_data.vacc_count += data_for_fit_v['data_for_fit_subdist'][(subdist, 1)].vacc_count
        vacc_data = np.concatenate([[0], vacc_data.vacc_count])
        axs[i].scatter(all_dates, np.tile(vacc_data, reps=len(seasons)), label='vaccination data', c='r', s=3)

    plt.show()


def plot_aggregated_fit_coupled_with_cloud(model_results_list, likelihood_lists, data_for_fit_i, data_for_fit_v,
                                           prep_data, age=None, vacc_start=92, plot_max=False):
    alpha = 0.1
    # Get median realization for each season
    meds = {season: np.argsort(np.array(like_list))[len(like_list) // 2] for season, like_list in likelihood_lists.items()}

    if plot_max:
        meds = {season: np.argmax(np.array(like_list)) for season, like_list in likelihood_lists.items()}

    # Create fig
    plt.figure(figsize=(20, 10))

    # Get relevant dates - for vaccination plot
    all_dates = []
    for season in seasons:
        # Get vaccination season start date (minus 1 week)
        start_date = pd.Timestamp(season - 1, 6, 1) + pd.Timedelta(days=vacc_start - 7)
        # Get the season range - form start date (minus week)
        dates = [start_date + pd.Timedelta(days=7) * i for i in range(27 + 1)]
        # Add to the list
        all_dates.extend(dates)

    # Go over results list
    for res_all_seasons in model_results_list:
        # Influenza model
        # Get data to plot and model results
        # If children
        if age == 0:
            infected_data_for_plot = data_for_fit_i['by_age'][0]
            model_weekly_cases = get_model_weekly_cases_all_seasons(res_all_seasons, prep_data, age=0)
        # If adult
        elif age == 1:
            infected_data_for_plot = data_for_fit_i['by_age'][1]
            model_weekly_cases = get_model_weekly_cases_all_seasons(res_all_seasons, prep_data, age=1)
        # If total
        else:
            infected_data_for_plot = data_for_fit_i['total']
            model_weekly_cases = get_model_weekly_cases_all_seasons(res_all_seasons, prep_data)

        # Plot flu model - all realization
        plt.plot(model_weekly_cases.index, model_weekly_cases.cases, linewidth=0.5, label='influenza model', c='C0', alpha=alpha)

        # Vaccination model
        # Get data to plot and model results
        # If children
        if age == 0:
            vacc_data_for_plot = data_for_fit_v['infected_data_agg_children']
            model_weekly_vacc = get_coupled_model_weekly_vacc_all_seasons(res_all_seasons, prep_data, age=age)
        # If adult
        elif age == 1:
            vacc_data_for_plot = data_for_fit_v['infected_data_agg_adult']
            model_weekly_vacc = get_coupled_model_weekly_vacc_all_seasons(res_all_seasons, prep_data, age=age)
        # If total
        else:
            vacc_data_for_plot = data_for_fit_v['infected_data_agg']
            model_weekly_vacc = get_coupled_model_weekly_vacc_all_seasons(res_all_seasons, prep_data, age=age)

        # Plot vaccination model - all realization
        plt.plot(all_dates, model_weekly_vacc, linewidth=0.5, label='vaccination model', c='C1', alpha=alpha)

    # Plot flu data
    flu_data_plt = plt.scatter(infected_data_for_plot.index, infected_data_for_plot.cases, c='b', label='influenza data')

    # Plot vaccination data
    # Add 0 before the first point - for plotting
    vacc_data_for_plot = np.concatenate([[0], vacc_data_for_plot.vacc_count])
    vacc_data_plt = plt.scatter(all_dates, np.tile(vacc_data_for_plot, reps=len(seasons)), c='r', label='vaccination data')

    # Plot median realizations
    # Initialize a list for vaccination resluts
    vacc_res = [[0]]

    # Go over seasons
    for s, (season, med) in enumerate(meds.items()):
        # Get model results - median result of current season
        med_results = model_results_list[med][s]

        # Influenza model
        # If children
        if age == 0:
            model_weekly_cases = get_model_weekly_cases(med_results, season, age=0)
        # If adult
        elif age == 1:
            model_weekly_cases = get_model_weekly_cases(med_results, season, age=1)
        # If total
        else:
            model_weekly_cases = get_model_weekly_cases(med_results, season)

        # Plot median result for current season
        med_plot_flu = plt.plot(model_weekly_cases.index, model_weekly_cases.cases, linewidth=2.5, label='influenza model', c='C0')

        # Vaccination model
        # If children
        if age == 0:
            model_weekly_vacc = get_coupled_model_weekly_vacc(med_results, prep_data, age=age)
        # If adult
        elif age == 1:
            model_weekly_vacc = get_coupled_model_weekly_vacc(med_results, prep_data, age=age)
        # If total
        else:
            model_weekly_vacc = get_coupled_model_weekly_vacc(med_results, prep_data)

        # Add to the list
        vacc_res.append(model_weekly_vacc)
        vacc_res.append([0])

    # Plot median result for current season - vaccination (plot the concatenation of all seasons)
    med_plot_vacc = plt.plot(all_dates, np.concatenate(vacc_res[:-1]), linewidth=2.5, label='vaccination model', c='C1')

    plt.title(f'Coupled Model - {"total" if age is None else ["children", "adults"][age]}', size=22)
    plt.xlabel('time', size=24, labelpad=5)
    plt.ylabel('number of individuals', size=24, labelpad=5)

    plt.tick_params(labelsize=22)

    plt.legend(handles=[med_plot_flu[0], med_plot_vacc[0], flu_data_plt, vacc_data_plt], fontsize=20, loc=(1.02, 0.75))

    plt.show()


def plot_model_states(model_results, prep_data):
    # Load states
    Si = model_results['Si']
    Vi = model_results['Vi']
    Is = model_results['Is']
    Ia = model_results['Ia']
    Ri = model_results['Ri']

    # Network size
    N = prep_data['N']

    Si_sizes_norm = np.array([len(st) / N for st in Si])
    Vi_sizes_norm = np.array([len(st) / N for st in Vi])
    Is_sizes_norm = np.array([len(st) / N for st in Is])
    Ia_sizes_norm = np.array([len(st) / N for st in Ia])
    Ri_sizes_norm = np.array([len(st) / N for st in Ri])

    ts = np.arange(len(Si))
    plt.figure(figsize=(20, 10))
    plt.plot(ts, Si_sizes_norm, label='Si')
    plt.plot(ts, Vi_sizes_norm, label='Vi')
    plt.plot(ts, Is_sizes_norm, label='Is')
    plt.plot(ts, Ia_sizes_norm, label='Ia')
    plt.plot(ts, Ri_sizes_norm, label='Ri')


    # plt.title()

    plt.xlabel('Time (days)', size=20)
    plt.ylabel('Number of indiviuals', size=20)

    plt.xlim([0, len(Si)-1])
    plt.xticks(ts[::50], size=15)

    # plt.ylim([0,100000])
    # plt.yticks(np.arange(0,120000,20000), size=15)
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1.2, 0.2), size=15)

    plt.legend(fontsize=20, loc='upper right')
    plt.show()


################################################
def plot_fit_by_subdist_coupled_with_cloud(model_results_list, likelihood_lists, data_for_fit_i, data_for_fit_v, prep_data, vacc_start=92,
                                           plot_max=False):
    alpha = 0.1

    # Create figure
    fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(10, 30))
    plt.tight_layout(w_pad=3, h_pad=5)

    # Get subdists names
    subdists_names = {11.0: 'Jerusalem', 41.0: 'Sharon', 42.0: 'Petah-Tikva', 43.0: 'Ramla', 44.0: 'Rehovot', 51.0: 'Tel-Aviv',
                      61.0: 'Ashkelon', 77.0: 'Judea & Samaria'}

    # Get median realization for each season
    meds = {season: np.argsort(np.array(like_list))[len(like_list) // 2] for season, like_list in likelihood_lists.items()}

    if plot_max:
        meds = {season: np.argmax(np.array(like_list)) for season, like_list in likelihood_lists.items()}

    # Get relevant dates - for vaccination plot
    all_dates = []
    for season in seasons:
        # Get vaccination season start date (minus 1 week)
        start_date = pd.Timestamp(season - 1, 6, 1) + pd.Timedelta(days=vacc_start - 7)
        # Get the season range - form start date (minus week)
        dates = [start_date + pd.Timedelta(days=7) * i for i in range(27 + 1)]
        # Add to the list
        all_dates.extend(dates)

    # Go over results list
    for res_all_seasons in model_results_list:
        # Influenza model
        # Get weekly cases for each subdist
        model_weekly_cases_by_subdist = get_model_weekly_cases_all_seasons(res_all_seasons, prep_data, by_subdist=True)

        for i, subdist in enumerate(model_weekly_cases_by_subdist):
            # Plot newly infected - model
            axs[i].plot(model_weekly_cases_by_subdist[subdist].index, model_weekly_cases_by_subdist[subdist].cases,
                        label='influenza model', linewidth=0.5, c='C0', alpha=alpha)
            # axs attributes
            axs[i].set_title(f'{subdists_names[subdist]} subdistrict', size=12, fontweight='bold')
            axs[i].set_xlabel('time', size=10, fontweight='bold')
            axs[i].set_ylabel('number of individuals', size=10, fontweight='bold', labelpad=10)

            axs[i].tick_params(labelsize=8)

        # Vaccination model
        # Get weekly vaccination for each subdist
        model_weekly_vacc_by_subdist = get_coupled_model_weekly_vacc_all_seasons(res_all_seasons, prep_data, by_subdist=True)

        for i, subdist in enumerate(model_weekly_vacc_by_subdist):
            # Plot newly infected - model
            axs[i].plot(all_dates, model_weekly_vacc_by_subdist[subdist],
                        label='vaccination model', linewidth=0.5, c='C1', alpha=alpha)

    # Plot data
    # Plot flu data
    for i, subdist in enumerate(model_weekly_cases_by_subdist):
        data = data_for_fit_i['by_subdist'][(subdist, 0)].copy()
        data.cases += data_for_fit_i['by_subdist'][(subdist, 1)].cases
        flu_data_plt = axs[i].scatter(data.index, data.cases, label='influenza data', c='b', s=3)

    # Plot vaccination data
    for i, subdist in enumerate(model_weekly_vacc_by_subdist):
        vacc_data = data_for_fit_v['data_for_fit_subdist'][(subdist, 0)].copy()
        vacc_data.vacc_count += data_for_fit_v['data_for_fit_subdist'][(subdist, 1)].vacc_count
        vacc_data = np.concatenate([[0], vacc_data.vacc_count])
        vacc_data_plt = axs[i].scatter(all_dates, np.tile(vacc_data, reps=len(seasons)), label='vaccination data', c='r', s=3)

    # Plot median realizations
    # Initialize a list for vaccination resluts
    vacc_res = {subdist: [[0]] for subdist in prep_data['relevant_subdists']}

    # Go over seasons
    for s, (season, med) in enumerate(meds.items()):
        # Get model results - median result of current season
        med_results = model_results_list[med][s]

        # Influenza model
        # Initialize dict for weekly cases by subdist - all arrays of 0s
        Is_by_subdist = {subdist: np.array([0] * 365) for subdist in prep_data['relevant_subdists']}

        # Go over clinics and aggregate by subdists
        for (clinic, age), data in med_results['Is_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            Is_by_subdist[subdist] = Is_by_subdist[subdist] + data

        # Get weekly cases for each subdist
        model_weekly_cases_by_subdist = {}
        for subdist, data in Is_by_subdist.items():
            model_weekly_cases_by_subdist[subdist] = get_model_weekly_cases(data, season, short=False, by_subdist=True)

        # Go over subdists and plot
        for i, subdist in enumerate(model_weekly_cases_by_subdist):
            # Plot newly infected - model
            med_plot_flu = axs[i].plot(model_weekly_cases_by_subdist[subdist].index, model_weekly_cases_by_subdist[subdist].cases,
                                       label='influenza model', linewidth=1.5, c='C0')

        # Vaccination model
        # Get weekly vaccination for each subdist
        model_weekly_vacc_by_subdist = get_coupled_model_weekly_vacc(med_results, prep_data, by_subdist=True)

        # Add to the relevant list
        for subdist, weekly_vacc in model_weekly_vacc_by_subdist.items():
            vacc_res[subdist].append(weekly_vacc)
            vacc_res[subdist].append([0])

    # Plot median result for current season - vaccination (plot the concatenation of all seasons)
    for i, subdist in enumerate(model_weekly_vacc_by_subdist):
        # Plot newly infected - model
        med_plot_vacc = axs[i].plot(all_dates, np.concatenate(vacc_res[subdist][:-1]), label='vaccination model',
                                    linewidth=1.5, c='C1')

    plt.legend(handles=[med_plot_flu[0], med_plot_vacc[0], flu_data_plt, vacc_data_plt], fontsize=12, loc=(1.02, 8.4))

    plt.show()


def plot_fit_by_subdist_age_coupled_with_cloud(model_results_list, likelihood_lists, data_for_fit_i, data_for_fit_v, prep_data, vacc_start=92,
                                           plot_max=False):
    alpha = 0.1
    # Create figure
    fig, axs = plt.subplots(nrows=14, ncols=1, figsize=(10, 30*2))
    plt.tight_layout(w_pad=3, h_pad=5)

    # Get subdists names
    subdists_names = {11.0: 'Jerusalem', 41.0: 'Sharon', 42.0: 'Petah-Tikva', 43.0: 'Ramla', 44.0: 'Rehovot', 51.0: 'Tel-Aviv',
                      61.0: 'Ashkelon', 77.0: 'Judea & Samaria'}

    # Get median realization for each season
    meds = {season: np.argsort(np.array(like_list))[len(like_list) // 2] for season, like_list in likelihood_lists.items()}

    if plot_max:
        meds = {season: np.argmax(np.array(like_list)) for season, like_list in likelihood_lists.items()}

    # Get relevant dates - for vaccination plot
    all_dates = []
    for season in seasons:
        # Get vaccination season start date (minus 1 week)
        start_date = pd.Timestamp(season - 1, 6, 1) + pd.Timedelta(days=vacc_start - 7)
        # Get the season range - form start date (minus week)
        dates = [start_date + pd.Timedelta(days=7) * i for i in range(27 + 1)]
        # Add to the list
        all_dates.extend(dates)

    # Go over results list
    for res_all_seasons in model_results_list:
        # Influenza model
        # Get weekly cases for each subdist
        model_weekly_cases_by_subdist_age = get_model_weekly_cases_all_seasons(res_all_seasons, prep_data, by_subdist_age=True)

        for i, (subdist, age) in enumerate(model_weekly_cases_by_subdist_age):
            # Plot newly infected - model
            axs[i].plot(model_weekly_cases_by_subdist_age[(subdist, age)].index, model_weekly_cases_by_subdist_age[(subdist, age)].cases,
                        label='influenza model', linewidth=0.5, c='C0', alpha=alpha)
            # axs attributes
            axs[i].set_title(f'{subdists_names[subdist]} subdistrict - {["children", "adults"][age]}', size=12, fontweight='bold')
            axs[i].set_xlabel('time', size=10, fontweight='bold')
            axs[i].set_ylabel('number of individuals', size=10, fontweight='bold', labelpad=10)

            axs[i].tick_params(labelsize=8)

        # Vaccination model
        # Get weekly vaccination for each subdist
        model_weekly_vacc_by_subdist_age = get_coupled_model_weekly_vacc_all_seasons(res_all_seasons, prep_data, by_subdist_age=True)

        for i, (subdist, age) in enumerate(model_weekly_vacc_by_subdist_age):
            # Plot newly infected - model
            axs[i].plot(all_dates, model_weekly_vacc_by_subdist_age[(subdist, age)],
                        label='vaccination model', linewidth=0.5, c='C1', alpha=alpha)

    # Plot data
    # Plot flu data
    for i, (subdist, age) in enumerate(model_weekly_cases_by_subdist_age):
        data = data_for_fit_i['by_subdist'][(subdist, age)].copy()
        flu_data_plt = axs[i].scatter(data.index, data.cases, label='influenza data', c='b', s=3)

    # Plot vaccination data
    for i, (subdist, age) in enumerate(model_weekly_vacc_by_subdist_age):
        vacc_data = data_for_fit_v['data_for_fit_subdist'][(subdist, age)].copy()
        vacc_data = np.concatenate([[0], vacc_data.vacc_count])
        vacc_data_plt = axs[i].scatter(all_dates, np.tile(vacc_data, reps=len(seasons)), label='vaccination data', c='r', s=3)

    # Plot median realizations
    # Initialize a list for vaccination resluts
    vacc_res = {(subdist, age): [[0]] for subdist, age in prep_data['relevant_subdists_age']}

    # Go over seasons
    for s, (season, med) in enumerate(meds.items()):
        # Get model results - median result of current season
        med_results = model_results_list[med][s]

        # Influenza model
        # Initialize dict for weekly cases by subdist - all arrays of 0s
        Is_by_subdist_age = {(subdist, age): np.array([0] * 365) for subdist, age in prep_data['relevant_subdists_age']}

        # Go over clinics and aggregate by subdists
        for (clinic, age), data in med_results['Is_by_clinic_age'].items():
            subdist = prep_data['clinics_stat_areas'].loc[clinic].subdist
            Is_by_subdist_age[(subdist, age)] = Is_by_subdist_age[(subdist, age)] + data

        # Get weekly cases for each subdist
        model_weekly_cases_by_subdist_age = {}
        for (subdist,age), data in Is_by_subdist_age.items():
            model_weekly_cases_by_subdist_age[(subdist, age)] = get_model_weekly_cases(data, season, short=False, by_subdist_age=True)

        # Go over subdists and plot
        for i, (subdist, age) in enumerate(model_weekly_cases_by_subdist_age):
            # Plot newly infected - model
            med_plot_flu = axs[i].plot(model_weekly_cases_by_subdist_age[(subdist, age)].index, model_weekly_cases_by_subdist_age[(subdist, age)].cases,
                                       label='influenza model', linewidth=1.5, c='C0')

        # Vaccination model
        # Get weekly vaccination for each subdist
        model_weekly_vacc_by_subdist_age = get_coupled_model_weekly_vacc(med_results, prep_data, by_subdist_age=True)

        # Add to the relevant list
        for (subdist, age), weekly_vacc in model_weekly_vacc_by_subdist_age.items():
            vacc_res[(subdist, age)].append(weekly_vacc)
            vacc_res[(subdist, age)].append([0])

    # Plot median result for current season - vaccination (plot the concatenation of all seasons)
    for i, (subdist, age) in enumerate(model_weekly_vacc_by_subdist_age):
        # Plot newly infected - model
        med_plot_vacc = axs[i].plot(all_dates, np.concatenate(vacc_res[(subdist, age)][:-1]), label='vaccination model',
                                    linewidth=1.5, c='C1')

    plt.legend(handles=[med_plot_flu[0], med_plot_vacc[0], flu_data_plt, vacc_data_plt], fontsize=12, loc=(1.02, 8.4))

    plt.show()


def plot_model_states_vacc(model_results, prep_data):
    # Load states
    Sv = model_results['Sv']
    Iv = model_results['Iv']
    Rv = model_results['Rv']
    Vv = model_results['Vv']
    Vn = model_results['Vn']

    # Network size
    N = prep_data['N']

    Sv_sizes_norm = np.array([len(st) / N for st in Sv])
    Iv_sizes_norm = np.array([len(st) / N for st in Iv])
    Rv_sizes_norm = np.array([len(st) / N for st in Rv])
    Vv_sizes_norm = np.array([len(st) / N for st in Vv])
    Vn_sizes_norm = np.array([len(st) / N for st in Vn])

    ts = np.arange(len(Sv))
    plt.figure(figsize=(20, 10))
    plt.plot(ts, Sv_sizes_norm, label='Sv')
    plt.plot(ts, Iv_sizes_norm, label='Iv')
    plt.plot(ts, Rv_sizes_norm, label='Rv')
    plt.plot(ts, Vv_sizes_norm, label='Vv')
    plt.plot(ts, Vn_sizes_norm, label='Vn')


    # plt.title()

    plt.xlabel('Time (days)', size=20)
    plt.ylabel('Number of indiviuals', size=20)

    plt.xlim([0, len(Sv)-1])
    plt.xticks(ts[::50], size=15)

    # plt.ylim([0,100000])
    # plt.yticks(np.arange(0,120000,20000), size=15)
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1.2, 0.2), size=15)

    plt.legend(fontsize=20, loc='upper right')
    plt.show()

