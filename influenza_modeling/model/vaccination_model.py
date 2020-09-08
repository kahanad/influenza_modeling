# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# from tqdm import tqdm
# import networkx as nx
# import pickle
#
# # Set default paths
# network_path = '../../data/static_network_100K.gpickle'
# contact_matrix_path = '../data/matrix/contact_matrix_final_sample.csv'
# vaccination_data_path = '../../Data/vaccination_data/vaccinated_patients.csv'
# stat_areas_clinics_path = '../../Data/vaccination_data/stat_areas_with_clinics.csv'
# population_by_clinic_path = '../../Data/vaccination_data/population_by_clinic.csv'
#
#
# default_paths = [network_path, contact_matrix_path, vaccination_data_path, stat_areas_clinics_path,
#                  population_by_clinic_path]
#
#
# def data_and_network_prep(network_path=default_paths[0], contact_matrix_path=default_paths[1],
#                           vaccination_data_path=default_paths[2], stat_areas_clinics_path=default_paths[3],
#                           population_by_clinic_path=default_paths[4]):
#     """Receives paths for the data (network, contact matrix, vaccination data, stat_areas-clinic data
#     and clinic population data).
#     Execute the pre-process on these data
#     Returns: processed network, processed vaccination data, list of relevant_clinics, node_clinic dictionary,
#             population_by_clinic dictionary and list of relevant days in season"""
#     #################################
#     # ---------- Network ---------- #
#     #################################
#     # Read the network from gpickle
#     network = nx.read_gpickle(network_path)
#
#     # Remove nodes with number of connection lower than threshold (0 in this case)
#     thresh = 0
#     nodes_to_remove = []
#     for n in network.nodes:
#         if network.degree[n] <= thresh and network.nodes[n]['contacts'] > thresh:
#             nodes_to_remove.append(n)
#
#     # Remove from the network
#     network.remove_nodes_from(nodes_to_remove)
#
#     # Network size
#     N = len(network.nodes)
#
#     ################################
#     # ------ Contact Matrix ------ #
#     ################################
#
#     # Loading stat area contact matrix
#     contact_matrix = pd.read_csv(contact_matrix_path)
#
#     # Setting index, dtype and replacing nan with 0
#     contact_matrix.set_index(contact_matrix.columns[0], inplace=True)
#     contact_matrix.columns = contact_matrix.columns.astype(int)
#     contact_matrix.fillna(0, inplace=True)
#
#     ################################
#     # ----- Vaccination Data ----- #
#     ################################
#
#     # Load vaccination data
#     vaccination_data = pd.read_csv(vaccination_data_path)
#     vaccination_data['vac_date'] = pd.to_datetime(vaccination_data['vac_date'])
#
#     # Remove incomplete seasons (2007 and 2018)
#     vaccination_data = vaccination_data[~vaccination_data.vac_season.isin([2007, 2018])].copy()
#
#     # Short list of dates (1.9-28.2) and days in season
#     dates_2017_short = [pd.Timestamp(2016, 9, 1) + pd.Timedelta(days=1) * i for i in range(181)]
#     day_in_season_short = [(date - pd.datetime(date.year if date.month > 5 else date.year - 1, 6, 1)).days
#                            for date in dates_2017_short]
#
#     ###################################
#     # --- Stat_areas-clinics Data --- #
#     ###################################
#
#     # Read stat_area-clinics data
#     stat_areas_clinics = pd.read_csv(stat_areas_clinics_path)
#
#     # Load data population by clinic data
#     population_by_clinic = pd.read_csv(population_by_clinic_path)
#     population_by_clinic.set_index('clinic_code', inplace=True)
#     population_by_clinic.columns = ['data_population']
#
#     # Get only relevant stat areas
#     stat_areas_clinics = stat_areas_clinics[stat_areas_clinics.stat_area_id.isin(contact_matrix.index)]
#     stat_areas_clinics.set_index('stat_area_id', inplace=True)
#
#     # Create a dictionary of stat area: clinic
#     stat_area_clinics_dict = {stat_area_id: stat_areas_clinics.loc[stat_area_id].clinic_code for stat_area_id in
#                               stat_areas_clinics.index}
#
#     # Create a list of relevant clinics
#     relevant_clinics = stat_areas_clinics.clinic_code.unique()
#
#     # Get population data only data for relevant clinics
#     population_by_clinic = population_by_clinic.loc[relevant_clinics].copy()
#
#     # Initialize a dictionary for the network population by clinic
#     network_pop_by_clinic = dict.fromkeys(relevant_clinics, 0)
#
#     # Add the relevant clinic code for each node and update the population by clinic dictionary
#     for n in network.nodes:
#         network.nodes[n]['clinic'] = stat_area_clinics_dict[network.nodes[n]['area']]
#         network_pop_by_clinic[network.nodes[n]['clinic']] += 1
#
#     # And create a dictionary {node: clinic}
#     node_clinic = {node: network.nodes[node]['clinic'] for node in network.nodes()}
#
#     # Add to the population data frame
#     population_by_clinic['network_population'] = population_by_clinic.index.map(
#         lambda clinic: network_pop_by_clinic[clinic])
#
#     # Calculate the factor between the real data and the network data
#     population_by_clinic['factor'] = population_by_clinic['network_population']/population_by_clinic['data_population']
#
#     prep_data = {'network': network, 'vaccination_data': vaccination_data, 'relevant_clinics': relevant_clinics,
#                  'node_clinic': node_clinic, 'population_by_clinic': population_by_clinic,
#                  'day_in_season_short': day_in_season_short}
#
#     return prep_data
#
#
# def create_data_for_fit(prep_data): #, alpha):
#     # Get prep data
#     vaccination_data = prep_data['vaccination_data']
#     relevant_clinics = prep_data['relevant_clinics']
#     population_by_clinic = prep_data['population_by_clinic']
#     day_in_season_short = prep_data['day_in_season_short']
#
#     ###############################################################
#     # --- Create data for the fit - average vaccination count --- #
#     ###############################################################
#
#     # Get relevant days #- shifted by 1/alpha
#     relevant_days = np.array(day_in_season_short)# + 1/alpha
#
#     # Get only relevant data (according to the short season definition)
#     vaccination_data_short_season = vaccination_data[vaccination_data.vac_day_of_season.isin(set(relevant_days))].copy()
#
#     # Create a dictionary for vaccination count by clinic at each stage (day of the season)
#     data_for_fit = dict.fromkeys(relevant_clinics, [0] * len(day_in_season_short))
#
#     # Go over the clinics
#     for clinic in relevant_clinics:
#         # Get only data of current clinics
#         cur_clinic_data = vaccination_data_short_season[vaccination_data_short_season.clinic_code == clinic]
#
#         # Group by dates and count the number of vaccination at each day
#         cur_clinic_avg_gb = cur_clinic_data.groupby('vac_day_of_season').count()[['random_ID']] / 10
#
#         # Get current clinic vaccination average at each day (including 0 if no vaccination)
#         cur_clinic_avg_vacc = np.array(
#             [cur_clinic_avg_gb.loc[day].random_ID if day in cur_clinic_avg_gb.index else 0 for day in relevant_days])
#
#         # Multiply by the factor between the real and model data
#         vacc_data_adj = cur_clinic_avg_vacc * population_by_clinic['factor'].loc[clinic]
#
#         data_for_fit[clinic] = vacc_data_adj
#
#     ##############################################
#     # --- Aggregated data to fit - for plots --- #
#     ##############################################
#
#     # Get only relevant data (according to the short season definition)
#     # Infected
#     vaccination_data_short_season = vaccination_data[
#         vaccination_data.vac_day_of_season.isin(day_in_season_short)].copy()
#
#     # Exposed
#     vaccination_data_short_season_exposed = vaccination_data[
#         vaccination_data.vac_day_of_season.isin(relevant_days)].copy()
#
#     # Group by day of season and calculate the mean number of vaccination per day
#     infected_data_agg = (vaccination_data_short_season.groupby('vac_day_of_season').count()
#                               ['random_ID'] / 10).values
#     exposed_data_agg = (vaccination_data_short_season_exposed.groupby('vac_day_of_season').count()
#                              ['random_ID'] / 10).values
#
#     # Multiply by the factor between the real and model data
#     total_factor = population_by_clinic['network_population'].sum()/population_by_clinic['data_population'].sum()
#     infected_data_agg = infected_data_agg * total_factor #population_by_clinic['factor']##.loc[clinic]
#     exposed_data_agg = exposed_data_agg * total_factor #population_by_clinic['factor']##.loc[clinic]
#
#     return {'data_for_fit': data_for_fit, 'infected_data_agg': infected_data_agg,
#             'exposed_data_agg': exposed_data_agg}
#
#
# def run_model(parameters, prep_data):
#     ##############################
#     # ----- Initialization ----- #
#     ##############################
#     # Get prep data
#     network, relevant_clinics, node_clinic = prep_data['network'], prep_data['relevant_clinics'],\
#                                              prep_data['node_clinic']
#     season_length = len(prep_data['day_in_season_short'])
#
#     # Get model parameters
#     beta_1 = parameters['beta_1']
#     beta_2 = parameters['beta_2']
#     alpha = parameters['alpha']
#     gamma = parameters['gamma']
#     I_0_size = parameters.get('I_0_size', 0.005)
#
#     # Infected - initialize 0.001 of the population - chose according to susceptibility score
#     I_0 = set(np.random.choice(list(network.nodes), replace=False, size=round(network.number_of_nodes()*I_0_size)))
#
#     # Susceptible
#     S_0 = set(network.nodes) - I_0
#
#     # Exposed - initialize to empty set
#     E_0 = set()
#
#     # Recovered - initialize to empty set
#     R_0 = set()
#
#     # Initialize lists to save all the states
#     S = [S_0]
#     E = [E_0]
#     I = [I_0]
#     R = [R_0]
#
#     # Initialize a list to save the newly infected
#     new_I = [set()]
#
#     # Initialize a list to save the newly exposed
#     new_E = [set()]
#
#     # Initialize a dictionary lambdas_kt
#     lambdas = dict.fromkeys(relevant_clinics, np.array([0.] * season_length))
#
#     # Initialize an array for lambda_t (aggregated)
#     lambdas_agg = np.array([0.] * season_length)
#
#     #############################
#     # ------- Run Model ------- #
#     #############################
#
#     for t in tqdm(range(season_length)):
#         new_exposed_t = set()
#         new_infected_t = set()
#         new_recovered_t = set()
#
#         # --- Infection from friends --- #
#         # Go over the infected individuals
#         for node in I[-1]:
#             for contact in network[node]:
#                 # If the contact is susceptible and not exposed in this stage yet
#                 if contact in S[t-1] and contact not in new_exposed_t:
#                     # Contact is exposed with probability beta_2
#                     new_exposed_t.add(contact) if np.random.rand() < beta_2 else None
#                     # Update lambda_kt
#                     lambdas[node_clinic[node]][t] += beta_2
#                     lambdas_agg[t] += beta_2
#
#         # --- Random Infection --- #
#         # Go over the susceptible individuals
#         for node in S[-1]:
#             # If node was not exposed in this stage yet
#             if node not in new_exposed_t:
#                 # Node is exposed with probability beta_1
#                 new_exposed_t.add(node) if np.random.rand() < beta_1 else None
#                 # Update lambda_kt
#                 lambdas[node_clinic[node]][t] += beta_1
#                 lambdas_agg[t] += beta_1
#
#         # Transmission from E to I
#         for node in E[-1]:
#             # Individuals transmitted from E to I with probability alpha
#             new_infected_t.add(node) if np.random.rand() < alpha else None
#
#         # Transmission from I to R
#         for node in I[-1]:
#             # Individuals transmitted from I to R with probability gamma
#             new_recovered_t.add(node) if np.random.rand() < gamma else None
#
#         # Update stages
#         S.append(S[-1] - new_exposed_t)
#         E.append(E[-1].union(new_exposed_t) - new_infected_t)
#         I.append(I[-1].union(new_infected_t) - new_recovered_t)
#         R.append(R[-1].union(new_recovered_t))
#
#         # Save the newly infected ad exposed
#         new_I.append(new_infected_t)
#         new_E.append(new_exposed_t)
#
#     # Save results to dictionary
#     model_results = {'S': S, 'E': E, 'I': I, 'R': R, 'new_I': new_I, 'new_E': new_E,
#                      'lambdas': lambdas, 'lambdas_agg': lambdas_agg, 'parameters': parameters,
#                      'N': network.number_of_nodes()}
#
#     return model_results
#
#
# def log_likelihood(lambdas, data_for_fit):
#     # Initialize a variable to sum the log-likelihood
#     log_like = 0
#
#     # Go over the clinics
#     for clinic in lambdas:
#         # Sum the log-likelihood for each stage
#         log_like += np.sum(-lambdas[clinic] + data_for_fit[clinic] * np.log(lambdas[clinic]))
#
#     return log_like
#
#
# def log_likelihood_agg(lambdas_agg, data_for_fit_agg):
#     # Sum the log-likelihood for each stage
#     log_like = np.sum(-lambdas_agg + data_for_fit_agg * np.log(lambdas_agg))
#
#     return log_like
#
#
# def get_vacc_coverage_by_clinic(model_results, prep_data):
#     # Get prep data
#     network, relevant_clinics, population_by_clinic = prep_data['network'], prep_data['relevant_clinics'],\
#                                                       prep_data['population_by_clinic']
#
#     # Initialize a dictionary to save the vaccination coverage by clinic
#     vacc_coverage_by_clinic = dict.fromkeys(relevant_clinics, 0)
#
#     # Get vaccinated nodes (I+R in the final stage)
#     vaccinated_nodes = model_results['R'][-1].union(model_results['I'][-1])
#
#     # Go over clinics and count the number of vaccinated
#     for node in vaccinated_nodes:
#         node_clinic = network.nodes[node]['clinic']
#         vacc_coverage_by_clinic[node_clinic] += 1
#
#     # Normalize by clinic network population to receive the coverage %
#     for clinic in vacc_coverage_by_clinic:
#         vacc_coverage_by_clinic[clinic] /= population_by_clinic.loc[clinic].network_population
#
#     # Add the total vaccination coverage
#     vacc_coverage_by_clinic['total'] = len(vaccinated_nodes)/model_results['N']
#
#     return vacc_coverage_by_clinic
#
#
# ###############################
# # ---------- Plots ---------- #
# ###############################
# def plot_aggregated_fit(model_results, data_for_fit):
#     # Get data to plot
#     new_I = model_results['new_I']
#     # new_I, new_E, = model_results['new_I'], model_results['new_E']
#     infected_data_for_plot = data_for_fit['infected_data_agg']
#     # exposed_data_for_plot = data_for_fit['exposed_data_agg']
#
#     # Get newly infected and exposed sizes
#     new_I_sizes = np.array([len(st) for st in new_I[1:]])
#     # new_E_sizes = np.array([len(st) for st in new_E[1:]])
#
#     ts = np.arange(len(new_I_sizes))
#
#     # # Newly exposed
#     # plt.figure(figsize=(20, 10))
#     # plt.plot(ts, new_E_sizes, label='newly exposed - model', linewidth=3)
#     # plt.scatter(ts, exposed_data_for_plot, label='newly exposed - data', c='r')
#     #
#     # plt.title('Newly Exposed', size=22)
#     #
#     # plt.xlabel('Time (days of season)', size=20)
#     # plt.ylabel('Number of individuals', size=20)
#     #
#     # plt.xlim([0, len(new_I_sizes)])
#     # plt.xticks(ts[::50], size=15)
#     #
#     # # Set the y_lim (according the the highest point)
#     # max_ = np.max([(max(new_E_sizes)), max(exposed_data_for_plot)])
#     # lim = np.ceil(max_ / 500) * 500
#     # plt.ylim([0, lim])
#     # plt.yticks(np.arange(0, lim + 500, 500), size=15)
#     #
#     # plt.legend(fontsize=20)
#     #
#     # plt.show()
#     #
#     # print('\n\n')
#
#     # Newly infected
#     plt.figure(figsize=(20, 10))
#     plt.plot(ts, new_I_sizes, label='newly infected - model', linewidth=3)
#     plt.scatter(ts, infected_data_for_plot, label='newly infected - data', c='r')
#
#     plt.title('Newly Infected', size=22)
#
#     plt.xlabel('Time (days of season)', size=20)
#     plt.ylabel('Number of indiviuals', size=20)
#
#     plt.xlim([0, len(new_I_sizes)])
#     plt.xticks(ts[::50], size=15)
#
#     # Set the y_lim (according the the highest point)
#     max_ = np.max([(max(new_I_sizes)), max(infected_data_for_plot)])
#     lim = np.ceil(max_ / 500) * 500
#     plt.ylim([0, lim])
#     plt.yticks(np.arange(0, lim + 500, 500), size=15)
#
#     plt.legend(fontsize=20)
#
#     plt.show()
#
#
# def print_model_final_state(model_results):
#     # Load states
#     S = model_results['S']
#     E = model_results['E']
#     I = model_results['I']
#     R = model_results['R']
#
#     # Network size
#     N = model_results['N']
#     print(f'Population size: {N:,d}')
#     print(f'Number of Susceptible: {len(S[-1]):,d} ({(len(S[-1])/N)*100:.2f}%)')
#     print(f'Number of Exposed: {len(E[-1]):,d} ({(len(I[-1])/N)*100:.2f}%)')
#     print(f'Number of Infected: {len(I[-1]):,d} ({(len(E[-1])/N)*100:.2f}%)')
#     print(f'Number of Recovered: {len(R[-1]):,d} ({(len(R[-1])/N)*100:.2f}%)')
#     print(f'Total number of vaccinated individuals (I+R): {len(I[-1])+len(R[-1]):,d}'
#           f'({((len(I[-1])+len(R[-1]))/N)*100:.2f}%)')
#
#
# def plot_model_states(model_results):
#     # Load states
#     S = model_results['S']
#     E = model_results['E']
#     I = model_results['I']
#     R = model_results['R']
#
#     # Network size
#     N = model_results['N']
#
#     S_sizes_norm = np.array([len(st) / N for st in S])
#     E_sizes_norm = np.array([len(st) / N for st in E])
#     I_sizes_norm = np.array([len(st) / N for st in I])
#     R_sizes_norm = np.array([len(st) / N for st in R])
#
#     ts = np.arange(len(S))
#     plt.figure(figsize=(20, 10))
#     plt.plot(ts, S_sizes_norm, label='S')
#     plt.plot(ts, E_sizes_norm, label='E')
#     plt.plot(ts, I_sizes_norm, label='I')
#     plt.plot(ts, R_sizes_norm, label='R')
#
#     # plt.title()
#
#     plt.xlabel('Time (days)', size=20)
#     plt.ylabel('Number of indiviuals', size=20)
#
#     plt.xlim([0, len(S)-1])
#     plt.xticks(ts[::50], size=15)
#
#     # plt.ylim([0,100000])
#     # plt.yticks(np.arange(0,120000,20000), size=15)
#     plt.ylim([0, 1])
#     plt.yticks(np.arange(0, 1.2, 0.2), size=15)
#
#     plt.legend(fontsize=20, loc='upper right')
#     plt.show()
#
#
# def plot_vacc_coverage(model_results, prep_data):
#     # Get model coverage by clinic
#     model_coverage = get_vacc_coverage_by_clinic(model_results, prep_data).copy()
#     model_coverage.pop('total') # drop total
#     model_coverage = pd.DataFrame(pd.Series(model_coverage), columns=['model_coverage'])
#
#     # Get vaccination coverage_data
#     data_coverage = pd.read_csv('./model/vacc_coverage.csv')
#     data_coverage = data_coverage.merge(model_coverage, left_on='clinic_code', right_index=True)
#
#     # Group by subdist and calculate mean
#     vacc_prop_gb_subdist = data_coverage.groupby('subdist').mean()[['vaccination_coverage', 'model_coverage']]
#     # vacc_prop_gb_clinic = data_coverage.groupby('clinic_code').mean()[['vaccination_coverage', 'model_coverage']]
#
#     # Plot
#     vacc_prop_gb_subdist.plot.bar(figsize=(15, 7))
#     plt.title('Vaccination Coverage by Subdistrict', size=20)
#     plt.xlabel('\nSubdistrict', size=15)
#     plt.ylabel('Vaccination coverage', size=15)
#     plt.xticks(np.arange(7), vacc_prop_gb_subdist.index, rotation='horizontal', size=14)
#     plt.legend(fontsize=15, labels=['data', 'model'], loc=(1.01, 0.87))
#     plt.show()
