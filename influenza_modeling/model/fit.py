from scipy.optimize import minimize
from .vaccination_model_with_age import *


def fit(likelihood_func_agg=True, method='Nelder-Mead'):
    # Prepare network and data
    prep_data = data_and_network_prep()

    # Create data for fit
    data_for_fit = create_data_for_fit(prep_data)

    # Initial guess
    initial_guess = np.array([0.0005, 0.02, 1/3, 1/3])

    # Minimize
    best_params = minimize(get_likelihood, x0=initial_guess, method=method,
                           args=(likelihood_func_agg, prep_data, data_for_fit))

    return best_params


def get_likelihood(params, *args):
    # Get additional args
    likelihood_func_agg, prep_data, data_for_fit = args

    # Get params to fit
    cur_params = {'beta_2': params[0], 'gamma': params[1], 'I_0_size': params[3]}

    # # Create data for fit
    # data_for_fit = create_data_for_fit(prep_data, cur_params['alpha'])

    # Run model and get results
    model_results = run_model(cur_params, prep_data)

    # If model result infeasible return MSE = inf
    if not model_results:
        return -10e20

    # Calculate and return the log likelihood (return the negative because we use minimize)
    # If aggregated likelihood function
    if likelihood_func_agg:
        return -log_likelihood_agg(model_results['lambdas_agg'], data_for_fit['infected_data_agg'])
        # return -log_likelihood_agg(model_results['lambdas_agg'], data_for_fit['exposed_data_agg'])

    # If likelihood by clinics
    else:
        return -log_likelihood(model_results['lambdas'], data_for_fit['data_for_fit'])
