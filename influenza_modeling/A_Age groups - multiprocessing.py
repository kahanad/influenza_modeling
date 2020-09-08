import pandas as pd
import numpy as np
import nafot
import os
import multiprocessing as mp
import pickle


# loc files names
sample_files_names = os.listdir('../../data/new_samples/with_stat_area')

args_list = []
for file_name in sample_files_names:
    # Paths for the data
    loc_data_path = f'../../data/new_samples/with_stat_area/{file_name}'
    home_data_path = '../../Data/new_samples/home_area/home_area_data_all_updated.csv'
    save_path = f'../../data/new_samples/children_imsi/{file_name[:9]}.npy'

    # Add to the list
    args_list.append((loc_data_path, home_data_path, save_path))


# Define a function for multiprocessing
def get_children_imsi_mp(args):
    print(mp.current_process())
    return nafot.get_children_imsi(args[0], args[1], args[2], loc_path=True, home_path=True, _print=True)


def mp_handler():
    # Create a pool of 24 processes
    pool = mp.Pool(24)
    # Process in parallel
    results = pool.map(get_children_imsi_mp, args_list)
    return results


if __name__ == '__main__':
    res = mp_handler()

    # Saving the results
    with open(f'../../data/new_samples/children_imsi/children_imsi_all_50.pickle', 'wb') as pickle_out:
        pickle.dump(res, pickle_out)
