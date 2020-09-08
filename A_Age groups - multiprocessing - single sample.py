import pandas as pd
import numpy as np
import nafot
import os
import multiprocessing as mp
import pickle

# Load a sample loc file
loc_data_sample = pd.read_csv('../../data/new_samples/with_stat_area/sample_00with_stat.csv')

# Get unique imsi
unique_imsi = loc_data_sample.imsi.unique()

# Split imsi list
n = 24 # num of chunks
imsi_list = np.split(unique_imsi[:int(unique_imsi.size/n)*(n-1)], (n-1)) + [unique_imsi[int(unique_imsi.size/n)*(n-1):]]

# Split into chunks by imsi
# loc_chunks = [loc_data_sample[loc_data_sample.imsi.isin(im_lst)].copy() for im_lst in imsi_list]


args_list = []
# for i, chunk in enumerate(loc_chunks):
for i, im_lst in enumerate(imsi_list):
    # Paths for the data
    home_data_path = '../../Data/new_samples/home_area/home_area_data_all_updated.csv'
    save_path = f'../../data/new_samples/children_imsi/sample 00_500m/sample_00_chunk{i+1}_500m.npy'

    # Add to the list
    # args_list.append((chunk, home_data_path, save_path))
    args_list.append((im_lst, home_data_path, save_path))


# Define a function for multiprocessing
def get_children_imsi_mp(args):
    print(mp.current_process())
    # return nafot.get_children_imsi(args[0], args[1], args[2], loc_path=False, home_path=True, _print=True)
    loc = loc_data_sample[loc_data_sample.imsi.isin(args[0])].copy()
    return nafot.get_children_imsi(loc, args[1], args[2], loc_path=False, home_path=True, _print=True)


def mp_handler():
    # Create a pool of 24 processes
    pool = mp.Pool(24)
    # Process in parallel
    results = pool.map(get_children_imsi_mp, args_list)
    return results


if __name__ == '__main__':
    res = mp_handler()

    # Saving the results
    with open(f'../../data/new_samples/children_imsi/sample_00_500_m.pickle', 'wb') as pickle_out:
        pickle.dump(res, pickle_out)
