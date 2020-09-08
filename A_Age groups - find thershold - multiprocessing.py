import pandas as pd
import numpy as np
import nafot
import os
import multiprocessing as mp
import pickle

# Load a sample loc file
loc_data_sample = pd.read_csv('../../data/new_samples/with_stat_area/sample_01with_stat.csv')

# Get unique imsi
unique_imsi = loc_data_sample.imsi.unique()

# Split imsi list
n = 48 # num of chunks
imsi_list = np.split(unique_imsi[:int(unique_imsi.size/n)*(n-1)], (n-1)) + [unique_imsi[int(unique_imsi.size/n)*(n-1):]]

# Split into chunks by imsi
# loc_chunks = [loc_data_sample[loc_data_sample.imsi.isin(im_lst)].copy() for im_lst in imsi_list]

# list of thresholds to check
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
args_list = [[] for i in range(len(thresholds))]
# for i, chunk in enumerate(loc_chunks):
for i, im_lst in enumerate(imsi_list):
    # Paths for the data
    home_data_path = '../../Data/new_samples/home_area/home_area_data_all_updated.csv'
    save_path = f'../../data/new_samples/children_imsi/sample 01/sample_01with_stat_chunk{i+1}.npy'

    for j, thresh in enumerate(thresholds):
        # Add to the list
        # args_list[j].append((chunk, home_data_path, save_path, thresh))
        args_list[j].append((im_lst, home_data_path, save_path, thresh))


# Define a function for multiprocessing
def get_children_imsi_mp(args):
    print(mp.current_process())
    # return nafot.get_children_imsi(args[0], args[1], args[2], loc_path=False, home_path=True, _print=True)
    loc = loc_data_sample[loc_data_sample.imsi.isin(args[0])].copy()
    # return nafot.get_children_imsi(args[0], args[1], args[2], loc_path=False, home_path=True,
    #                                _print=True, threshold=args[3])
    return nafot.get_children_imsi(loc, args[1], args[2], loc_path=False, home_path=True,
                                   _print=True, threshold=args[3])

def mp_handler():
    results = [[] for j in range(len(thresholds))]
    # Create a pool of 24 processes
    pool = mp.Pool(24)
    for j in range(len(thresholds)):
        # Process in parallel
        results[j] = pool.map(get_children_imsi_mp, args_list[j])
        # Saving the results
        with open(f'../../data/new_samples/children_imsi/children_imsi_thresh{thresholds[j]}.pickle', 'wb') as pickle_out:
            pickle.dump(results[j], pickle_out)
    return results


if __name__ == '__main__':
    res = mp_handler()

    # Saving the results
    with open(f'../../data/new_samples/children_imsi/thresholds_all_results_sm01.pickle', 'wb') as pickle_out:
        pickle.dump(res, pickle_out)
