import pandas as pd
import numpy as np
import nafot.stat_areas as sa
import time

loc_data_path = ''

loc_data = pd.read_csv(loc_data_path)

# Getting the start time
start_time = time.time() 
print('Strat time: ' + time.ctime())

# Add stat area column
loc_data['stat_area'] = loc_data.apply(lambda row: sa.get_stat_area(row.longtitude, row.latitude),axis=1)

# Getting the end time
end_time = time.time() 

# Getting thr run time in hours
run_time = (end_time - start_time) / 3600

# Getting the minutes and seconds
hours = int(run_time)
minutes = int((run_time - hours) * 60)
seconds  = int((((run_time - hours) * 60) - minutes)*60)

print('End time: ' + time.ctime())
print('Run time: ' + str(hours) + ' hours ' + str(minutes) + ' minutes ' + str(seconds) + ' seconds')

# Export to csv
loc_data.to_csv('{}_with_stat'.format(loc_data_path[:-4]), index=False)
