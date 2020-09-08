import pandas as pd
import geopandas as gp
import numpy as np
import time
from tqdm import tqdm
import pickle
from shapely.ops import cascaded_union
from shapely.geometry import Point
from itertools import product
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from geopy.distance import vincenty
from .coordinates_conversion import *

# Load gdf
gdf = gp.read_file('./nafot/gdf.shp')
gdf.set_index('stat_id', inplace=True)
gdf.columns = ['OBJECTID', 'SEMEL_YISH', 'STAT08', 'YISHUV_STA', 'Shem_Yis_1', 'Shape_Leng', 'Shape_Area',
               'DistrictCode', 'SubDistrictCode', 'pop_thou', 'religion', 'age0_17_pcnt', 'geometry']

# Remove irrelevant aggregative stat areas
gdf = gdf[gdf.STAT08 != 0].copy()

# # Load the shapefile and convert from itm to wgs
# stat_area_df = gp.read_file('./nafot/50400_stat_area_2008/stat_2008_NEW_04Nov_1335.shp', encoding='windows-1255')
#
#
# # Polygon coordinates conversion functions
# def poly_conversion(poly):
#     points = []
#     xs, ys = poly.exterior.coords.xy
#     for x, y in zip(xs, ys):
#         lon, lat = itm_to_wgs(x,y)
#         point = (lon, lat)
#         points.append(point)
#     return Polygon(points)
#
#
# def multipolygon_conversion(multipoly):
#     polys = []
#     for poly in multipoly:
#         polys.append(poly_conversion(poly))
#     return MultiPolygon(polys)
#
#
# def shape_conversion(shape):
#     if shape.geom_type == 'MultiPolygon':
#         return multipolygon_conversion(shape)
#     else:
#         return poly_conversion(shape)
#
#
# # Create new stat area df with polygons in wgs instead of itm
# stat_area_wgs_df = stat_area_df.copy()
# stat_area_wgs_df['geometry'] = stat_area_df.apply(lambda row: shape_conversion(row.geometry), axis=1)
#
# # Get the statistical areas df with wgs coordinates
# stat_area_df = stat_area_wgs_df.copy()
#
# """ Create GeoDataFrame of the statistical areas (gdf) with all the data and distinctive index """
#
# # Load extra data about the statistical areas (from excel file)
# extra_df = pd.read_excel('./nafot/Pop_Sex_Age_Religion - edited.xlsx', encoding='windows-1255')
#
# # Get only the wanted columns
# extra_df = extra_df[['SEMEL_YISHUV', 'STAT08', 'DistrictCode', 'DistrictHeb',
#                      'SubDistrictCode', 'SubDistrictHeb', 'MetrCode', 'MetrHeb', 'pop_thou', 'religion', 'age0_17_pcnt']]
#
# # Convert STAT08 into an integer
# extra_df['STAT08'] = extra_df.STAT08.astype('int64', copy=False)
#
# # Create a distinctive id for a statistical area (By concatenating SEMEL_YISHUV and STAT08)
# # Create the ID column in the shapefile df (and set it as an index)
# stat_area_df['stat_id'] = stat_area_df.apply(lambda row: int(str(row['SEMEL_YISH']).strip() +
#                                                              str(row['STAT08']).strip()), axis=1)
# stat_area_df.set_index('stat_id', inplace=True, verify_integrity=True)
#
# # Create the ID column in the extra data df (and set it as an index)
# extra_df['stat_id'] = extra_df.apply(lambda row: int(str(row['SEMEL_YISHUV']).strip() +
#                                                      str(row['STAT08']).strip()), axis=1)
# extra_df.set_index('stat_id', inplace=True, verify_integrity=True)
# extra_df.drop(['SEMEL_YISHUV', 'STAT08'], inplace=True, axis=1)
#
# # Join the data frames
# stat_area_df_joined = stat_area_df.join(extra_df)
#
# # Save as new GeoDataFrame
# gdf = gp.GeoDataFrame(stat_area_df_joined, geometry='geometry')
#
# # Remove no jurisdiction area
# gdf.drop(-20, inplace=True)
#
# # Convert from percent
# gdf.age0_17_pcnt = gdf.age0_17_pcnt/100


# English names of districts
districts_names = {1.0: 'Jerusalem', 2.0: 'Northern', 3.0: 'Haifa', 4.0: 'Central', 5.0: 'Tel Aviv', 60: 'Southern',
                   7.0: 'Judea & Samaria'}

# English names of subdistricts
subdistricts_names = {11.0: 'Jerusalem', 21.0: 'Safed ', 22.0: 'Kinneret', 23.0: "Yizre'el", 24.0: 'Akko',
                      29.0: 'Golan', 31.0: 'Haifa', 32.0: 'Hadera', 41.0: 'Sharon', 42.0: 'Petah Tikva', 43.0: 'Ramla',
                      44.0: 'Rehovot', 51.0: 'Tel Aviv', 61.0: 'Ashkelon', 62.0: "Be'er Sheva", 77.0: 'Judea & Samaria'}

# English names of yeshuvim
yesuvim_names = {code: name for code, name in gdf[['SEMEL_YISH','Shem_Yis_1']].values}

for key, val in yesuvim_names.items():
    if val is None:
        yesuvim_names[key] = 'NONE'

# Convert religion to text
religion_dict = {1: 'Jewish', 2: 'Muslim', 3: 'Christian', 5: 'Druze', 6: 'Other'}
# gdf.religion = gdf.religion.apply(lambda x: religion_dict.get(x))

# Load socio-economic data
# eshcol_data = pd.read_csv('./nafot/social_economic_status.csv', index_col='stat_id')
eshcol_data = pd.read_csv('./nafot/social_economic_status_with_pop.csv', index_col='stat_id')
# Add as column
# gdf['socio-economic_score'] = gdf.apply(lambda row: eshcol_data.eshcol.loc[row.name], axis=1)

# Load education data
education = pd.read_csv('./nafot/education_levels.csv', index_col='stat_id')

# Load religion data
religion_with_orthodox = pd.read_csv('./nafot/religion_with_orthodox.csv', index_col='stat_id')

# Load School locations
school_locations = pd.read_csv('./nafot/school_locations.csv')

# Weekend dates in the data
weekends_dates = ['2012-11-30', '2012-12-01', '2012-12-07', '2012-12-08', '2012-12-14', '2012-12-15', '2012-12-21',
                      '2012-12-22', '2012-12-28', '2012-12-29', '2013-01-04', '2013-01-05', '2013-01-11', '2013-01-12',
                      '2013-01-18', '2013-01-19', '2013-01-25', '2013-01-26' '2013-02-01', '2013-02-02']


# Create/Load percent of children by areas (weighted averge)
# children_percent = {}
# for area_col in ['DistrictCode', 'SubDistrictCode', 'SEMEL_YISH']:
#     ids = pd.unique(gdf[area_col])
#     children_percent[area_col] = {area_code: sum([row.age0_17_pcnt*(row.pop_thou/gdf[gdf[area_col]==area_code].pop_thou.sum())
#                                                   for _, row in gdf[gdf[area_col]==area_code].iterrows()])
#                                   for area_code in ids}

# Read from pickle
with open('./nafot/children_percent.pickle', 'rb') as pickle_in:
    children_percent = pickle.load(pickle_in)


''' Create aggregated polygons and GeoDataFrames for each level '''


def get_districts_gdf():
    """ Get a GeoDataFrame with districts' polygons """
    # Create Districts polygons
    # Get all the districts codes
    district_codes = pd.unique(gdf[~gdf.DistrictCode.isnull()].DistrictCode)

    # Create a dictionary of the districts polygons
    district_polygons = {code: cascaded_union(gdf[gdf.DistrictCode == code].geometry.values) for code in
                         district_codes}

    # Create District GeoDataFrame
    districts_gdf = pd.DataFrame(district_codes, columns=['DistrictCode'])
    districts_gdf['geometry'] = districts_gdf.apply(lambda row: district_polygons[row.DistrictCode], axis=1)
    districts_gdf.set_index('DistrictCode', inplace=True)
    districts_gdf = gp.GeoDataFrame(districts_gdf, geometry='geometry')

    return districts_gdf


def get_subdistricts_gdf():
    """ Get a GeoDataFrame with subdistricts' polygons """
    # Create SubDistricts polygons
    # Get all the subdistricts codes
    subdistrict_codes = pd.unique(gdf[~gdf.SubDistrictCode.isnull()].SubDistrictCode)

    # Create a dictionary of the subdistricts polygons
    subdistrict_polygons = {code: cascaded_union(gdf[gdf.SubDistrictCode == code].geometry.values) for code in
                            subdistrict_codes}

    # Create SubDistrict GeoDataFrame
    subdistricts_gdf = pd.DataFrame(subdistrict_codes, columns=['SubDistrictCode'])
    subdistricts_gdf['geometry'] = subdistricts_gdf.apply(lambda row: subdistrict_polygons[row.SubDistrictCode], axis=1)
    subdistricts_gdf.set_index('SubDistrictCode', inplace=True)
    subdistricts_gdf = gp.GeoDataFrame(subdistricts_gdf, geometry='geometry')

    return subdistricts_gdf


def get_yeshuvim_gdf():
    """ Get a GeoDataFrame with yeshuvim polygons """
    # Get all the districts codes
    yeshuvim_codes = pd.unique(gdf[~gdf.SEMEL_YISH.isnull()].SEMEL_YISH)

    # Create a dictionary of the districts polygons
    yeshuvim_polygons = {code: cascaded_union(gdf[gdf.SEMEL_YISH == code].geometry.values) for code in yeshuvim_codes}

    yeshuvim_gdf = pd.DataFrame(yeshuvim_codes, columns=['SEMEL_YISH'])
    yeshuvim_gdf['geometry'] = yeshuvim_gdf.apply(lambda row: yeshuvim_polygons[row.SEMEL_YISH], axis=1)
    yeshuvim_gdf.set_index('SEMEL_YISH', inplace=True)
    yeshuvim_gdf = gp.GeoDataFrame(yeshuvim_gdf, geometry='geometry')

    return yeshuvim_gdf


def get_stat_areas_gdf():
    """ Get a GeoDataFrame with stat_areas polygons """
    stat_areas_gdf = gdf[['geometry']]
    return stat_areas_gdf


def get_single_stat_araes_codes():
    """ Get a list of stat_areas without hierarchy """
    single_stat_codes = gdf[gdf.DistrictCode.isnull()].index.values
    return single_stat_codes


''' Create Hierarchical lists '''


def get_hierarchical_list(first_level, second_level=None):
    """ Get hierarchical_list of 2 levels. Select 2 levels from: "district", "subdistrict", "yeshuv"
        and "stat_area" """
    if first_level == 'district':
        district_codes = pd.unique(gdf[~gdf.DistrictCode.isnull()].DistrictCode)
        if second_level == 'subdistrict':
            # Create a dict containing a list of SubDistrict for each District
            districts_sub = {code: pd.unique(gdf[gdf.DistrictCode == code].SubDistrictCode.values) for code in
                             district_codes}
            return districts_sub

        elif second_level == 'stat_area':
            # Create a dict containing a list of stat_areas for each District
            districts_stat = {code: pd.unique(gdf[gdf.DistrictCode == code].index.values) for code in district_codes}
            return districts_stat

    if first_level == 'subdistrict':
        subdistrict_codes = pd.unique(gdf[~gdf.SubDistrictCode.isnull()].SubDistrictCode)
        if second_level == 'yeshuv':
            # Create a dict containing a list of Yeshuvim for each SubDistrict
            subdistrict_yesuvim = {code: pd.unique(gdf[gdf.SubDistrictCode == code].SEMEL_YISH.values) for code in
                                   subdistrict_codes}
            return subdistrict_yesuvim

        elif (second_level == 'stat_area') or (second_level is None):
            # Create a dict containing a list of stat_areas for each SubDistrict
            subdistrict_stats = {code: pd.unique(gdf[gdf.SubDistrictCode == code].index.values) for code in
                                 subdistrict_codes}
            return subdistrict_stats

    if first_level == 'yeshuv':
        yeshuvim_codes = pd.unique(gdf[~gdf.SEMEL_YISH.isnull()].SEMEL_YISH)
        # Create a dict containing a list of Statistical areas for each Yeshuv
        yeshuvim_stat = {code: pd.unique(gdf[gdf.SEMEL_YISH == code].index.values) for code in yeshuvim_codes}
        return yeshuvim_stat


def get_population_dict(level):
    """Returns the population dictionary by 'level' (stat_area/yeshuv/subsitrict/district/religion/social-economic/
    religion_with_orthodox/education)"""
    # Getting the relevant level area-ids
    if level in ['district', 'subdistrict', 'yeshuv']:
        # Get codes column names in gdf
        code_col_names = {'district': 'DistrictCode', 'subdistrict': 'SubDistrictCode', 'yeshuv': 'SEMEL_YISH'}

        # If the level is not stat area, add the relevant level to the location data
        area_col = code_col_names[level]

        # Get a list of the area's ids
        ids = pd.unique(gdf[area_col])

        # Remove NaN
        ids = np.array(ids[~np.isnan(ids)])

    elif level == 'stat_area':
        ids = np.array(gdf.index.values)

    elif level == 'social-economic':
        ids = np.array(range(1, 21))

    elif level == 'religion':
        ids = np.array(religion_dict.values())

    elif level == 'religion_with_orthodox':
        ids = np.array(religion_with_orthodox.columns[:-1])

    elif level == 'education':
        ids = np.array(education.columns[:-1])

    # Getting areas population
    if level in ['district', 'subdistrict', 'yeshuv']:
        gb_gdf = gdf.groupby(code_col_names[level])
        population_dict = {area_id: gb_gdf.pop_thou.sum().loc[area_id] * 1000 for area_id in ids}

    elif level == 'stat_area':
        population_dict = {area_id: gdf.iloc[i].pop_thou * 1000 for i, area_id in enumerate(ids)}

    elif level == 'social-economic':
        gb_gdf = eshcol_data.groupby('eshcol')
        population_dict = {eshcol: gb_gdf.pop_thou.sum().loc[eshcol] * 1000 for eshcol in ids}

    elif level == 'religion':
        gb_gdf = gdf.groupby('religion')
        population_dict = {rel: gb_gdf.pop_thou.sum().loc[rel] * 1000 for rel in ids}

    elif level == 'religion_with_orthodox':
        population_dict = {rel: ((religion_with_orthodox[rel] * religion_with_orthodox.pop_thou).sum() * 1000) for rel
                           in ids}

    elif level == 'education':
        population_dict = {ed_lvl: ((education[ed_lvl] * education.pop_thou).sum() * 1000) for ed_lvl in ids}

    return population_dict

''' Find statistical area for a given point (longtitude and latitude)'''

# Pre-processing
# Get subdistricts and stat_areas gdfs
subdistricts_gdf = get_subdistricts_gdf()
stat_areas_gdf = get_stat_areas_gdf()

# Get subdistrict codes
subdistrict_codes = pd.unique(gdf[~gdf.SubDistrictCode.isnull()].SubDistrictCode)

# Get single stat_area codes
single_stat_codes = get_single_stat_araes_codes()

# Get arrays of subdistricts' ids and polygons
subdistrict_ids = subdistricts_gdf.index.copy().values
subdistrict_polys = subdistricts_gdf.geometry.copy().values

# Get arrays of stat_areas' ids and polygons
stat_ids = stat_areas_gdf.index.copy().values
stat_polys = stat_areas_gdf.geometry.copy().values

# Get stat areas without subdistrict
stat_polys_singles = stat_polys.copy()[[np.where(stat_ids == code)[0][0] for code in single_stat_codes]]

# Get a dict containing a list of stat_areas for each SubDistrict
subdistrict_stat = get_hierarchical_list('subdistrict', 'stat_area')

# Create a dict containing a list of stat_areas' polygons for each SubDistrict
subdistrict_stat_polys = {subdistrict: stat_areas_gdf.geometry.loc[subdistrict_stat[subdistrict]].copy().values
                          for subdistrict in subdistrict_codes}


def get_stat_area(longtitude, latitude):
    # Create a Point object
    point = Point(longtitude, latitude)

    # Get the subdistrict
    subdistrict = which_subdistrict(point)

    # Get the stat area
    stat_area = which_stat(point, subdistrict)

    return stat_area


def which_subdistrict(point):
    # Go over the subdistricts
    for i, poly in enumerate(subdistrict_polys):
        if poly.contains(point):
            return subdistrict_ids[i]


def which_stat(point, subdistrict=None):
    # Get subdistricts to check
    if subdistrict:
        stat_areas = subdistrict_stat_polys[subdistrict]
    # Go over the stat_areas
        for i, poly in enumerate(stat_areas):
            if poly.contains(point):
                return subdistrict_stat[subdistrict][i]

    else:
        stat_areas = stat_polys_singles
        # Go over the stat_areas
        for i, poly in enumerate(stat_areas):
            if poly.contains(point):
                return single_stat_codes[i]


def add_stat_area_column(loc_data, path=True):
    """ Receives location data and adds stat area column.
    Exports the result to a new csv file. If path=True loc_data is a path to the location data"""

    if path:
        # Read the csv file into a data frame
        loc_data = pd.read_csv(loc_data, encoding='windows-1255')

    # Getting the start time
    start_time = time.time()
    print('Start time: ' + time.ctime())

    # Add stat area column
    loc_data['stat_area_id'] = loc_data.apply(lambda row: get_stat_area(row.longtitude, row.latitude), axis=1)

    # Getting the end time
    end_time = time.time()

    # Getting thr run time in hours
    run_time = (end_time - start_time) / 3600

    # Getting the minutes and seconds
    hours = int(run_time)
    minutes = int((run_time - hours) * 60)
    seconds = int((((run_time - hours) * 60) - minutes) * 60)

    print('End time: ' + time.ctime())
    print('Run time: ' + str(hours) + ' hours ' + str(minutes) + ' minutes ' + str(seconds) + ' seconds')

    if path:
        # Export to csv
        loc_data.to_csv('{}_with_stat.csv'.format(loc_data[:-4]), index=False)
    else:
        return loc_data


def get_home_stat_area(loc_data, path=True):
    """Receives location data and returns a dictionary the home stat area: {imsi: home stat_area_id}"""
    if path:
        # Read the csv file into a data frame
        loc_data = pd.read_csv(loc_data, encoding='windows-1255')

    # Creating a dict to save the home stat area
    home_area = {}

    # Defining night hours - 20:00 - 08:00 by half hour index
    night_hours = list(range(16 + 1))
    night_hours.extend(list(range(40, 47 + 1)))

    # Getting only calls at night - 20:00 - 08:00
    night_data = loc_data[loc_data['halfhouridx'].isin(night_hours)]

    # Creating a list of the imsi and areas
    imsi_list = list(pd.unique(night_data['imsi']))
    areas_list = pd.unique(night_data['stat_area_id'])

    # Creating a dict to save n, x1, x2 for each imsi - key: imsi, value:
    # ((n, x1, x2), (first stat area name, second stat area name))
    poll_dist = {}

    # Go over the imsi list
    for curr_imsi in imsi_list:
        # creating a histogram of calls from each stat_area
        areas_count = {area_id: 0 for area_id in areas_list}

        # creating a df of the relevant imsi
        imsi_data = night_data[night_data['imsi'] == curr_imsi]

        # update the histogram
        for area_id in imsi_data['stat_area_id']:
            areas_count[area_id] += 1

        # Getting the numbers of calls at night
        n = len(imsi_data.index)

        # If number of calls at night is greater than 50
        if len(imsi_data.index) > 50:

            # getting the stat_area with largest number of calls (name, number of calls)
            first_area_name = max(areas_count, key=areas_count.get)
            first_area = (first_area_name, areas_count.pop(first_area_name))

            # getting the stat_area with second largest number of calls (name, number of calls)
            second_area_name = max(areas_count, key=areas_count.get)
            second_area = (second_area_name, areas_count.pop(second_area_name))

            # getting the number of calls in the first and second stat_area
            x1 = first_area[1]
            x2 = second_area[1]

            # Saving n, x1 and x2 for each imsi
            poll_dist[curr_imsi] = ((n, x1, x2), (first_area[0], second_area[0]))

        # if the numbers of calls at night is lower than 50
        else:
            home_area[curr_imsi] = 'NotDetermined'

    # Defining a threshold and significance level (in percents)
    significance = 25  # in %

    # Go over the poll distributions and run a simulation for each imsi
    for curr_imsi in poll_dist:  # go over the dict keys
        # getting the estimated values
        n, x1, x2 = poll_dist[curr_imsi][0]
        diff = x1 - x2

        # p - the average of p1 and p2
        p = (x1 + x2) / (2 * n)

        # The simulation
        X = np.random.multinomial(n, [p, p, 1 - 2 * p], size=10000)
        # Calculating the proportions differences
        diffs = X[:,0] - X[:,1]

        # The cutoffs are the significance% and (100-significance) percentiles
        upper_cutoff = np.percentile(diffs, 100 - significance / 2)
        lower_cutoff = np.percentile(diffs, significance / 2)

        # if the difference between x1 and x2 in the data is higher than the upper cutoff
        # then the first stat_area is the home stat area
        if diff > upper_cutoff:
            home_area[curr_imsi] = poll_dist[curr_imsi][1][0]

        # if the difference between x1 and x2 in the data is lower than the lower cutoff
        # then the second stat_area is the home stat area
        elif diff < lower_cutoff:
            home_area[curr_imsi] = poll_dist[curr_imsi][1][1]

        # if the difference between x1 and x2 in the data is between the lower and uppper cutoffs
        # then we cannot determine a home stat area
        else:
            home_area[curr_imsi] = 'NotDetermined'

    home_stat_area_data = pd.DataFrame(list(home_area.items()), columns=['imsi', 'home_stat_area'])

    return home_stat_area_data


def calculate_visit_matrix(loc_data, home_area_data):
    """Receives location data and home area data and returns the visit matrix"""
    # Get only data in active hours 6:00 - 00:00
    active_hours = list(range(12, 47 + 1))
    active_hours_data = loc_data[loc_data['halfhouridx'].isin(active_hours)]

    # Get a list of the stat_area's ids
    stat_ids = list(gdf.index.values)

    # Initializing the meeting matrix to all zeros
    matrix_A = np.zeros((len(stat_ids), len(stat_ids)), dtype=float)

    # Go over the stat_areas
    for i, curr_stat in enumerate(stat_ids):
        # get all the imsi which their home stat_area is the current stat_area
        home_imsi = home_area_data[home_area_data['home_stat_area'] == curr_stat]['imsi']

        # Go over the users living in the current stat_area
        for curr_imsi in home_imsi:
            # creating a df of the relevant imsi
            imsi_data = active_hours_data[active_hours_data['imsi'] == curr_imsi]

            # getting the number of calls of the current imsi
            n = len(imsi_data.index)

            # if there are no calls on active hours
            if n == 0:
                continue

            # creating a histogram of call distribution by stat_area - for the current imsi
            stat_area_count = {stat_area_id: 0 for stat_area_id in stat_ids}

            # update the histogram
            for stat_area_id, count in imsi_data['stat_area_id'].value_counts().iteritems():
                stat_area_count[stat_area_id] = count

            # for stat_area_id in imsi_data['stat_area_id']:
            #     stat_area_count[stat_area_id] += 1

            # Update the row corresponding to the current stat_area
            for visiting_stat_area in stat_area_count:
                # row: i - the current home stat_area
                # column: the visiting stat_area
                # value: adding the proportion of calls from visiting stat_area for current imsi
                matrix_A[i, stat_ids.index(visiting_stat_area)] += (stat_area_count[visiting_stat_area] / n)

    return matrix_A


def calculate_visit_matrix_with_age(loc_data, home_area_data):
    # Get only data in active hours 6:00 - 00:00
    active_hours = list(range(12, 47 + 1))
    active_hours_data = loc_data[loc_data['halfhouridx'].isin(active_hours)]

    # Get a list of the stat_area's ids
    stat_ids = list(gdf.index.values)

    # Initializing the meeting matrix to all zeros
    matrix_A = np.zeros((len(stat_ids)*2, len(stat_ids)), dtype=float)

    # Go over the stat_areas
    for i, curr_stat in enumerate(stat_ids):
        # Go over the 2 age groups
        for j in range(2):
            # Get all the imsi which their home stat_area is the current stat_area
            # and their age group is the current age group
            home_imsi = home_area_data[(home_area_data['home_stat_area'] == curr_stat)
                                       & (home_area_data.is_adult == j)]['imsi']

            # Go over the users living in the current stat_area
            for curr_imsi in home_imsi:
                # Creating a df of the relevant imsi
                imsi_data = active_hours_data[active_hours_data['imsi'] == curr_imsi]

                # Getting the number of calls of the current imsi
                n = len(imsi_data.index)

                # If there are no calls on active hours
                if n == 0:
                    continue

                # Creating a histogram of call distribution by stat_area - for the current imsi
                stat_area_count = {stat_area_id: 0 for stat_area_id in stat_ids}

                # Update the histogram
                for stat_area_id, count in imsi_data['stat_area_id'].value_counts().iteritems():
                    stat_area_count[stat_area_id] = count

                # Update the row corresponding to the current stat_area
                for visiting_stat_area in stat_area_count:
                    # row: i - the current home stat_area
                    # column: the visiting stat_area
                    # value: adding the proportion of calls from visiting stat_area for current imsi
                    matrix_A[i*2+j, stat_ids.index(visiting_stat_area)] += (stat_area_count[visiting_stat_area] / n)

    return matrix_A


def adjust_visit_matrix_with_age(visit_matrix):
    """Receives a visit matrix with row including age and columns not including age.
    Returns a matrix with rows and columns including age"""

    # Initialize a zero matrix
    visit_mat_adj = np.zeros((visit_matrix.shape[0], visit_matrix.shape[0]))

    # Go over the raw visit matrix and split according to population proportion of each age group
    for i in range(visit_matrix.shape[1]):
        visit_mat_adj[:, i * 2] = visit_matrix[:, i] * gdf.age0_17_pcnt.loc[stat_ids[i]]
        visit_mat_adj[:, i * 2 + 1] = visit_matrix[:, i] * (1 - gdf.age0_17_pcnt.loc[stat_ids[i]])

    return visit_mat_adj


def visits_to_contact_matrix(visits_matrix, level, use_names=False):
    """Receives a visits matrix aggregated by 'level' and returns a contact matrix"""
    #### Level adjustments ####
    if level in ['district', 'subdistrict', 'yeshuv']:
        # Get codes column names in gdf
        code_col_names = {'district': 'DistrictCode', 'subdistrict': 'SubDistrictCode', 'yeshuv': 'SEMEL_YISH'}

        # If the level is not stat area, add the relevant level to the location data
        area_col = code_col_names[level]

        # Get a list of the area's ids
        ids = pd.unique(gdf[area_col])

        # Remove NaN
        ids = np.array(ids[~np.isnan(ids)])

        # Get areas names
        if level == 'district':
            areas_names = districts_names.copy()
        elif level == 'subdistrict':
            areas_names = subdistricts_names.copy()
        elif level == 'yeshuv':
            areas_names = yesuvim_names.copy()

    elif level == 'stat_area':
        ids = np.array(gdf.index.values)

    elif level == 'social-economic':
        ids = np.array(range(1, 21))

    elif level == 'religion':
        ids = np.array(list(religion_dict.values()))

    elif level == 'religion_with_orthodox':
        ids = np.array(religion_with_orthodox.columns[:-1])

    elif level == 'education':
        ids = np.array(education.columns[:-1])

    #### Matrix B ####
    # Normalizing the meeting matrix by dividing each row by it's sum
    matrix_B = visits_matrix.copy()
    for i in range(matrix_B.shape[0]):
        if matrix_B[i].sum() > 0:
            matrix_B[i] = matrix_B[i] / (matrix_B[i].sum())

    #### Matrix C ####
    no_pop = 0
    no_pop2 = 0

    # Multiply each row by the population of the stat_area
    matrix_C = matrix_B.copy()

    # Getting areas population
    if level in ['district', 'subdistrict', 'yeshuv']:
        gb_gdf = gdf.groupby(area_col)
        population_dict = {area_id: gb_gdf.pop_thou.sum().loc[area_id] * 1000 for area_id in ids}

    elif level == 'stat_area':
        population_dict = {area_id: gdf.iloc[i].pop_thou * 1000 for i, area_id in enumerate(ids)}

    elif level == 'social-economic':
        gb_gdf = eshcol_data.groupby('eshcol')
        population_dict = {eshcol: gb_gdf.pop_thou.sum().loc[eshcol] * 1000 for eshcol in ids}

    elif level == 'religion':
        gb_gdf = gdf.groupby('religion')
        population_dict = {rel: gb_gdf.pop_thou.sum().loc[rel] * 1000 for rel in ids}

    elif level == 'religion_with_orthodox':
        population_dict = {rel: ((religion_with_orthodox[rel] * religion_with_orthodox.pop_thou).sum() * 1000) for rel
                           in ids}

    elif level == 'education':
        population_dict = {ed_lvl: ((education[ed_lvl] * education.pop_thou).sum() * 1000) for ed_lvl in ids}

    # Initialize a list for the areas without population data
    no_pop_areas = []

    for i in range(matrix_C.shape[0]):
        # Get the stat_area population (the data is in thousands)
        area_pop = population_dict[ids[i]]
        if np.isnan(area_pop) or area_pop == 0:
            area_pop = 0
            no_pop_areas.append(i)
            no_pop += 1
            if matrix_C[i].sum() > 0:
                no_pop2 += 1

        # Multiply the row by the relevant population
        matrix_C[i] *= area_pop

    print('Number of areas without population data (not home stat area): {}'.format(no_pop))
    print('Number of areas without population data (home stat area): {}'.format(no_pop2))

    # Remove rows and columns without population data
    mask = np.ones(matrix_C.shape[0], dtype=bool)
    mask[no_pop_areas] = False
    remaining_areas = np.arange(matrix_C.shape[0])[mask]
    matrix_C = matrix_C[:, remaining_areas][remaining_areas].copy()
    matrix_B = matrix_B[:, remaining_areas][remaining_areas].copy()

    #### Matrix D ####
    matrix_D = matrix_C.copy()

    # Normalize each column (sum of 1)
    for i in range(matrix_C.shape[1]):
        if matrix_D[:, i].sum() > 0:
            matrix_D[:, i] = matrix_D[:, i] / (matrix_D[:, i].sum())

    #### Meeting Matrix P ####
    meeting_matrix_P = np.zeros(matrix_B.shape, dtype=float)

    for i in tqdm(range(meeting_matrix_P.shape[0])):
        for j in range(meeting_matrix_P.shape[1]):
            meeting_matrix_P[i, j] = np.sum(matrix_B[i] * matrix_B[j] * matrix_D[j])

    # Create a DataFrame
    if level in ['district', 'subdistrict', 'yeshuv'] and use_names:
        names = [areas_names[area_id] for area_id in ids[mask]]
        meeting_matrix_P_df = pd.DataFrame(meeting_matrix_P, index=names, columns=names)

    else:
        meeting_matrix_P_df = pd.DataFrame(meeting_matrix_P, index=ids[mask], columns=ids[mask])

    # Remove rows and columns with 0 only
    meeting_matrix_P_df = meeting_matrix_P_df.applymap(lambda x: np.nan if x == 0 else x)
    meeting_matrix_P_df.dropna(axis=0, how='all', inplace=True)
    meeting_matrix_P_df.dropna(axis=1, how='all', inplace=True)
    #     meeting_matrix_P_df = meeting_matrix_P_df.applymap(lambda x: 0 if np.isnan(x) else x)

    # Normalizing each row to a sum of 1
    meeting_matrix_P_df_norm = meeting_matrix_P_df.values / meeting_matrix_P_df.sum(1).values.reshape(
        (meeting_matrix_P_df.shape[0], 1))

    # Saving to DataFrame
    meeting_matrix_P_df_norm = pd.DataFrame(meeting_matrix_P_df_norm, columns=meeting_matrix_P_df.columns,
                                            index=meeting_matrix_P_df.index)

    return meeting_matrix_P_df_norm


def visits_to_contact_matrix_with_age(visits_matrix, level, use_names=False):
    """Receives a visits matrix aggregated by 'level' and returns a contact matrix"""
    #### Level adjustments ####
    if level in ['district', 'subdistrict', 'yeshuv']:
        # Get codes column names in gdf
        code_col_names = {'district': 'DistrictCode', 'subdistrict': 'SubDistrictCode', 'yeshuv': 'SEMEL_YISH'}

        # If the level is not stat area, add the relevant level to the location data
        area_col = code_col_names[level]

        # Get a list of the area's ids
        ids = pd.unique(gdf[area_col])

        # Remove NaN
        ids = np.array(ids[~np.isnan(ids)])

        # Get areas names
        if level == 'district':
            areas_names = districts_names.copy()
        elif level == 'subdistrict':
            areas_names = subdistricts_names.copy()
        elif level == 'yeshuv':
            areas_names = yesuvim_names.copy()

    elif level == 'stat_area':
        ids = np.array(gdf.index.values)

    elif level == 'social-economic':
        ids = np.array(range(1, 21))

    elif level == 'religion':
        ids = np.array(religion_dict.values())

    elif level == 'religion_with_orthodox':
        ids = np.array(religion_with_orthodox.columns[:-1])

    elif level == 'education':
        ids = np.array(education.columns[:-1])

    # Create a list for ids with age group
    ids_with_age = list(product(ids, [0, 1]))[:]

    #### Matrix B ####
    # Normalizing the meeting matrix by dividing each row by it's sum
    matrix_B = visits_matrix.copy()
    for i in range(matrix_B.shape[0]):
        if matrix_B[i].sum() > 0:
            matrix_B[i] = matrix_B[i] / (matrix_B[i].sum())

    #### Matrix C ####
    no_pop = 0
    no_pop2 = 0

    # Multiply each row by the population of the stat_area
    matrix_C = matrix_B.copy()

    # Getting areas population
    if level in ['district', 'subdistrict', 'yeshuv']:
        # Getting areas population
        gb_gdf = gdf.groupby(area_col)
        population_dict = {(area_id, age):
                               children_percent[area_col][area_id] * gb_gdf.pop_thou.sum().loc[area_id] * 1000 if age == 0 else
                               (1 - children_percent[area_col][area_id]) * gb_gdf.pop_thou.sum().loc[area_id] * 1000
                           for area_id, age in ids_with_age}

    elif level == 'stat_area':
        population_dict = {
        (area_id, age): gdf.iloc[i // 2].pop_thou * gdf.iloc[i // 2].age0_17_pcnt * 1000 if age == 0 else
            gdf.iloc[i // 2].pop_thou * (1 - gdf.iloc[i // 2].age0_17_pcnt) * 1000
            for i, (area_id, age) in enumerate(ids_with_age)}

    ############ TODO: add age group
    # elif level == 'social-economic':
    #     gb_gdf = eshcol_data.groupby('eshcol')
    #     population_dict = {eshcol: gb_gdf.pop_thou.sum().loc[eshcol] * 1000 for eshcol in ids}
    #
    # elif level == 'religion':
    #     gb_gdf = gdf.groupby('religion')
    #     population_dict = {rel: gb_gdf.pop_thou.sum().loc[rel] * 1000 for rel in ids}
    #
    # elif level == 'religion_with_orthodox':
    #     population_dict = {rel: ((religion_with_orthodox[rel] * religion_with_orthodox.pop_thou).sum() * 1000) for rel
    #                        in ids}
    #
    # elif level == 'education':
    #     population_dict = {ed_lvl: ((education[ed_lvl] * education.pop_thou).sum() * 1000) for ed_lvl in ids}

    # Initialize a list for the areas without population data
    no_pop_areas = []

    for i in range(matrix_C.shape[0]):
        # Get the stat_area population (the data is in thousands)
        area_pop = population_dict[ids_with_age[i]] # [(ids[i//2], i % 2)]
        if np.isnan(area_pop) or area_pop == 0:
            area_pop = 0
            no_pop_areas.append(i)
            no_pop += 1
            if matrix_C[i].sum() > 0:
                no_pop2 += 1

        # Multiply the row by the relevant population
        matrix_C[i] *= area_pop

    print('Number of areas without population data (not home stat area): {}'.format(no_pop))
    print('Number of areas without population data (home stat area): {}'.format(no_pop2))

    # Remove rows and columns without population data
    mask = np.ones(matrix_C.shape[0], dtype=bool)
    mask[no_pop_areas] = False
    remaining_areas = np.arange(matrix_C.shape[0])[mask]
    matrix_C = matrix_C[:, remaining_areas][remaining_areas].copy()
    matrix_B = matrix_B[:, remaining_areas][remaining_areas].copy()

    #### Matrix D ####
    matrix_D = matrix_C.copy()

    # Normalize each column (sum of 1)
    for i in range(matrix_C.shape[1]):
        if matrix_D[:, i].sum() > 0:
            matrix_D[:, i] = matrix_D[:, i] / (matrix_D[:, i].sum())

    #### Meeting Matrix P ####
    meeting_matrix_P = np.zeros(matrix_B.shape, dtype=float)

    for i in tqdm(range(meeting_matrix_P.shape[0])):
        for j in range(meeting_matrix_P.shape[1]):
            meeting_matrix_P[i, j] = np.sum(matrix_B[i] * matrix_B[j] * matrix_D[j])

    # TODO: FIX
    # Create a DataFrame
    # if level in ['district', 'subdistrict', 'yeshuv'] and use_names:
    #     names = [areas_names[area_id] for area_id in ids[mask]]
    #     meeting_matrix_P_df = pd.DataFrame(meeting_matrix_P, index=names, columns=names)

    else:
        # meeting_matrix_P_df = pd.DataFrame(meeting_matrix_P, index=ids_with_age[mask], columns=ids_with_age[mask])
        headers = [ids_with_age[i] for i in remaining_areas]
        meeting_matrix_P_df = pd.DataFrame(meeting_matrix_P, index=headers, columns=headers)

    # Remove rows and columns with 0 only
    meeting_matrix_P_df = meeting_matrix_P_df.applymap(lambda x: np.nan if x == 0 else x)
    meeting_matrix_P_df.dropna(axis=0, how='all', inplace=True)
    meeting_matrix_P_df.dropna(axis=1, how='all', inplace=True)
    #     meeting_matrix_P_df = meeting_matrix_P_df.applymap(lambda x: 0 if np.isnan(x) else x)

    # Normalizing each row to a sum of 1
    meeting_matrix_P_df_norm = meeting_matrix_P_df.values / meeting_matrix_P_df.sum(1).values.reshape(
        (meeting_matrix_P_df.shape[0], 1))

    # Saving to DataFrame
    meeting_matrix_P_df_norm = pd.DataFrame(meeting_matrix_P_df_norm, columns=meeting_matrix_P_df.columns,
                                            index=meeting_matrix_P_df.index)

    return meeting_matrix_P_df_norm


def visits_to_contact_matrix_with_age_updated(visits_matrix, level, use_names=False):
    """Receives a visits matrix aggregated by 'level' and returns a contact matrix"""
    #### Level adjustments ####
    if level in ['district', 'subdistrict', 'yeshuv']:
        # Get codes column names in gdf
        code_col_names = {'district': 'DistrictCode', 'subdistrict': 'SubDistrictCode', 'yeshuv': 'SEMEL_YISH'}

        # If the level is not stat area, add the relevant level to the location data
        area_col = code_col_names[level]

        # Get a list of the area's ids
        ids = pd.unique(gdf[area_col])

        # Remove NaN
        ids = np.array(ids[~np.isnan(ids)])

        # Get areas names
        if level == 'district':
            areas_names = districts_names.copy()
        elif level == 'subdistrict':
            areas_names = subdistricts_names.copy()
        elif level == 'yeshuv':
            areas_names = yesuvim_names.copy()

    elif level == 'stat_area':
        ids = np.array(gdf.index.values)

    elif level == 'social-economic':
        ids = np.array(range(1, 21))

    elif level == 'religion':
        ids = np.array(religion_dict.values())

    elif level == 'religion_with_orthodox':
        ids = np.array(religion_with_orthodox.columns[:-1])

    elif level == 'education':
        ids = np.array(education.columns[:-1])

    # Create a list for ids with age group
    ids_with_age = list(product(ids, [0, 1]))[:]

    #### Matrix B ####
    # Normalizing the meeting matrix by dividing each row by it's sum
    matrix_B = visits_matrix.copy()
    for i in range(matrix_B.shape[0]):
        if matrix_B[i].sum() > 0:
            matrix_B[i] = matrix_B[i] / (matrix_B[i].sum())

    #### Matrix C ####
    no_pop = 0
    no_pop2 = 0

    # Multiply each row by the population of the stat_area
    matrix_C = matrix_B.copy()

    # Getting areas population
    if level in ['district', 'subdistrict', 'yeshuv']:
        # Getting areas population
        gb_gdf = gdf.groupby(area_col)
        population_dict = {(area_id, age):
                               children_percent[area_col][area_id] * gb_gdf.pop_thou.sum().loc[area_id] * 1000 if age == 0 else
                               (1 - children_percent[area_col][area_id]) * gb_gdf.pop_thou.sum().loc[area_id] * 1000
                           for area_id, age in ids_with_age}

    elif level == 'stat_area':
        population_dict = {
            (area_id, age): gdf.iloc[i // 2].pop_thou * gdf.iloc[i // 2].age0_17_pcnt * 1000 if age == 0 else
            gdf.iloc[i // 2].pop_thou * (1 - gdf.iloc[i // 2].age0_17_pcnt) * 1000
            for i, (area_id, age) in enumerate(ids_with_age)}

    no_pop_areas = []

    for i in range(matrix_C.shape[0]):
        # Get the stat_area population (the data is in thousands)
        area_pop = population_dict[ids_with_age[i]]  # [(ids[i//2], i % 2)]
        if np.isnan(area_pop) or area_pop == 0:
            area_pop = 0
            no_pop_areas.append(i)
            no_pop += 1
            if matrix_C[i].sum() > 0:
                no_pop2 += 1

        # Multiply the row by the relevant population
        matrix_C[i] *= area_pop

    print('Number of areas without population data (not home stat area): {}'.format(no_pop))
    print('Number of areas without population data (home stat area): {}'.format(no_pop2))

    # Remove rows and columns without population data
    # For rows
    mask = np.ones(matrix_C.shape[0], dtype=bool)
    mask[no_pop_areas] = False
    remaining_areas = np.arange(matrix_C.shape[0])[mask]

    # For cols
    no_pop_areas_cols = np.unique(np.array([i // 2 for i in no_pop_areas]))
    mask_cols = np.ones(matrix_C.shape[1], dtype=bool)
    mask_cols[list(no_pop_areas_cols)] = False
    remaining_areas_cols = np.arange(matrix_C.shape[1])[mask_cols]

    # Update matrices
    matrix_C = matrix_C[:, remaining_areas_cols][remaining_areas].copy()
    matrix_B = matrix_B[:, remaining_areas_cols][remaining_areas].copy()

    #### Matrix D ####
    matrix_D = matrix_C.copy()

    # Normalize each column (sum of 1)
    for i in range(matrix_C.shape[1]):
        if matrix_D[:, i].sum() > 0:
            matrix_D[:, i] = matrix_D[:, i] / (matrix_D[:, i].sum())

    #### Meeting Matrix P ####
    meeting_matrix_P = np.zeros((matrix_B.shape[0], matrix_B.shape[0]), dtype=float)

    for i in tqdm(range(meeting_matrix_P.shape[0])):
        for j in range(meeting_matrix_P.shape[1]):
            meeting_matrix_P[i, j] = np.sum(matrix_B[i] * matrix_B[j] * matrix_D[j])

    # TODO: FIX
    # Create a DataFrame
    # if level in ['district', 'subdistrict', 'yeshuv'] and use_names:
    #     names = [areas_names[area_id] for area_id in ids[mask]]
    #     meeting_matrix_P_df = pd.DataFrame(meeting_matrix_P, index=names, columns=names)

    else:
        # meeting_matrix_P_df = pd.DataFrame(meeting_matrix_P, index=ids_with_age[mask], columns=ids_with_age[mask])
        headers = [ids_with_age[i] for i in remaining_areas]
        meeting_matrix_P_df = pd.DataFrame(meeting_matrix_P, index=headers, columns=headers)

    # Remove rows and columns with 0 only
    meeting_matrix_P_df = meeting_matrix_P_df.applymap(lambda x: np.nan if x == 0 else x)
    meeting_matrix_P_df.dropna(axis=0, how='all', inplace=True)
    meeting_matrix_P_df.dropna(axis=1, how='all', inplace=True)
    #     meeting_matrix_P_df = meeting_matrix_P_df.applymap(lambda x: 0 if np.isnan(x) else x)

    # Normalizing each row to a sum of 1
    meeting_matrix_P_df_norm = meeting_matrix_P_df.values / meeting_matrix_P_df.sum(1).values.reshape(
        (meeting_matrix_P_df.shape[0], 1))

    # Saving to DataFrame
    meeting_matrix_P_df_norm = pd.DataFrame(meeting_matrix_P_df_norm, columns=meeting_matrix_P_df.columns,
                                            index=meeting_matrix_P_df.index)

    return meeting_matrix_P_df_norm


def is_near_school(lat, lon, thresh, yeshuv=None, return_school_id=False):
    if yeshuv:
        schools = school_locations[school_locations.yeshuv == yeshuv].copy()
        if schools.empty:
            return None if return_school_id else False
    else:
        schools = school_locations

    # Get a vector where True is near a school and False is not
    nearby_schools = schools.latlon.apply(lambda latlon: vincenty((lat, lon),latlon).meters < thresh)

    # Is the point near a school
    near_school = nearby_schools.any()

    if near_school:
        return schools.UNIQ_ID[nearby_schools].values[0] if return_school_id else True

    else:
        return None if return_school_id else False


def get_children_imsi(loc_data, home_data, save_path=None, loc_path=True, home_path=True, _print=True, threshold=0.5):
    """Receives location and home data (or path to them) and returns a list of imsi defined as children (under 18)"""
    if loc_path:
        # Read the csv file into a data frame
        loc_data = pd.read_csv(loc_data, encoding='windows-1255')

    # Remove records without stat area
    loc_data.dropna(inplace=True)

    # Add yeshuv column
    loc_data['yeshuv'] = loc_data.stat_area_id.apply(lambda x: gdf.SEMEL_YISH.loc[x])

    # Get only weekdays data
    loc_data = loc_data[~loc_data.date_stamp.isin(weekends_dates)].copy()

    # Add home stat area data
    # Get home data
    if home_path:
        home_data = pd.read_csv(home_data)
        home_data.set_index('imsi', inplace=True)
        home_data.home_stat_area = home_data.home_stat_area.apply(lambda x: float(x) if x != 'NotDetermined' else x)

    # Adding the centroid of the home stat_area
    home_data = home_data.merge(pd.DataFrame(gdf.geometry.centroid), how='left', left_on='home_stat_area',
                                right_index=True)
    home_data.columns = ['home_stat_area', 'home_center']

    # Add home stat area and center columns
    loc_data = loc_data.merge(home_data, how='left', left_on='imsi', right_index=True)

    # Remove users without home stat area
    loc_data = loc_data[loc_data.home_stat_area != 'NotDetermined'].copy()

    # Getting only signals between 9:00-11:00
    morning_hours = range(18, 22)
    loc_data_morning = loc_data[loc_data.halfhouridx.isin(morning_hours)].copy()

    if _print:
        print('Pre-processing completed'.format(loc_data))

    # Add is_near_school column
    tqdm().pandas()
    loc_data_morning['is_near_school'] =\
        loc_data_morning.progress_apply(lambda row: is_near_school(row.latitude,row.longtitude, 200,row.yeshuv),axis=1)

    # Group by imsi
    gb_imsi_mor = loc_data_morning.groupby('imsi')

    # Filter users that where near school at least 80% of the time
    users_near_school = np.array(sorted(pd.unique(loc_data_morning.imsi))) \
        [np.where((gb_imsi_mor.is_near_school.sum() / gb_imsi_mor.is_near_school.count()) > threshold)[0]]

    # Save users_near_school
    np.save(save_path, users_near_school)

    if _print:
        print('The children imsi list was saved'.format(loc_data))

    return users_near_school
