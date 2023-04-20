from datetime import timedelta
import numpy as np
import pandas as pd
import os
from datetime import datetime
import netCDF4 as nc
import time

# Data
from matplotlib import pyplot as plt

stations_GOM = ['41117',
 '41112',
 '42001',
 '42002',
 '42012',
 '42013',
 '42019',
 '42020',
 '42022',
 '42023',
 '42026',
 '42036',
 '42039',
 '42040',
 '42055',
 '42084',
 '42091',
 '42095',
 '42097',
 '42098',
 '42099',
 'AMRL1',
 'ANPT2',
 'APCF1',
 'APXF1',
 'ARPF1',
 'AWRT2',
 'BABT2',
 'BKBF1',
 'BKTL1',
 'BSCA1',
 'BURL1',
 'BYGL1',
 'BZST2',
 'CAPL1',
 'CARL1',
 'CDRF1',
 'CNBF1',
 'CRTA1',
 'CWAF1',
 'CWBF1',
 'DILA1',
 'DMSF1',
 'EBEF1',
 'EFLA1',
 'EINL1',
 'EMAT2',
 'EPTT2',
 'FHPF1',
 'FMOA1',
 'FMRF1',
 'FPST2',
 'FRDF1',
 'GBIF1',
 'GCTF1',
 'GISL1',
 'GKYF1',
 'GNJT2',
 'GRRT2',
 'GTOT2',
 'GTXF1',
 'HIST2',
 'HIVT2',
 'HREF1',
 'IRDT2',
 'JXUF1',
 'KATP',
 'KBMG1',
 'KBQX',
 'KDLP',
 'KGRY',
 'KGUL',
 'KHHV',
 'KIKT',
 'KTNF1',
 'KVAF',
 'KVOA',
 'KYWF1',
 'LCLL1',
 'LMRF1',
 'LQAT2',
 'LTJF1',
 'LUIT2',
 'MAXT2',
 'MBET2',
 'MBPA1',
 'MCGA1',
 'MGPT2',
 'MHBT2',
 'MTBF1',
 'MYPF1',
 'NCHT2',
 'NFDF1',
 'NUET2',
 'NWCL1',
 'OBLA1',
 'OPTF1',
 'PACF1',
 'PACT2',
 'PCBF1',
 'PCGT2',
 'PCLF1',
 'PCNT2',
 'PILL1',
 'PMAF1',
 'PMNT2',
 'PNLM6',
 'PORT2',
 'PSTL1',
 'PTAT2',
 'PTBM6',
 'PTIT2',
 'PTOA1',
 'RCPT2',
 'RKXF1',
 'RLIT2',
 'RLOT2',
 'RSJT2',
 'RTAT2',
 'SAPF1',
 'SAUF1',
 'SDRT2',
 'SGNT2',
 'SGOF1',
 'SHBL1',
 'SHPF1',
 'SKCF1',
 'SMKF1',
 'SREF1',
 'SRST2',
 'TAQT2',
 'TESL1',
 'TLVT2',
 'TPAF1',
 'TSHF1',
 'TXPT2',
 'TXVT2',
 'UTVT2',
 'VCAF1',
 'VCAT2',
 'VENF1',
 'VTBT2',
 'WIWF1',
 'WKXA1',
 'WPLF1',
 'WYCM6']

# = cleaned_stations_GOM ... includes only buoys which do have data for 2020. Calculated in 2_timestamp_analysis.ipynb
cleaned_stations_GOM = ['41117',
 '41112',
 '42001',
 '42002',
 '42012',
 '42013',
 '42019',
 '42020',
 '42022',
 '42023',
 '42026',
 '42036',
 '42039',
 '42040',
 '42055',
 '42095',
 '42097',
 '42098',
 '42099',
 'AMRL1',
 'ANPT2',
 'APCF1',
 'ARPF1',
 'AWRT2',
 'BABT2',
 'BKTL1',
 'BSCA1',
 'BURL1',
 'BYGL1',
 'BZST2',
 'CAPL1',
 'CARL1',
 'CDRF1',
 'CNBF1',
 'CRTA1',
 'CWAF1',
 'CWBF1',
 'DMSF1',
 'EBEF1',
 'EINL1',
 'EMAT2',
 'EPTT2',
 'FHPF1',
 'FMOA1',
 'FMRF1',
 'FPST2',
 'FRDF1',
 'GBIF1',
 'GCTF1',
 'GISL1',
 'GKYF1',
 'GNJT2',
 'GRRT2',
 'GTOT2',
 'HIST2',
 'HREF1',
 'IRDT2',
 'JXUF1',
 'KTNF1',
 'KYWF1',
 'LCLL1',
 'LMRF1',
 'LTJF1',
 'LUIT2',
 'MBET2',
 'MBPA1',
 'MCGA1',
 'MGPT2',
 'MHBT2',
 'MTBF1',
 'MYPF1',
 'NCHT2',
 'NFDF1',
 'NUET2',
 'NWCL1',
 'OBLA1',
 'OPTF1',
 'PACF1',
 'PACT2',
 'PCBF1',
 'PCGT2',
 'PCLF1',
 'PCNT2',
 'PILL1',
 'PMAF1',
 'PMNT2',
 'PNLM6',
 'PORT2',
 'PSTL1',
 'PTAT2',
 'PTBM6',
 'PTIT2',
 'PTOA1',
 'RCPT2',
 'RLIT2',
 'RLOT2',
 'RSJT2',
 'RTAT2',
 'SAPF1',
 'SAUF1',
 'SDRT2',
 'SGNT2',
 'SGOF1',
 'SHBL1',
 'SHPF1',
 'SKCF1',
 'SMKF1',
 'SREF1',
 'SRST2',
 'TAQT2',
 'TESL1',
 'TLVT2',
 'TPAF1',
 'TSHF1',
 'TXPT2',
 'TXVT2',
 'UTVT2',
 'VCAF1',
 'VCAT2',
 'VENF1',
 'WIWF1',
 'WPLF1',
 'WYCM6']

failed_stations_GOM = ['42084',
                       '42091',
                       'apxf1',
                       'bkbf1',
                       'dila1',
                       'efla1',
                       'gtxf1',
                       'hivt2',
                       'katp',
                       'kbmg1',
                       'kbqx',
                       'kdlp',
                       'kgry',
                       'kgul',
                       'khhv',
                       'kikt',
                       'kvaf',
                       'kvoa',
                       'lqat2',
                       'maxt2',
                       'rkxf1',
                       'vtbt2',
                       'wkxa1']


# Functions
# removes files which do not exist for the given year
# returns cleaned list and list of not available stations
def clean_station_list(station_numbers, year):
    # convert to lowercase
    station_numbers = [each_string.lower() for each_string in station_numbers]

    base_url = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
    cleaned_list = list()
    not_found_list = list()
    for station_number in station_numbers:
        filename = station_number + "h" + year + ".txt.gz"
        url = base_url + filename
        try:
            df_data = pd.read_csv(url, delim_whitespace=True, low_memory=False)
            cleaned_list.append(station_number)
        except BaseException as e:
            print('Failed to get file: {}'.format(e))
            not_found_list.append(station_number)

    return cleaned_list, not_found_list


# get stdmet (=Standard Meteorological) data by station and year
# Params: station_number: string, year: string
# return: if exists: tuple(station_number: str, data: dataframe)
def get_data_file(station_number, year):
    base_url = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
    filename = station_number + "h" + year + ".txt.gz"
    url = base_url + filename
    filepath = f"{os.path.dirname(__file__)}/data/Station_Data_RAW/{station_number}h{year}.csv"

    # Try to read from memory
    if os.path.isfile(filepath):
        df_data = pd.read_csv(filepath, low_memory=False)

        # in memory, the index is duplicated ... remove column "Unnamed: 0"
        df_data.drop(df_data.columns[0], axis=1, inplace=True)
        print("from disc")
        return station_number, df_data  # return as tuple
    else:
        # try to download
        try:
            df_data = pd.read_csv(url, delim_whitespace=True, low_memory=False)
            print("from web")
            # save to memory
            df_data.to_csv(filepath, index=True)

            return station_number, df_data  # return as tuple

        # Not in memory and can not be downloaded
        except BaseException as e:
            print('Failed to get file: {}'.format(e))
            return None


# Parameter: data is the return of get_data_file = tupe of station number and dataframe
# What it does:
# drop row with units
# replace time-columns with one timestamp column as index
# add station name to column header
# return: tuple of filename and modified dataframe
def df_modification(data):
    filename = data[0]
    df = data[1]
    df = df.drop(labels=0, axis=0)  # drop row with units

    # replace time columns with timestamp index
    df['timestamp'] = df['#YY'] + "-" \
                      + df['MM'] + "-" \
                      + df['DD'] + " " \
                      + df['hh'] + ":" \
                      + df['mm']
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M')
    df.drop(columns=['#YY', 'MM', 'DD', 'hh', 'mm'], inplace=True)
    df.set_index('timestamp', inplace=True)

    df.columns += "_" + filename  # rename columns

    return filename, df


# The function build_dataset(station_numbers, year) builds a dataset for one specific year considering the
# given station numbers. Each instance uses a timestamp (index) as identifier.
# Features are all features from all stations. (station id is stored in column name).
# One instance should represent the state of an area (given by station numbers) at a certain timestamp.
# This can be used to forecast future states of the same region.
def build_dataset(station_numbers, year):
    dataframes = list()

    for station_number in station_numbers:
        data = get_data_file(station_number, year)  # load file

        if data is not None:
            current_df = df_modification(data)[1]  # [1] ... df and not the whole tuple
            current_df = current_df.loc[~current_df.index.duplicated(keep='first')]
            dataframes.append(current_df)

    merged_data = pd.concat(dataframes, axis=1, join="outer")  # outer join also includes NaN, inner join removes them
    return merged_data


#Function to get the buoy data and return a well prepared dataset
#What happens:
#   Download data
#   Merge all time-columns to timestamp
#   Ensure that all 1h timestamps but no others exist
#   delete columns which are not covered by ERA5
#   if file does not exist, a empty DF is returned
def get_buoy_data(station_id, year):
    timestamp_filter_list = create_timestamp_list2(year)
    df_NDBC = get_data_file(station_id, year)

    if df_NDBC is None:
        # add empty dataframe
        df_NDBC = pd.DataFrame(columns=[
            f'WDIR_{station_id}',
            f'WSPD_{station_id}',
            f'WVHT_{station_id}',
            f'APD_{station_id}',
            f'MWD_{station_id}',
            f'PRES_{station_id}',
            f'ATMP_{station_id}',
            f'WTMP_{station_id}',
            f'DEWP_{station_id}', ]
        )

    else:
        df_NDBC = df_modification(df_NDBC)[1]
        df_NDBC = replace_with_NaN(df_NDBC)

        # Handling duplicated index:
        num_of_duplicates = df_NDBC.index.duplicated().sum()
        if num_of_duplicates > 0: print(f"Found {num_of_duplicates} duplicates is {station_id}h{year} and removed them!")
        df_NDBC = df_NDBC.loc[~df_NDBC.index.duplicated(keep='first')]

        df_NDBC = df_NDBC.filter(timestamp_filter_list, axis=0)
        df_NDBC.drop([f'GST_{station_id}',
                      f'DPD_{station_id}',
                      f'VIS_{station_id}',
                      f'TIDE_{station_id}'], axis=1, inplace=True)

    # some data rows are missed. Those are filled up with NaN:
    for timestamp in timestamp_filter_list:
        if not timestamp in df_NDBC.index:  # might must be timestamp instead of index
            df_NDBC.loc[timestamp] = [np.NAN] * 9

    df_NDBC.sort_index(inplace=True)
    df_NDBC = df_NDBC.astype(float)  # convert string to float
    return df_NDBC


# Same as get_buoy_data but for multiple files.
# Files of several stations and years are properly merged into one DF
# Furthermore, Download time is meassured and a NaN statistic is created!
def build_NDBC_dataset(STATION_LIST, YEARS):

    time_ref = time.time()

    # create the new dataframe filled with False values
    file_nan_count = pd.DataFrame(
        [[-1 for _ in range(len(YEARS))] for _ in range(len(STATION_LIST))],
        index=STATION_LIST,
        columns=YEARS
    )

    data_list_annual = list()    # each element in this list is a df containing data of one certain year and all stations
    for year in YEARS:

        print("Started with ", year, ". Previous year took:  ", time.time() - time_ref , "seconds")
        time_ref = time.time()

        buoy_data_list = list() # each element in this list is a df containing data of one certain year and one certain station.
        for station in STATION_LIST:
            buoy_data = get_buoy_data(station, year)  # load file
            buoy_data_list.append(buoy_data)

            #Create NaN Statistic
            nan_rate = buoy_data.isna().sum().sum() / (buoy_data.shape[0] * buoy_data.shape[1])
            file_nan_count.loc[station,year] = nan_rate


        merged_buoy_data = pd.concat(buoy_data_list, axis=1, join="outer")  # outer join also includes NaN, inner join removes them
        data_list_annual.append(merged_buoy_data)

    print("Finished downloading - now merging it together!")

    dataset_NDBC = pd.concat(data_list_annual, axis=0)

    return dataset_NDBC, file_nan_count


# Parameter example:
# first_timestamp = datetime.strptime("2020-07-01 00:00", '%Y-%m-%d %H:%M')
# last_timestamp = datetime.strptime("2020-08-01 00:00", '%Y-%m-%d %H:%M')
def create_timestamp_list(first_timestamp, last_timestamp):  # TODO: could be replaced with simplified function below!
    timestamps = list()
    while first_timestamp != last_timestamp:
        timestamps.append(first_timestamp)
        first_timestamp = first_timestamp + timedelta(hours=1)

    return timestamps


# overload function with simplified parameter
def create_timestamp_list2(year):
    first_timestamp = datetime.strptime(f"{year}-01-01 00:00", '%Y-%m-%d %H:%M')
    following_year = str(int(year) + 1)
    last_timestamp = datetime.strptime(f"{following_year}-01-01 00:00", '%Y-%m-%d %H:%M')
    return create_timestamp_list(first_timestamp, last_timestamp)



# Params:
#   df ... Dataframe with numerical data
#   value_list ... list of values which should be replaced by NaN
# Return:
#   dataframe (with additional NaN values)
def replace_with_NaN(df, value_list=None):
    if value_list is None:
        # TODO: doublecheck if this list is complete.
        value_list = ['9999', '999', '99', '9999.0', '999.0', '99.0', '9999.00', '999.00', '99.00', '9.9']

    new_df = df
    for value in value_list:
        new_df.replace(value, np.NaN, inplace=True)
        new_df.replace(float(value), np.NaN, inplace=True)  # Datatype security, since write and read csv changes type!
    return new_df


def print_NaN_statistic(df, heading, silent=False):
    if not silent:
        print("\n", heading)
    num_values = df.shape[0] * df.shape[1]
    if not silent:
        print("number of values: ", num_values)
    num_NaN = df.isna().sum().sum()
    num_NaN_percentage = round(100 * num_NaN / num_values, 2)
    if not silent:
        print("number of NaN values: ", num_NaN, "(", num_NaN_percentage, "%)\n")

    # Print percentage of NaN values per pprint_column
    num_of_rows = len(df.index)
    count_NaN = list()
    percentage_NaN = list()
    features = list()
    for column in df:
        current_num_of_NaN = df[column].isna().sum()
        current_percentage_of_NaN = round(100 * current_num_of_NaN / num_of_rows, 2)

        features.append(column)
        count_NaN.append(current_num_of_NaN)
        percentage_NaN.append(current_percentage_of_NaN)

    NaN_by_feature = pd.DataFrame({'Feature': features, '#NaN': count_NaN, '%NaN': percentage_NaN})
    # display(NaN_by_feature)

    return num_values, num_NaN_percentage, NaN_by_feature


# Drop all columns and rows if they only exist of NaN values!
def drop_NaN_rows_and_cols(df):
    clean_df = df.dropna(axis=1, how='all')  # drop cols if all values are NaN
    clean_df = clean_df.dropna(axis=0, how="all")  # drop rows if all values are NaN
    return clean_df


# ERA5 Stuff
import cdsapi
import math

# API authentication
api_key_path = f"{os.path.dirname(__file__)}/ERA5/.cdsapirc"
api_keyfile = open(api_key_path, "r")
lines = api_keyfile.readlines()
url = lines[0].rstrip().replace("url: ", "")
key = lines[1].rstrip().replace("key: ", "")


# Download ERA5 Dataset of a single location with 1h timestamps.
# It contains all the features that are also covered by NDBC!
# The file will be downloaded and stored in ERA5/ERA5_downloads/singleStations/{station_id}_{year}.nc

default_variables = [
                # '10m_u_component_of_neutral_wind', '10m_v_component_of_neutral_wind',   # TEST !!
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'mean_sea_level_pressure', 'mean_wave_direction',
                'mean_wave_period', 'sea_surface_temperature', 'significant_height_of_total_swell',
            ]
def download_ERA5_singlePoint(station_id, year, variables=None):
    # https://stackoverflow.com/questions/65186216/how-to-download-era5-data-for-specific-location-via-python
    # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

    if variables is None:
        variables = default_variables

    path = f"{os.path.dirname(__file__)}/data/ERA5_downloads/singleStations/{station_id}_{year}.nc"
    print(path)

    metadata = pd.read_csv('../data/my_metadata.csv')
    metadata.set_index("StationID", inplace=True)
    lat = metadata.loc[station_id.upper()]["lat"]
    lon = metadata.loc[station_id.upper()]["lon"]
    coords = [lat, lon] * 2

    c = cdsapi.Client(url, key)

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',

            'variable': variables,

            'year': year,
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
            ],
            'day': [
                '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
            ],
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
            ],
            # 'area': station_point_coord,  # Area: NORTH, WEST, SOUTH, EAST
            'area': coords,
            'format': 'netcdf',
        },
        path
    )


# Reads from memory
# If file does not exists --> download
# returns dataframe
def get_ERA5_singlePoint(station_id, year):
    path = f"{os.path.dirname(__file__)}/data/ERA5_downloads/singleStations/{station_id}_{year}.nc"

    if not os.path.exists(path):
        print(f"donwloading {station_id}_{year}.nc ...")
        download_ERA5_singlePoint(station_id, year)    #only download if not found
        print(f"Completed download of {station_id}_{year}.nc!")

    ds_ERA5 = nc.Dataset(path)  #read from file

    # handle this strange nc data format and store in list
    u10 = ds_ERA5.variables["u10"][:, :, 0].data
    v10 = ds_ERA5.variables["v10"][:, :, 0].data
    d2m = ds_ERA5.variables["d2m"][:, :, 0].data
    t2m = ds_ERA5.variables["t2m"][:, :, 0].data
    msl = ds_ERA5.variables["msl"][:, :, 0].data
    sst = ds_ERA5.variables["sst"][:, :, 0].data
    mwd = ds_ERA5.variables["mwd"][:, :, 0].data
    mwp = ds_ERA5.variables["mwp"][:, :, 0].data
    shts = ds_ERA5.variables["shts"][:, :, 0].data

    # Convert u,v wind components to direction and speed
    WDIR = []
    WSPD = []
    for v, u in zip(v10, u10):
        WDIR.append(calc_WDIR(v, u))
        WSPD.append(calcWSPD(v, u))

    # save to dataframe - same as NDBC data!
    df_ERA5 = pd.DataFrame({
        f'WDIR_{station_id}': WDIR,  # WDIR
        f'WSPD_{station_id}': WSPD,  # WSPD
        f'DEWP_{station_id}': d2m[:, 0],  # D2M
        f'ATMP_{station_id}': t2m[:, 0],  # T2M
        f'PRES_{station_id}': msl[:, 0],  # MSL
        f'WTMP_{station_id}': sst[:, 0],  # SST
        f'WVHT_{station_id}': shts[:, 0],  # SHTS
        f'APD_{station_id}':  mwp[:, 0],  # MWP
        f'MWD_{station_id}':  mwd[:, 0],  # MWD
    })

    # Convert to same units as NDBC uses
    df_ERA5["timestamp"] = create_timestamp_list2(year)
    df_ERA5.set_index('timestamp', inplace=True)
    df_ERA5[f"DEWP_{station_id}"] -= 273.15  # convert to degree Celsius
    df_ERA5[f"ATMP_{station_id}"] -= 273.15
    df_ERA5[f"PRES_{station_id}"] = df_ERA5[f"PRES_{station_id}"] / 100  # convert Pa to hPa
    df_ERA5[f"WTMP_{station_id}"] -= 273.15

    return df_ERA5


# v ... x-axis (North is plus)
# u ... y-axis (East is plus)
def calc_WDIR(v, u):

    # No wind at all
    if v == 0 and u == 0:
        return np.NAN

    # Wind straight to one coordinate direction
    if v == 0 and u > 0:    # North Wind
        return 0

    if v == 0 and u < 0:    # South Wind
        return 180

    if u == 0 and v > 0:    # East Wind
        return 90

    if u == 0 and v < 0:    # West wind
        return 270

    # Angle needs to be calculated
    if u > 0 and v > 0:     # North-East (+u +v)
        alpha = math.degrees(math.atan(u / v))
        return 90 - alpha

    if u < 0 and v > 0:     # South-East (-u +v)
        alpha = math.degrees(math.atan(u / v))
        return 90 + abs(alpha)

    if u < 0 and v < 0:    # South-West (-u, -v)
        alpha = math.degrees(math.atan(u / v))
        return 270 - alpha

    if u > 0 and v < 0:    # North-West (+u, -v)
        alpha = math.degrees(math.atan(u / v))
        return 270 + abs(alpha)


def calcWSPD(v, u):
    return math.sqrt((u * u) + (v * v))  # Pythagoras


# TODO: For directions, take distance across north into account!    #???
def compare_NDBC_ERA5(df_NDBC, df_ERA5):
    features = list(df_NDBC.columns.values)

    delta_absolute = pd.DataFrame({"timestamp": df_NDBC.index})
    delta_absolute.set_index("timestamp", inplace=True)

    delta_relative = delta_absolute.copy(deep=True)

    for feature in features:
        # for directions, also consider distance across north
        if feature.startswith("WDIR") or feature.startswith("MWD"):

            delta_corrected = []
            for value1, value2 in zip(df_NDBC[feature], df_ERA5[feature]):
                delta = abs(value1 - value2)
                if delta > 180:
                    delta = abs(value1 + value2 - 360)
                delta_corrected.append(delta)

            delta_absolute[feature] = delta_corrected

        else:
            delta_absolute[feature] = abs(df_NDBC[feature] - df_ERA5[feature])
            delta_relative[feature] = round(delta_absolute[feature] * 100 / abs(df_NDBC[feature]),
                                            2)  # deviation from NDBC data in %
    return delta_absolute, delta_relative


def plot_parameter_comparison(data_ndbc, data_era5, title):
    #plt.figure().set_figheight(10)
    plt.figure().set_figwidth(30)

    plt.plot(data_ndbc, label="NDBC", linewidth=0.5)
    plt.plot(data_era5, label="ERA5", linewidth=0.5)
    plt.title(title)
    plt.legend()
    plt.show()


















