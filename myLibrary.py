from datetime import timedelta
import numpy as np
import pandas as pd
import os
from datetime import datetime
import netCDF4 as nc
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import cdsapi
import math
from typing import Union

from dataclasses import dataclass
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from keras import backend as K

#--------------------------------------------------------------------------------------
################## GENERAL STUFF #########################################################
#--------------------------------------------------------------------------------------

measurements = ["WDIR", "WSPD", "WVHT", "APD", "MWD", "PRES", "ATMP", "WTMP", "DEWP"]

#creates a list of all full hour timestamps of the given year
def create_timestamp_list(year):
    first_timestamp = datetime.strptime(f"{year}-01-01 00:00", '%Y-%m-%d %H:%M')
    following_year = str(int(year) + 1)
    last_timestamp = datetime.strptime(f"{following_year}-01-01 00:00", '%Y-%m-%d %H:%M')

    #return create_timestamp_list(first_timestamp, last_timestamp)

    timestamps = list()
    while first_timestamp != last_timestamp:
        timestamps.append(first_timestamp)
        first_timestamp = first_timestamp + timedelta(hours=1)

    return timestamps


#--------------------------------------------------------------------------------------
################## NDBC STUFF #########################################################
#--------------------------------------------------------------------------------------
class NDBC_lib:
    # Data
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
    @staticmethod
    def clean_station_list(stations: list, year):
        # convert to lowercase
        stations = [each_string.lower() for each_string in stations]

        base_url = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
        cleaned_list = list()
        not_found_list = list()
        for station_number in stations:
            filename = station_number + "h" + year + ".txt.gz"
            url = base_url + filename
            try:
                df_data = pd.read_csv(url, delim_whitespace=True, low_memory=False)
                cleaned_list.append(station_number)
            except BaseException as e:
                print(f'Failed to get {filename}: {e}')
                not_found_list.append(station_number)

        return cleaned_list, not_found_list


    # get stdmet (=Standard Meteorological) data by station and year
    # Params: station_id: string, year: string
    # return: if exists: tuple(station_id: str, data: dataframe)
    @staticmethod
    def get_data_file(station_id, year):
        base_url = "https://www.ndbc.noaa.gov/data/historical/stdmet/"

        #station_ID to lowercase:
        station_id = ''.join(char.lower() if char.isalpha() else char for char in station_id)

        #filename = station_id + "h" + year + ".txt.gz"
        filename = f"{station_id}h{year}.txt.gz"
        url = base_url + filename
        filepath = f"{os.path.dirname(__file__)}/data/NDBC_downloads/singleStation/{station_id}h{year}.csv"

        # Try to read from memory
        if os.path.isfile(filepath):
            df_data = pd.read_csv(filepath, low_memory=False)

            # in memory, the index is duplicated ... remove column "Unnamed: 0"
            df_data.drop(df_data.columns[0], axis=1, inplace=True)
            print("from disc")
            return station_id, df_data  # return as tuple
        else:
            # try to download
            try:
                df_data = pd.read_csv(url, delim_whitespace=True, low_memory=False)
                print("from web")
                # save to memory
                df_data.to_csv(filepath, index=True)

                return station_id, df_data  # return as tuple

            # Not in memory and can not be downloaded
            except BaseException as e:
                print(f'Failed to get {filename}: {e}')
                #print('Failed to get file: {}'.format(e))
                return None


    # Parameter: data is the return of get_data_file = tupe of station number and dataframe
    # What it does:
    # drop row with units
    # replace time-columns with one timestamp column as index
    # add station name to column header
    # return: tuple of filename and modified dataframe
    @staticmethod
    def df_modification(data):
        filename = data[0]
        df = data[1]
        df = df.drop(labels=0, axis=0)  # drop row with units

        # replace time columns with timestamp index

        # Some dataframes use slightly different headers
        if 'YYYY' in df.columns:
            df = df.rename(columns={'YYYY': '#YY'})     # e.g. 31201h2005
        if 'YY' in df.columns:
            df = df.rename(columns={'YY': '#YY'})       # e.g. 3201h1984
        if 'WD' in df.columns:
            df = df.rename(columns={'WD': 'WDIR'})      # e.g. 31201h2005
        if "mm" not in df.columns:                      # missing minutes e.g. 32301h1984
            df.loc[:, "mm"] = "00"
        if 'BAR' in df.columns:
            df = df.rename(columns={'BAR': 'PRES'})       # e.g. 41112h2007

        # In some dataframes (at least one--> 31201h2005) those columns are int64 instead of string!
        df['#YY'] = df['#YY'].astype(str)
        df['MM'] = df['MM'].astype(str)
        df['DD'] = df['DD'].astype(str)
        df['hh'] = df['hh'].astype(str)
        df['mm'] = df['mm'].astype(str)

        # check year component length and add "19" if necessary e.g. 32301h1984
        #TODO (non-urgant): this could also be 20 in some cases!
        df['#YY'] = ['19' + yy if len(yy) == 2 else yy for yy in df['#YY']]

        df['timestamp'] = df['#YY'] + "-" \
                          + df['MM'] + "-" \
                          + df['DD'] + " " \
                          + df['hh'] + ":" \
                          + df['mm']

        for i, row in df.iterrows():
            try:
                df.at[i, 'timestamp'] = pd.to_datetime(row['timestamp'], format='%Y-%m-%d %H:%M')
            except ValueError as e:
                #Needed to handle unit-rows within the dataset!
                print("VALUE ERROR: ", e, " - DROPPED ROW")
                df.drop(i, inplace=True)

        df.drop(columns=['#YY', 'MM', 'DD', 'hh', 'mm'], inplace=True)
        df.set_index('timestamp', inplace=True)

        df.columns += "_" + filename  # rename columns

        return filename, df


    # The function build_dataset(station_numbers, year) builds a dataset for one specific year considering the
    # given station numbers. Each instance uses a timestamp (index) as identifier.
    # Features are all features from all stations. (station id is stored in column name).
    # One instance should represent the state of an area (given by station numbers) at a certain timestamp.
    # This can be used to forecast future states of the same region.

    # UNUSED AND UNNEEDED FILE!

    # @staticmethod
    # def build_dataset(station_numbers, year):
    #     dataframes = list()
    #
    #     for station_number in station_numbers:
    #         data = NDBC_lib.get_data_file(station_number, year)  # load file
    #
    #         if data is not None:
    #             current_df = NDBC_lib.df_modification(data)[1]  # [1] ... df and not the whole tuple
    #             current_df = current_df.loc[~current_df.index.duplicated(keep='first')]
    #             dataframes.append(current_df)
    #
    #     merged_data = pd.concat(dataframes, axis=1, join="outer")  # outer join also includes NaN, inner join removes them
    #     return merged_data


    #Function to get the buoy data and return a well prepared dataset
    #What happens:
    #   Download data
    #   Merge all time-columns to timestamp
    #   Ensure that all 1h timestamps but no others exist
    #   delete columns which are not covered by ERA5
    #   if file does not exist, a empty DF is returned
    @staticmethod
    #def get_buoy_data(station_id, year):     Renamed to:
    def get_NDBC_singleStation(station_id, year):
        # station_ID to lowercase:
        station_id = ''.join(char.lower() if char.isalpha() else char for char in station_id)

        timestamp_filter_list = create_timestamp_list(year)
        df_NDBC = NDBC_lib.get_data_file(station_id, year)

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
            df_NDBC = NDBC_lib.df_modification(df_NDBC)[1]
            df_NDBC = NDBC_lib.replace_with_NaN(df_NDBC)

            # Handling duplicated index:
            num_of_duplicates = df_NDBC.index.duplicated().sum()
            if num_of_duplicates > 0: print(f"Found {num_of_duplicates} duplicates is {station_id}h{year} and removed them!")
            df_NDBC = df_NDBC.loc[~df_NDBC.index.duplicated(keep='first')]

            df_NDBC = df_NDBC.filter(timestamp_filter_list, axis=0)

            if f'GST_{station_id}' in df_NDBC.columns:
                df_NDBC.drop(f'GST_{station_id}', axis=1, inplace=True)
            if f'DPD_{station_id}' in df_NDBC.columns:
                df_NDBC.drop(f'DPD_{station_id}', axis=1, inplace=True)
            if f'VIS_{station_id}' in df_NDBC.columns:
                df_NDBC.drop(f'VIS_{station_id}', axis=1, inplace=True)
            if f'TIDE_{station_id}' in df_NDBC.columns:
                df_NDBC.drop(f'TIDE_{station_id}', axis=1, inplace=True)

        # some data rows are missed. Those are filled up with NaN:
        for timestamp in timestamp_filter_list:
            if not timestamp in df_NDBC.index:  # might must be timestamp instead of index
                df_NDBC.loc[timestamp] = [np.NAN] * 9

        df_NDBC.sort_index(inplace=True)
        df_NDBC = df_NDBC.apply(pd.to_numeric, errors='coerce') #convert to numeric (int or float, non numeric strings will be converted to NaN
        return df_NDBC


    # Same as get_NDBC_singleStation but for multiple files.
    # Files of several stations and years are properly merged into one DF
    # Furthermore, Download time is meassured and a NaN statistic is created!
    @staticmethod
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
                buoy_data = NDBC_lib.get_NDBC_singleStation(station, year)  # load file
                buoy_data_list.append(buoy_data)

                #Create NaN Statistic
                nan_rate = buoy_data.isna().sum().sum() / (buoy_data.shape[0] * buoy_data.shape[1])
                file_nan_count.loc[station,year] = nan_rate


            merged_buoy_data = pd.concat(buoy_data_list, axis=1, join="outer")  # outer join also includes NaN, inner join removes them
            data_list_annual.append(merged_buoy_data)

        print("Finished downloading - now merging it together!")

        dataset_NDBC = pd.concat(data_list_annual, axis=0)

        return dataset_NDBC, file_nan_count


    # Params:
    #   df ... Dataframe with numerical data
    #   value_list ... list of values which should be replaced by NaN
    # Return:
    #   dataframe (with additional NaN values)
    @staticmethod
    def replace_with_NaN(df, value_list=None):
        if value_list is None:
            value_list = ['9999', '999', '99', '9999.0', '999.0', '99.0', '9999.00', '999.00', '99.00', '9.9']

        new_df = df
        for value in value_list:
            new_df.replace(value, np.NaN, inplace=True)
            new_df.replace(float(value), np.NaN, inplace=True)  # Datatype security, since write and read csv changes type!
        return new_df

    @staticmethod
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
    # @staticmethod
    # def drop_NaN_rows_and_cols(df):
    #     clean_df = df.dropna(axis=1, how='all')  # drop cols if all values are NaN
    #     clean_df = clean_df.dropna(axis=0, how="all")  # drop rows if all values are NaN
    #     return clean_df

    @staticmethod
    def feature_selection_nan(df: pd.DataFrame , nan_threshold: float):
        '''
        Input: Dataframe
        Threshold: all columns with a nan_rate above the threshold are removed!
        '''
        # Calculate the percentage of NaN values in each column
        nan_percentages = df.isna().sum() / len(df)

        # Select the columns where the percentage of NaN values is below the threshold
        keep_columns = nan_percentages[nan_percentages <= nan_threshold].index

        # Filter the DataFrame to keep only the selected columns
        return df[keep_columns]

    @staticmethod
    def feature_selection_custom(df: pd.DataFrame, features: list = None):
        '''
        :param df: Dataframe with NDBC data. Column header must be of form: FEATURE_STATIONID. Example: WDIR_42036
        :param features: list with subset of ["WDIR", "WSPD", "WVHT", "APD", "MWD", "PRES", "ATMP", "WTMP", "DEWP"] or None for all features
        :return: Dataframe with only those features.
        '''
        if features == None: return df

        # Get a list of all column headers in the dataframe
        all_columns = list(df.columns)

        # Keep only the columns whose headers start with one of the valid starts
        valid_columns = [col for col in all_columns if col.startswith(tuple(features))]

        # Return the dataframe with only the valid columns
        return df[valid_columns]


#--------------------------------------------------------------------------------------
################## ERA5 STUFF #########################################################
#--------------------------------------------------------------------------------------
class ERA5_lib:

    # Download ERA5 Dataset of a single location with 1h timestamps.
    # It contains all the features that are also covered by NDBC!
    # The file will be downloaded and stored in ERA5/ERA5_downloads/singleStation/{station_id}_{year}.nc
    @staticmethod
    def download_ERA5_singlePoint(station_id, year, variables=None):
        # https://stackoverflow.com/questions/65186216/how-to-download-era5-data-for-specific-location-via-python
        # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

        # API parameters
        api_key_path = f"{os.path.dirname(__file__)}/ERA5/.cdsapirc"
        api_keyfile = open(api_key_path, "r")
        lines = api_keyfile.readlines()
        url = lines[0].rstrip().replace("url: ", "")
        key = lines[1].rstrip().replace("key: ", "")

        if variables is None:
            variables = [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                    '2m_temperature', 'mean_sea_level_pressure', 'mean_wave_direction',
                    'mean_wave_period', 'sea_surface_temperature', 'significant_height_of_total_swell',
                ]

        path = f"{os.path.dirname(__file__)}/data/ERA5_downloads/singleStation/{station_id}_{year}.nc"

        #Read Metadata
        metadata = pd.read_csv(f"{os.path.dirname(__file__)}/data/metadata/metadata_2023_03_14.csv")
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
    @staticmethod
    def get_ERA5_singlePoint(station_id, year):
        #station_ID to lowercase:
        station_id = ''.join(char.lower() if char.isalpha() else char for char in station_id)

        path = f"{os.path.dirname(__file__)}/data/ERA5_downloads/singleStation/{station_id}_{year}.nc"

        if not os.path.exists(path):
            print(f"donwloading {station_id}_{year}.nc ...")
            ERA5_lib.download_ERA5_singlePoint(station_id, year)    # only download if not found
            print(f"Completed download of {station_id}_{year}.nc!")

        ds_ERA5 = nc.Dataset(path)  #read from file

        # Extract data from .nc file
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
            WDIR.append(ERA5_lib.calc_WDIR(v, u))
            WSPD.append(ERA5_lib.calcWSPD(v, u))

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
        df_ERA5["timestamp"] = create_timestamp_list(year)
        df_ERA5.set_index('timestamp', inplace=True)
        df_ERA5[f"DEWP_{station_id}"] -= 273.15  # convert to degree Celsius
        df_ERA5[f"ATMP_{station_id}"] -= 273.15
        df_ERA5[f"PRES_{station_id}"] = df_ERA5[f"PRES_{station_id}"] / 100  # convert Pa to hPa
        df_ERA5[f"WTMP_{station_id}"] -= 273.15

        return df_ERA5


    # Creates equivalent of build_NDBC_dataset.
    # Considered Feeatures: ["WDIR", "WSPD", "WVHT", "APD", "MWD", "PRES", "ATMP", "WTMP", "DEWP"]
    # Timesteps: 1h
    @staticmethod
    def build_ERA5_dataset(stations: list, years: list):

        time_ref = time.time()

        data_list_annual = list()    # each element in this list is a df containing data of one certain year and all locations
        for year in years:
            print("Started with ", year, ". Previous year took:  ", time.time() - time_ref , "seconds")
            time_ref = time.time()

            point_data_list = list() # each element in this list is a df containing data of one certain year and one certain station.
            for station in stations:
                point_data = ERA5_lib.get_ERA5_singlePoint(station, year)
                point_data_list.append(point_data)

            merged_buoy_data = pd.concat(point_data_list, axis=1, join="outer")  # outer join also includes NaN, inner join removes them
            data_list_annual.append(merged_buoy_data)

        print("Finished downloading - now merging it together!")

        dataset_ERA5 = pd.concat(data_list_annual, axis=0)

        return dataset_ERA5


    # v ... x-axis (North is plus)
    # u ... y-axis (East is plus)
    @staticmethod
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

    @staticmethod
    def calcWSPD(v, u):
        return math.sqrt((u * u) + (v * v))  # Pythagoras


#--------------------------------------------------------------------------------------
################## COMPARE NDBC and ERA5 ##############################################
#--------------------------------------------------------------------------------------

# TODO (non-urgant) : For directions, take distance across north into account!    #???
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


def get_data_A(stations: list, years:list,
             nan_threshold: float,             #0..1 percentage of NaN values to drop feature
             features: list = None,
             era5: bool = False):
    '''
    This function prepares a Dataset ready for training a neural network
    Inkl. NaN imputation
    Each row is a instance identified by a timestamp. Timestep = 1h
    Each column is a feature. feture name: FEATURE_STATIONID
    Features can be selected by providing a subset of:
    ["WDIR", "WSPD", "WVHT", "APD", "MWD", "PRES", "ATMP", "WTMP", "DEWP"] or None for all features
    Features with a nan rate above the threshold are removed
    If ERA5 = True, all ERA5 equivalents are loaded. ERA5 features are marked with the prefix _ERA5
    Inkl. NaN imputation
    '''

    #GET_NDBC
    data_NDBC, NaN_Statistic = NDBC_lib.build_NDBC_dataset(stations, years)

    #Feature selection by NaN threshold
    data_NDBC = NDBC_lib.feature_selection_nan(data_NDBC, nan_threshold)

    # Feature Selection by parameter
    data_NDBC = NDBC_lib.feature_selection_custom(data_NDBC, features)

    #NaN imputation
    data_NDBC.fillna(method='ffill', inplace=True)
    data_NDBC.fillna(method='bfill', inplace=True)

    if not era5: return data_NDBC

    #GET_ERA5
    data_ERA5 = ERA5_lib.build_ERA5_dataset(stations, years)

    #MERGE DATA
    feature_cols = list(set(data_NDBC.columns).intersection(set(data_ERA5.columns)))
    merged_df = data_NDBC.copy()     # create a new dataframe with columns from ndbc
    for col in feature_cols:         # add columns from era5 where the names match
        new_col_name = col + '_ERA5'
        merged_df[new_col_name] = data_ERA5[col]

    return merged_df


def get_NDBC_and_ERA5(station_id, year, features=None, add_ERA5=True):
    station_id = station_id.lower()

    # Get NDBC
    data_NDBC = NDBC_lib.get_NDBC_singleStation(station_id, year)

    # Feature selection by parameter
    if features == None:
        features = measurements
    data_NDBC = data_NDBC[[f"{feature}_{station_id}" for feature in features]]

    if not add_ERA5:
        return data_NDBC

    # Get ERA5
    data_ERA5 = ERA5_lib.get_ERA5_singlePoint(station_id, year)

    # Merge NDBC and ERA5
    data = data_NDBC.copy()
    for col in data.columns:         # add columns from era5 where the names match
        new_col_name = col + '_ERA5'
        data[new_col_name] = data_ERA5[col]

    data.columns = data.columns.str.replace(f'_{station_id}', '')
    return data


def get_data_B(files, features=None, n_in=1, n_out=1, nan_threshold=1, add_ERA5=True):

    # build dataset
    data_list = []
    for file in files:
        station_id = file[0]
        year = file[1]
        data = get_NDBC_and_ERA5(station_id, year, features, add_ERA5)
        data_list.append(data)

    dataset = pd.concat(data_list, axis=0)

    # feature selection by NaN rate
    nan_percentages = dataset.isna().sum() / len(dataset)
    remove_columns = nan_percentages[nan_percentages > nan_threshold].index
    for col in dataset.columns:
        for string in remove_columns:
            if col.startswith(string):  # to also cover _ERA5 columns!
                dataset.drop(columns=col, inplace=True)

    # dataset to supervised
    dataset = DataProcessor.data_to_supervised(dataset, n_in, n_out, dropnan=False)

    # Drop rows with NaN values
    n_rows_before = dataset.shape[0]
    dataset.dropna(axis=0, inplace=True)
    n_removed_rows = n_rows_before - dataset.shape[0]
    print(f"Removed {n_removed_rows} rows with NaN values!")

    return dataset

#--------------------------------------------------------------------------------------
################## DATA PREPARATION (prepare data as NN input) ########################
#--------------------------------------------------------------------------------------
class DataProcessor:

    # Differenciates the data. It replaces each value with the diference to the previous value.
    # First row is removed
    @staticmethod
    def data_to_stationary(data: pd.DataFrame, n: int =1):
        data_stationary = pd.DataFrame()

        for col in data.columns:
            data_stationary[col] = data[col] - data[col].shift(n)  # y = value(i) - value(i-n)

        data_stationary = data_stationary.iloc[n:]  # remove first n entries since there is no delta value for them
        return data_stationary

    # Inverting data_to_stationary
    # Input:
    # data_stationary: Stationary time series
    # first column: Row of ture observations. Initial value before time series
        # Usually: data.iloc[-len(data_stationary)-1]
    # returns a numpy array with absolute instead of differenced values!
    # Not usable for one-shot-forecasting!!
    @staticmethod
    def stationary_to_data(data_stationary: np.array, first_row: pd.Series, n: int = 1):
        #Copy data
        data = data_stationary.copy()

        #Fill with initial values
        for col in range(data.shape[1]):
            data[0][col] = data[0][col] + first_row[col]



        #Loop columns and add stationary value (=dif) to previous value
        for row in range(n, data.shape[0]):   #loop rows
            for col in range(data.shape[1]):   #loop cols
                data[row][col] = data[row][col] + data[row-n][col]  #value(t) = value(t-1) + value_stationary(t)

        return data

    # Frame a time series as a supervised learning dataset.
    # Arguments:
    # data: Sequence of observations as a pd.Dataframe
    # n_in: Number of observations as input (X).
    # n_out: Number of observations as output (y).
    # dropnan: Boolean whether or not to drop rows with NaN values.
    # Returns:
    # Pandas DataFrame of series framed for supervised learning.
    # https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    @staticmethod
    def data_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = data.shape[1]
        col_names = data.columns

        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(data.shift(i))
            names += [('%s(t-%d)' % (col_names[j], i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)

        for i in range(0, n_out):
            cols.append(data.shift(-i))
            if i == 0:
                names += [('%s(t)' % (col_names[j])) for j in range(n_vars)]
            else:
                names += [('%s(t+%d)' % (col_names[j], i)) for j in range(n_vars)]
            # put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values

        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # Input:
    # Dataframe existing of columns that follow the following name convention:
    # input values are marked with suffix: t-n
    # output is marked with suffix: t or t+n
    # depending on the suffix, the data is seperated into input (X) and output (y)
    # Furthermore, the data is devided into train and test. The last n_test_hours rows represent the test set.
    # Return:
    #   train_X, test_X: numpy.array
    #   train_y, test_y: pd.Dataframe
    @staticmethod
    def train_test_split(data: pd.DataFrame, n_test_hours: int):
        # split into train and test sets
        train = data.head(-n_test_hours).values
        test = data.tail(n_test_hours).values

        # get indices of input and output columns
        input_cols = [i for i in range(data.values.shape[1]) if 't-' in data.columns[i]]
        output_cols = [i for i in range(data.values.shape[1]) if ('(t)' in data.columns[i]) or ('t+' in data.columns[i])]

        # split into input and outputs
        train_X, train_y = train[:, input_cols], train[:, output_cols]
        test_X, test_y = test[:, input_cols], test[:, output_cols]

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        return train_X, train_y, test_X, test_y

    # Scales the complete data on a scale between -1 and 1
    # Only considers training data to train scalar
    # Returns data in the same shape and the scaler which is needed to inverse the scaling!
    @staticmethod
    def scale_data(train_X, train_y, test_X, test_y):
        # Reshape data to 2D arrays
        train_X_2d = train_X.reshape(train_X.shape[0], -1)
        train_y_2d = train_y.reshape(train_y.shape[0], -1)
        test_X_2d = test_X.reshape(test_X.shape[0], -1)
        test_y_2d = test_y.reshape(test_y.shape[0], -1)

        # Define scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))

        # Fit scaler on training data
        scaler.fit(train_X_2d)

        # Scale training and testing data
        train_X_scaled = scaler.transform(train_X_2d).reshape(train_X.shape)
        train_y_scaled = scaler.transform(train_y_2d).reshape(train_y.shape)
        test_X_scaled = scaler.transform(test_X_2d).reshape(test_X.shape)
        test_y_scaled = scaler.transform(test_y_2d).reshape(test_y.shape)

        return train_X_scaled, train_y_scaled, test_X_scaled, test_y_scaled, scaler

    # Inverses function to scale_data for predictions
    @staticmethod
    def invert_scaling(predictions, scaler):
        # Reshape predictions to 2D arrays
        predictions_2d = predictions.reshape(predictions.shape[0], -1)

        # Invert scaling of predictions
        inverted_2d = scaler.inverse_transform(predictions_2d)

        # Reshape predictions to match original shape
        inverted = inverted_2d.reshape(predictions.shape)

        return inverted


#--------------------------------------------------------------------------------------
################## EXPERIMENT-DATA-CONTAINER AND EVALUATION  ##########################
#--------------------------------------------------------------------------------------
# This class is used to store data of experiment as .pickle file.
# Furthermore, it contains all evaluation functions
@dataclass
class Experiment():
    name: str
    description: str

    stations: list
    years: list
    nan_threshold: float
    features: list
    era5: bool
    stationary_shift: int
    n_test_hours: int

    #Preprocessing:
    #stationary: bool
    scaler: object

    #Model:
    model_name: str
    model_summary: str

    one_shot_forecast: Union[pd.DataFrame, None]
    recursive_forecast: Union[pd.DataFrame, None]

    one_shot_forecast_MSE: float  = -1.0
    one_shot_forecast_MAE: float  = -1.0
    recursive_forecast_MAE: float = -1.0
    recursive_forecast_MSE: float = -1.0

    def __post_init__(self):
        # One shot forecasting:
        if self.one_shot_forecast is not None:
            wtmp_true = [col for col in self.one_shot_forecast.columns if col.startswith("WTMP")][0]

            self.one_shot_forecast_MAE = mean_absolute_error(
                self.one_shot_forecast[wtmp_true],
                self.one_shot_forecast[f"{wtmp_true}_pred"]
            )

            self.one_shot_forecast_MSE = mean_squared_error(
                self.one_shot_forecast[wtmp_true],
                self.one_shot_forecast[f"{wtmp_true}_pred"]
            )
        # recurent forecasting:
        if self.recursive_forecast is not None:
            wtmp_true = [col for col in self.recursive_forecast.columns if col.startswith("WTMP")][0]

            self.recursive_forecast_MAE = mean_absolute_error(
                self.recursive_forecast[wtmp_true],
                self.recursive_forecast[f"{wtmp_true}_pred"]
            )

            self.recursive_forecast_MSE = mean_squared_error(
                self.recursive_forecast[wtmp_true],
                self.recursive_forecast[f"{wtmp_true}_pred"]
            )

    def get_raw_data(self):
        data = get_data_A(
            stations=self.stations,
            years=self.years,
            nan_threshold=self.nan_threshold,
            features=self.features,
            era5=self.era5
        )
        return data

    def print_settings(self):
        print(f"Experiment: {self.name}")
        print(f"{self.description}")
        print(f"---------------------------------------")
        print(f"Stations: {self.stations}")
        print(f"Years: {self.years}")
        print(f"NaN_Threshold: {self.nan_threshold}")
        print(f"Features: {self.features}")
        print(f"ERA5: {self.era5}, Stationary Shift: {self.stationary_shift}, Test-Hours:{self.n_test_hours}")
        print(f"\n---------------------------------------")
        # print(f"Preprocessing: Stationary: {self.stationary}")
        print(f"Normalized: {True if self.scaler is not None else False}")
        print(f"\n---------------------------------------")
        print(f"Internal Model name: {self.model_name}")
        print(self.model_summary)

    def print_metrics(self):
        print("One-Shot-Forecasting:")
        print(f"MAE: {self.one_shot_forecast_MAE} \tMSE: {self.one_shot_forecast_MSE}")
        print("\nRecurrent-Forecasting:")
        print(f"MAE: {self.recursive_forecast_MAE} \tMSE: {self.recursive_forecast_MSE}")

    # PLOT CHARTS
    # __ is the prefix for private functions!
    def __print_chart_helper(self, df):
        pred_cols = df.columns[df.columns.str.contains('_pred')]

        # Extract the feature names from the column names using regular expression
        features = [re.sub('_pred$', '', col) for col in pred_cols]

        # Plot the ground truth and prediction using the selected columns
        fig, ax = plt.subplots()
        df.plot(y=[f'{feature}_pred' for feature in features] + [f'{feature}' for feature in features], ax=ax)

        # Set the title and legend
        plt.title('Ground Truth vs Prediction')
        plt.legend([f'{feature}_pred' for feature in features] + [f'{feature}' for feature in features])
        plt.show()

    def print_one_shot_forecast(self):
        self.__print_chart_helper(self.one_shot_forecast)

    def print_recursive_forecast(self):
        self.__print_chart_helper(self.recursive_forecast)

    def print_one_shot_WTMP(self):
        wtmp_true = [col for col in self.one_shot_forecast.columns if col.startswith("WTMP")][0]

        # Plot the ground truth and prediction using the selected columns
        fig, ax = plt.subplots()
        self.one_shot_forecast.plot(y=[wtmp_true, f"{wtmp_true}_pred"], ax=ax)

        # Set the title and legend
        plt.title('Ground Truth vs Prediction')
        plt.legend([wtmp_true, f"{wtmp_true}_pred"])
        plt.show()

    def print_recursive_WTMP(self):
        wtmp_true = [col for col in self.one_shot_forecast.columns if col.startswith("WTMP")][0]

        # Plot the ground truth and prediction using the selected columns
        fig, ax = plt.subplots()
        self.recursive_forecast.plot(y=[wtmp_true, f"{wtmp_true}_pred"], ax=ax)

        # Set the title and legend
        plt.title('Ground Truth vs Prediction')
        plt.legend([wtmp_true, f"{wtmp_true}_pred"])
        plt.show()

#--------------------------------------------------------------------------------------
############################# MACHINE LEARNING MODELS  ################################
#--------------------------------------------------------------------------------------
class Models():
    # LSTM:
    def lstm_0(train_X, train_y):

        # design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(train_X.shape[2]))  # Predict all features!
        model.compile(loss='mean_squared_error', optimizer='adam')

        # fit network
        model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=0, shuffle=False)

        return model

    def lstm_1(train_X, train_y):
        # design network
        model = Sequential()
        model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(train_X.shape[2]))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # fit network
        history = model.fit(train_X, train_y, epochs=100, batch_size=64, verbose=1, shuffle=False, validation_split=0.1)

        return model

    def lstm_2(train_X, train_y):
        # design network
        model = Sequential()
        model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16))
        model.add(Dense(train_X.shape[2]))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # fit network
        history = model.fit(train_X, train_y, epochs=100, batch_size=64, verbose=1, shuffle=False, validation_split=0.1)

        return model

    # ------------------------------------------------------------------
    # PINN:
    def pinn_0(train_X, train_y, alpha):

        def custom_loss():
            def loss(y_true, y_pred):
                # Split y_true and y_pred into two features
                y_true_f1, y_true_f2 = tf.split(y_true, num_or_size_splits=2, axis=1)
                y_pred_f1, y_pred_f2 = tf.split(y_pred, num_or_size_splits=2, axis=1)

                # Calculate the mean squared error for each feature
                mse_f1 = K.mean(K.square(y_true_f1 - y_pred_f1), axis=-1)
                mse_f2 = K.mean(K.square(y_true_f2 - y_pred_f2), axis=-1)

                # Calculate the weighted loss
                weighted_loss = alpha * mse_f1 + (1 - alpha) * mse_f2

                return weighted_loss

            return loss

        # design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(train_X.shape[2]))  # Predict all features!

        model.compile(optimizer='adam',
                      loss=custom_loss(),
                      )

        # fit network
        model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=0, shuffle=False)

        return model

    def pinn_1(train_X, train_y, alpha):

        def custom_loss():
            def loss(y_true, y_pred):
                # Split y_true and y_pred into two features
                y_true_f1, y_true_f2 = tf.split(y_true, num_or_size_splits=2, axis=1)
                y_pred_f1, y_pred_f2 = tf.split(y_pred, num_or_size_splits=2, axis=1)

                # Calculate the mean squared error for each feature
                mse_f1 = K.mean(K.square(y_true_f1 - y_pred_f1), axis=-1)
                mse_f2 = K.mean(K.square(y_true_f2 - y_pred_f2), axis=-1)

                # Calculate the weighted loss
                weighted_loss = alpha * mse_f1 + (1 - alpha) * mse_f2

                return weighted_loss

            return loss

        # design network
        model = Sequential()
        model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(train_X.shape[2]))

        model.compile(optimizer='adam',
                      loss=custom_loss(),
                      )

        # fit network
        model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=0, shuffle=False)

        return model

    # ------------------------------------------------------------------

    model_dictionary = {
        # LSTM:
        "lstm_0": lstm_0,
        "lstm_1": lstm_1,
        "lstm_2": lstm_2,

        # PINN:
        "pinn_0": pinn_0,
        "pinn_1": pinn_1
        # ....
    }

    @staticmethod
    def get_model(model_name, train_X, train_y, alpha=None):

        if model_name.startswith("pinn"):
            model_function = Models.model_dictionary[model_name]
            return model_function(train_X, train_y, alpha)
        else:
            model_function = Models.model_dictionary[model_name]
            return model_function(train_X, train_y)


