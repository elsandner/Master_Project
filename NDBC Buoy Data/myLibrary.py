from datetime import timedelta

import numpy as np
import pandas as pd


# Data

stations_GOM = list(("41117",
"41112",
"42001",
"42002",
"42012",
"42013",
"42019",
"42020",
"42022",
"42023",
"42026",
"42036",
"42039",
"42040",
"42055",
"42084",
"42091",
"42095",
"42097",
"42098",
"42099",
"AMRL1",
"ANPT2",
"APCF1",
"APXF1",
"ARPF1",
"AWRT2",
"BABT2",
"BKBF1",
"BKTL1",
"BSCA1",
"BURL1",
"BYGL1",
"BZST2",
"CAPL1",
"CARL1",
"CDRF1",
"CNBF1",
"CRTA1",
"CWAF1",
"CWBF1",
"DILA1",
"DMSF1",
"EBEF1",
"EFLA1",
"EINL1",
"EMAT2",
"EPTT2",
"FHPF1",
"FMOA1",
"FMRF1",
"FPST2",
"FRDF1",
"GBIF1",
"GCTF1",
"GISL1",
"GKYF1",
"GNJT2",
"GRRT2",
"GTOT2",
"GTXF1",
"HIST2",
"HIVT2",
"HREF1",
"IRDT2",
"JXUF1",
"KATP",
"KBMG1",
"KBQX",
"KDLP",
"KGRY",
"KGUL",
"KHHV",
"KIKT",
"KTNF1",
"KVAF",
"KVOA",
"KYWF1",
"LCLL1",
"LMRF1",
"LQAT2",
"LTJF1",
"LUIT2",
"MAXT2",
"MBET2",
"MBPA1",
"MCGA1",
"MGPT2",
"MHBT2",
"MTBF1",
"MYPF1",
"NCHT2",
"NFDF1",
"NUET2",
"NWCL1",
"OBLA1",
"OPTF1",
"PACF1",
"PACT2",
"PCBF1",
"PCGT2",
"PCLF1",
"PCNT2",
"PILL1",
"PMAF1",
"PMNT2",
"PNLM6",
"PORT2",
"PSTL1",
"PTAT2",
"PTBM6",
"PTIT2",
"PTOA1",
"RCPT2",
"RKXF1",
"RLIT2",
"RLOT2",
"RSJT2",
"RTAT2",
"SAPF1",
"SAUF1",
"SDRT2",
"SGNT2",
"SGOF1",
"SHBL1",
"SHPF1",
"SKCF1",
"SMKF1",
"SREF1",
"SRST2",
"TAQT2",
"TESL1",
"TLVT2",
"TPAF1",
"TSHF1",
"TXPT2",
"TXVT2",
"UTVT2",
"VCAF1",
"VCAT2",
"VENF1",
"VTBT2",
"WIWF1",
"WKXA1",
"WPLF1",
"WYCM6",))

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
 'amrl1',
 'anpt2',
 'apcf1',
 'arpf1',
 'awrt2',
 'babt2',
 'bktl1',
 'bsca1',
 'burl1',
 'bygl1',
 'bzst2',
 'capl1',
 'carl1',
 'cdrf1',
 'cnbf1',
 'crta1',
 'cwaf1',
 'cwbf1',
 'dmsf1',
 'ebef1',
 'einl1',
 'emat2',
 'eptt2',
 'fhpf1',
 'fmoa1',
 'fmrf1',
 'fpst2',
 'frdf1',
 'gbif1',
 'gctf1',
 'gisl1',
 'gkyf1',
 'gnjt2',
 'grrt2',
 'gtot2',
 'hist2',
 'href1',
 'irdt2',
 'jxuf1',
 'ktnf1',
 'kywf1',
 'lcll1',
 'lmrf1',
 'ltjf1',
 'luit2',
 'mbet2',
 'mbpa1',
 'mcga1',
 'mgpt2',
 'mhbt2',
 'mtbf1',
 'mypf1',
 'ncht2',
 'nfdf1',
 'nuet2',
 'nwcl1',
 'obla1',
 'optf1',
 'pacf1',
 'pact2',
 'pcbf1',
 'pcgt2',
 'pclf1',
 'pcnt2',
 'pill1',
 'pmaf1',
 'pmnt2',
 'pnlm6',
 'port2',
 'pstl1',
 'ptat2',
 'ptbm6',
 'ptit2',
 'ptoa1',
 'rcpt2',
 'rlit2',
 'rlot2',
 'rsjt2',
 'rtat2',
 'sapf1',
 'sauf1',
 'sdrt2',
 'sgnt2',
 'sgof1',
 'shbl1',
 'shpf1',
 'skcf1',
 'smkf1',
 'sref1',
 'srst2',
 'taqt2',
 'tesl1',
 'tlvt2',
 'tpaf1',
 'tshf1',
 'txpt2',
 'txvt2',
 'utvt2',
 'vcaf1',
 'vcat2',
 'venf1',
 'wiwf1',
 'wplf1',
 'wycm6']

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
    #convert to lowercase
    station_numbers = [each_string.lower() for each_string in station_numbers]

    base_url ="https://www.ndbc.noaa.gov/data/historical/stdmet/"
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
# return: if exists: tuple(station_number: str, data: dataframe
def get_data_file(station_number, year):
    base_url ="https://www.ndbc.noaa.gov/data/historical/stdmet/"
    filename = station_number +"h"+ year + ".txt.gz"
    url = base_url + filename

    try:
        df_data = pd.read_csv(url, delim_whitespace=True, low_memory=False)
        return station_number, df_data    #return as tuple
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
    df = df.drop(labels=0, axis=0) #drop row with units

    #replace time columns with timestamp index
    df['timestamp'] =  df['#YY'] +"-" \
                       + df['MM'] +"-"  \
                       + df['DD'] +" " \
                       + df['hh'] +":"\
                       + df['mm']
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M')
    df.drop(columns=['#YY', 'MM', 'DD', 'hh', 'mm'], inplace=True)
    df.set_index('timestamp', inplace = True)

    df.columns += "_"+filename #rename columns

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
            current_df = df_modification(data)[1]   # [1] ... df and not the whole tuple
            current_df = current_df.loc[~current_df.index.duplicated(keep='first')]
            dataframes.append(current_df)

    merged_data = pd.concat(dataframes, axis=1, join="outer")   # outer join also includes NaN, inner join removes them
    return merged_data


# Parameter example:
# first_timestamp = datetime.strptime("2020-07-01 00:00", '%Y-%m-%d %H:%M')
# last_timestamp = datetime.strptime("2020-08-01 00:00", '%Y-%m-%d %H:%M')
def create_timestamp_list(first_timestamp, last_timestamp):

    timestamps = list()
    while first_timestamp != last_timestamp:
        timestamps.append(first_timestamp)
        first_timestamp = first_timestamp + timedelta(hours=1)

    return timestamps


def replace_with_NaN(df, value_list=None):
    if value_list is None:
        # TODO: doublecheck if this list is complete.
        value_list = ['9999', '999', '99', '9999.0', '999.0', '99.0', '9999.00', '999.00','99.00', '9.9']

    new_df = df
    for value in value_list:
        new_df.replace(value, np.NaN, inplace=True)
        new_df.replace(float(value), np.NaN, inplace=True)  #Datatype security, since write and read csv changes type!
    return new_df


def print_NaN_statistic(df, heading):
    print("\n",heading)
    num_values = df.shape[0] * df.shape[1]
    print("number of values: ", num_values)
    num_NaN = df.isna().sum().sum()
    num_NaN_percentage = round(100* num_NaN / num_values, 2)
    print("number of NaN values: ", num_NaN, "(",num_NaN_percentage,"%)\n")

    # Print percentage of NaN values per pprint_column
    num_of_rows =  len(df.index)
    count_NaN = list()
    percentage_NaN = list()
    features = list()
    for column in df:
        current_num_of_NaN = df[column].isna().sum()
        current_percentage_of_NaN = round(100 * current_num_of_NaN / num_of_rows, 2)

        features.append(column)
        count_NaN.append(current_num_of_NaN)
        percentage_NaN.append(current_percentage_of_NaN)

    NaN_by_feature = pd.DataFrame({'Feature':features,'#NaN':count_NaN, '%NaN':percentage_NaN})
    #display(NaN_by_feature)

    return num_values, num_NaN_percentage, NaN_by_feature


#Drop all columns and rows if they only exist of NaN values!
def drop_NaN_rows_and_cols(df):
    clean_df = df.dropna(axis=1,how='all') # drop cols if all values are NaN
    clean_df = clean_df.dropna(axis=0, how="all")  # drop rows if all values are NaN
    return clean_df