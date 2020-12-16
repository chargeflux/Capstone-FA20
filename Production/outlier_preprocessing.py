import pandas as pd
import numpy as np
from sklearn import preprocessing

def eda_numerical(input_data):
    features = list(input_data)
    dict_output_data = {"variable":[],
                        "number_nan":[],
                       }
    for feature in features:
        dict_output_data["variable"].append(feature)
        dict_output_data["number_nan"].append(input_data[feature].isna().sum())
    output_data = pd.DataFrame(dict_output_data)
    return output_data

def eda_string(input_data):
    features = list(input_data)
    dict_output_data = {"variable":[],
                        "number_nan":[],
                       }
    for feature in features:
        dict_output_data["variable"].append(feature)
        dict_output_data["number_nan"].append(input_data[feature].isna().sum())
        dict_output_data["number_distinct"].append(input_data[feature].nunique())
    output_data = pd.DataFrame(dict_output_data)
    return output_data

def impute_missing_values(input_data):
    input_data_numerical = input_data.select_dtypes(include=[np.number])
    miss_filter = eda_numerical(input_data_numerical)['number_nan'] > 0
    vars_with_missing = list(eda_numerical(input_data_numerical)[miss_filter].variable)
    if sorted(vars_with_missing) != ['committed_instructions_per_cycle', 'instructions']:
        raise ValueError(f"Can not perform imputation on columns other than committed_instructions_per_cycle and instructions\n\
missing variables: {vars_with_missing}")

    CIPC_filter = input_data.groupby('program_name')['committed_instructions_per_cycle'].apply(lambda x: x.notnull().sum()/len(x)*100).reset_index(name='percent')
    I_filter = input_data.groupby('program_name')['instructions'].apply(lambda x: x.notnull().sum()/len(x)*100).reset_index(name='percent')

    # Consider cases where all values for columns for a program is null
    CIPC_filter_all_null = CIPC_filter["program_name"][CIPC_filter['percent']==0].reset_index(name='program_name')
    I_filter_all_null = I_filter["program_name"][I_filter['percent']==0].reset_index(name='program_name')

    input_data.loc[input_data['program_name'].isin(CIPC_filter_all_null['program_name']) & input_data['committed_instructions_per_cycle'].isna(),\
                   'committed_instructions_per_cycle'] = (input_data['commit_total'] / input_data['cycles'])
    
    input_data.loc[input_data['program_name'].isin(I_filter_all_null['program_name']) & input_data['instructions'].isna(), 'instructions'] = input_data['commit_total'] 

    # Consider cases where some values for columns for a program is null
    CIPC_filter_some_null = CIPC_filter["program_name"][(CIPC_filter['percent']!=0) & (CIPC_filter['percent']!= 100)].reset_index(name='program_name')
    I_filter_some_null = I_filter["program_name"][(I_filter['percent']!=0) & (I_filter['percent']!= 100)].reset_index(name='program_name')

    CIPC_median_df = input_data[input_data['program_name'].isin(CIPC_filter_some_null['program_name'])].groupby('program_name')\
        ['committed_instructions_per_cycle'].apply(lambda x: x.median()).reset_index(name='median')

    for program in CIPC_filter_some_null['program_name']:
        input_data.loc[(input_data['program_name'] == program) & (input_data['committed_instructions_per_cycle'].isna()),\
                    'committed_instructions_per_cycle'] = CIPC_median_df[CIPC_median_df['program_name'] == program]['median'].values[0]

    I_median_df = input_data[input_data['program_name'].isin(I_filter_some_null['program_name'])].groupby('program_name')['instructions'].\
        apply(lambda x: x.median()).reset_index(name='median')

    for program in I_filter_some_null['program_name']:
        input_data.loc[(input_data['program_name'] == program) & (input_data['instructions'].isna()), 'instructions'] =\
                    I_median_df[I_median_df['program_name'] == program]['median'].values[0]

    # necessary only for imputation of instructions
    input_data.drop(columns=['commit_total'], inplace=True)

    # Assuming threads and hw_threads are equal per run and across runs so it is not necessary
    input_data.drop(columns=['hw_threads', 'threads'], inplace=True)

    return input_data

def scale_features(input_data):
    input_numerical = input_data.select_dtypes(include=[np.number])
    input_cols = [col for col in input_numerical.columns if col not in ['cycles']]
    input_data_features_only = input_numerical[input_cols]
    data_scaled = preprocessing.scale(input_data_features_only)
    data_scaled_df = pd.DataFrame(data_scaled, columns = input_cols)
    return data_scaled_df
