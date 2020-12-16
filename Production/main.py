import data_loader
from pathlib import Path
import pandas as pd
import outlier_preprocessing
import outlier_detection
import model

def preprocess_outlier_detection(final_df, parameter_names):
    final_df.drop(parameter_names, axis=1, inplace=True)
    imputed = outlier_preprocessing.impute_missing_values(final_df)
    scaled_imputed = outlier_preprocessing.scale_features(imputed)
    imputed.update(scaled_imputed)
    imputed.drop('cycles', axis=1, inplace=True)
    return imputed

def verify_exclusion(config_cycle_data, excluded_program):
    if excluded_program:
        excluded = config_cycle_data[config_cycle_data.program_name != excluded_program]
        return len(config_cycle_data.config.unique()) == len(excluded.config.unique())
    return None

if __name__ == "__main__":
    previous_runs_path = input("Directory path to previous runs: ")
    if not Path(previous_runs_path).exists() or not Path(previous_runs_path).is_dir():
        raise ValueError("Directory for previous runs not found")
    
    unknown_program_path = input("Directory path to unknown program run: ")
    if not Path(unknown_program_path).exists() or not Path(unknown_program_path).is_dir():
        raise ValueError("Directory for unknown program run not found")

    print("Creating DF for all previous run data")
    print("="*50)
    all_previous_run_data = data_loader.get_dataframe_all_data(previous_runs_path)
    print("="*50)
    print("Creating DF for cycle count only per program and config for previous runs")
    print("="*50)
    previous_run_config_cycle_data, previous_runs_parameter_names = data_loader.get_config_cycles(previous_runs_path)
    
    print("="*50)
    print("Creating DF for unknown program data")
    print("="*50)
    unknown_program_run_data = data_loader.get_dataframe_all_data(unknown_program_path)
    print("="*50)
    print("Creating DF for cycle count only for unknown run")
    print("="*50)
    unknown_program_run_config_cycle_data, unknown_program_parameter_names = data_loader.get_config_cycles(unknown_program_path)
    print("="*50)
    if len(unknown_program_run_config_cycle_data) > 1 or len(unknown_program_run_data) > 1:
        raise ValueError("Please have only one unknown run")
    if previous_runs_parameter_names != unknown_program_parameter_names:
        raise ValueError(f"Unknown program parameters do not match parameters of previous runs\n\
previous_runs: {previous_runs_parameter_names}\nunknown_run: {unknown_program_parameter_names}")

    all_data_final = pd.concat([all_previous_run_data, unknown_program_run_data]).reset_index(drop=True)

    outlier_detection_data = preprocess_outlier_detection(all_data_final, previous_runs_parameter_names)

    remove_program = outlier_detection.detect_outliers(outlier_detection_data)
    print("="*50)
    if verify_exclusion(previous_run_config_cycle_data, remove_program):
        previous_run_config_cycle_data = previous_run_config_cycle_data[previous_run_config_cycle_data.program_name != remove_program]

    config_cycle_data_final = pd.concat([previous_run_config_cycle_data, unknown_program_run_config_cycle_data]).reset_index(drop=True)
    best_config = model.get_best_config(config_cycle_data_final, unknown_program_run_config_cycle_data.program_name.values[0])

    unknown_config = unknown_program_run_config_cycle_data.config.values[0].split(",")

    num_parameters_to_consider = 0
    names_parameters_to_consider = []
    for idx, (best_parameter_value, unknown_parameter_value) in enumerate(zip(best_config, unknown_config)):
        if best_parameter_value != unknown_parameter_value:
            num_parameters_to_consider += 1
            names_parameters_to_consider.append(unknown_program_parameter_names[idx])
    
    print("="*50)
    print(f"{num_parameters_to_consider} parameters")
    print(f"{', '.join(names_parameters_to_consider)}")

    print("Moving unknown program files to previous runs")

    unk_path = Path(unknown_program_path)
    prev_run_path = Path(previous_runs_path)

    out_files = list(unk_path.glob('*.out'))
    for file in out_files:
        if not (prev_run_path / file.name).exists():
            file.replace(prev_run_path / file.name)
        else:
            raise FileExistsError(f"Unknown program data file already exists in previous runs location: {file}. \
Results may be compromised")
    
    txt_file = list(unk_path.glob('*.txt'))
    assert len(txt_file) == 1

    with open(txt_file[0], 'r') as f:
        unknown_program_output = f.read()
    if (prev_run_path / txt_file[0].name).exists():
        with open(prev_run_path / txt_file[0].name, 'a') as f:
            f.write("\n" + unknown_program_output)
    else:
        with open(prev_run_path / txt_file[0].name, 'a') as f:
            f.write(unknown_program_output)
    txt_file[0].unlink()


     