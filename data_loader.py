# Changed 'mem.w4.l1-32k.l2-1m-l3-32m.out' to 'swaptions.mem.w4.l1-32k.l2-1m-l3-32m.out' because the x86 version existed for this configuration and 
# swaptions was the only program missing a mem output file for this configuration
# changed experiments.txt to blackscholes-experiments.txt since the file name was missing the program name but held data for blackscholes.
import configparser
from pathlib import Path
import pandas as pd

class BenchmarkProgram:
    def __init__(self, program_name, exp_config, output_data, mem_data, x86_data):
        self.program_name = program_name
        self.config = exp_config
        
        self.threads = 8 # Fixed
        self.width = None
        self.l1_cache = None
        self.l2_cache = None
        self.l3_cache = None

        self.knobs = None
        self.parse_knobs()

        self.committed_instructions_per_cycle = None
        self.branch_prediction_accuracy = None
        self.instructions = None
        try:
            self.parse_output_data(output_data)
        except:
            raise FileNotFoundError(f"Could not find output file: {self.program_name}-experiments.txt")

        self.accesses_mod_shared_mm = None
        self.accesses_mod_shared_l3 = None
        self.hits_mod_shared_l3 = None
        self.misses_mod_shared_l3 = None
        self.reads_mod_shared_l3 = None
        self.writes_mod_shared_l3 = None
        try:
            self.parse_mem_data(mem_data)
        except:
            raise FileNotFoundError(f"Could not find mem data file: {self.program_name}.mem.{self.config}.out")
        

        self.hw_threads = None
        self.cycles = None
        self.dispatch_ipc = None
        self.issue_ipc = None
        self.commit_total = None # proxy for instructions
        self.commit_ipc = None
        self.commit_pred_acc = None
        self.issue_commit_totals_diff = None
        self.dispatch_issue_totals_diff = None
        try:
            self.parse_x86_data(x86_data)
        except:
            raise FileNotFoundError(f"Could not find x86 data file: {self.program_name}.x86.{self.config}.out")

    def parse_knobs(self):
        width_raw, l1_raw, l2_l3_raw = self.config.split(".")
        self.width = width_raw[1:]
        self.l1_cache = l1_raw.split("l1-")[-1] 
        self.l2_cache = l2_l3_raw.split("-")[1] # HACK
        self.l3_cache = l2_l3_raw.split("-")[-1] # HACK
        self.knobs = ','.join([self.width, self.l1_cache, self.l2_cache, self.l3_cache])

    def parse_x86_data(self, data):
        core_count = int(data[" Config.General "]["Cores"])
        thread_count = int(data[" Config.General "]["Threads"])
        self.hw_threads = int(core_count) * int(thread_count)
        assert data[" Config.Pipeline "]["DecodeWidth"] == data[" Config.Pipeline "]["IssueWidth"] == data[" Config.Pipeline "]["CommitWidth"]
        assert self.width == data[" Config.Pipeline "]["DecodeWidth"]

        self.cycles = int(data[" Global "]["Cycles"])
        self.dispatch_ipc = float(data[" Global "]["Dispatch.IPC"])
        self.issue_ipc = float(data[" Global "]["Issue.IPC"])
        self.commit_total = int(data[" Global "]["Commit.Total"])
        self.commit_ipc = float(data[" Global "]["Commit.IPC"])
        self.commit_pred_acc = float(data[" Global "]["Commit.PredAcc"])
        if self.branch_prediction_accuracy:
            assert self.commit_pred_acc == self.branch_prediction_accuracy
        self.issue_commit_totals_diff = int(data[" Global "]["Issue.Total"]) - int(data[" Global "]["Commit.Total"])
        self.dispatch_issue_totals_diff = int(data[" Global "]["Dispatch.Total"]) - int(data[" Global "]["Issue.Total"])

    def parse_mem_data(self, data):
        self.accesses_mod_shared_mm = int(data[" mod-shared-mm "]["accesses"])
        self.accesses_mod_shared_l3 = int(data[" mod-shared-l3 "]["accesses"])
        
        self.hits_mod_shared_l3 = int(data[" mod-shared-l3 "]["Hits"])
        self.misses_mod_shared_l3 = int(data[" mod-shared-l3 "]["Misses"])
        self.reads_mod_shared_l3 = int(data[" mod-shared-l3 "]["Reads"])
        self.writes_mod_shared_l3 = int(data[" mod-shared-l3 "]["Writes"])

    def parse_output_data(self, data):
        try:
            raw_data = data.split(self.config)[1]
            committed_instructions_per_cycle_raw = raw_data.split("CommittedInstructionsPerCycle = ")[1]
        except IndexError:
            print("Could not parse output data for", self.program_name,"-", self.config)
            return
        committed_instructions_per_cycle_raw_end_idx = committed_instructions_per_cycle_raw.find("\n")
        self.committed_instructions_per_cycle = float(committed_instructions_per_cycle_raw[:committed_instructions_per_cycle_raw_end_idx])

        branch_prediction_accuracy_raw = raw_data.split("BranchPredictionAccuracy = ")[1]
        branch_prediction_accuracy_raw_end_idx = branch_prediction_accuracy_raw.find("\n")
        self.branch_prediction_accuracy = float(branch_prediction_accuracy_raw[:branch_prediction_accuracy_raw_end_idx])

        instructions_raw = raw_data.split("Instructions = ")[1]
        instructions_raw_end_idx = instructions_raw.find("\n")
        self.instructions = int(instructions_raw[:instructions_raw_end_idx])
    
    def get_data(self):
        current_data = vars(self).copy()
        current_data.pop("config")
        current_data.pop("knobs")
        return current_data

def process_data(path):
    configs = {}

    for path_to_file in path.glob('*.out'):
        file_name = path_to_file.name
        program_name = file_name.split('.')[0]
        config = '.'.join(file_name.split(".")[2:-1])
        configs.setdefault(program_name, {}).setdefault(config, []).append(path_to_file)

    program_config_results = []
    benchmark_programs = []

    for program_name, config_data in configs.items():
        for config, files in config_data.items():
            parser_mem = None
            parser_x86 = None
            current_program_name = files[0].name.split(".")[0]
            for file_path in files:
                if 'mem' in file_path.name:
                    parser_mem = configparser.ConfigParser(allow_no_value=True, strict=False)

                    parser_mem.read(file_path)
                if 'x86' in file_path.name:
                    parser_x86 = configparser.ConfigParser(allow_no_value=True, strict=False)

                    parser_x86.read(file_path)
                file_name_output = path / (current_program_name + "-" + "experiments.txt")

            with open(file_name_output, "r") as f:
                program_output = f.read()

            print(f"Processing {current_program_name} - {config}")
            benchmark_program = BenchmarkProgram(current_program_name, config, program_output, parser_mem, parser_x86)
            benchmark_data = benchmark_program.get_data()
            program_config_results.append(benchmark_data)
            benchmark_programs.append(benchmark_program)
    return program_config_results, benchmark_programs

def get_dataframe_all_data():
    program_config_results, _ = process_data(Path("experiments"))
    df = pd.DataFrame(program_config_results)
    return df

def get_config_cycles():
    _, benchmark_programs = process_data(Path("experiments"))
    config_cycles = [(i.program_name, i.knobs, i.cycles) for i in benchmark_programs]
    df = pd.DataFrame(config_cycles, columns=["program_name", "config", "cycles"])
    return df

print("Getting all experiment data")
print("="*50)
df = get_dataframe_all_data()
df.to_csv("experiment_data.csv", index=False)
print("="*50)
print("Getting cycle count per program and config")
print("="*50)
df2 = get_config_cycles()
df2.to_csv("config_cycles.csv", index=False)


