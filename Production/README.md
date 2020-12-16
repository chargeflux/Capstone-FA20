# Installation

The following packages will be required:
1. pandas
2. surprise 
3. cython 
4. scipy 
5. scikit-learn 
6. numpy

# Steps

1. Create a `previous_runs` and `unknown_run` folder
2. Add an initial set of runs to the `previous_runs` folder for the model to learn from
   1. This set of initial runs must all be of the same size and match any future unknown runs. It should cover all possible hardware configurations per program type. However, at a minimum one or two programs should cover all those configurations if getting such data is not possible.
3. Add one run (3 files - hardware, software and general statistics) to `unknown_run`
4. `python main.py` and it will ask for the directory paths for `previous_runs` and `unknown_run`. Do not escape spaces in the directory path and enter as-is.
5. You will receive the number of parameters and the parameters itself.
6. The unknown program files will be moved to the `previous_runs` to be used for future inference.

## Data Formatting

There are 3 important files per simulation run: x86, mem and simulation output.

The general format for the first 2 files is: `program_name.x86.parameter_name-parameter_value.parameter_name-parameter_value`. The second term is x86 or mem. It is very important the format is followed for the parameters: parameter name and parameter value separated by a "-".

The general format for simulation output is `program_name-experiments.txt`. The only value to change is program_name.

Additonally, avoid using commas in the file names.

### Example

blackscholes-experiments.txt

blackscholes.mem.w-4.l1-32k.l2-1m.l3-16m.out

blackscholes.x86.w-4.l1-32k.l2-1m.l3-16m.out

**Parameters**:
- w - 4
- l1 - 32k
- l2 - 1m
- l3 - 16m

## Important information

There can only be null values for committed instructions per cycle and instructions.

