# semihdp-scripts

C++ scripts for reproducing the numerical results of the paper 
"The semi-hierarchical Dirichlet Process and its application to clustering homogeneous distributions" [(Bayesian Analysis, 2021)](https://doi.org/10.1214/21-BA1278)


### Installation

1. Clone this repo with its submodules
```
git clone --recurse-submodules  https://github.com/mberaha/semihdp-scripts.git 
```

2. Build the executable
```
mkdir build
cd build
cmake ..
make run_from_file
```

### Running on your dataset

From the root of the directory you can call
```
./build/run_from_file \
  DATASET_FILE.csv \
  semihdp_params.asciipb \
  CHAINS_FILE.recordio \
  LATENT_VARS_FILE.csv \
  DENSITY_GRID.csv \
  PATH_TO_OUTPUT_DENSITIES
```
where

1. `DATASET_FILE.csv` is the path to a csv file with two columns: the group id and the observation (no header)
2. `semihdp_params.asciipb` contains all the prior hyperparameters, it is in the root folder of the repo
3. `CHAINS_FILE.recordio` is where the MCMC chains will be stored (as a sequence of serialized protocol buffers)
4. `LATENT_VARS_FILE.csv`is where the latent variables associated to each observation will be saved, useful to identify the clusters. This will be a csv file with four columns: [iteration_number, group_id, mean, var]
5. `DENSITY_GRID.csv`is a csv file with the grid over which to evaluate the (log) density. This is common for all the groups
6. `PATH_TO_OUTPUT_DENSITIES` is a path where one csv file for each group will be created. Each file will store the mixture density evaluated on the grid at each iteration of the MCMC chain (by row).

For instance, to run the semihdp on the dataset in `example/data.csv` and evaluating the density on the grid `example/xgrid.csv`:

```
./build/run_from_file \
  example/data.csv \
  semihdp_params.asciipb \
  example/chains.recordio \
  example/latent_vars.csv \
  example/xgrid.csv \
  example/dens
```

