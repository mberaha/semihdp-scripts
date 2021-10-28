# semihdp-scripts

C++ scripts for reproducing the numerical results of the paper 
"The semi-hierarchical Dirichlet Process and its application to clustering homogeneous distributions" [(Bayesian Analysis, 2021)](https://doi.org/10.1214/21-BA1278)


### Installation

1) Install the `bayesmix` library: clone the Github repo https://github.com/bayesmix-dev/bayesmix in your favorite directory
```shell
mkdir my_fav_directory
cd my_fav_directory
git clone --recurse-submodules https://github.com/bayesmix-dev/bayesmix.git
```
Install the dependencies
```shell
cd bayesmix
python3 build_tbb.py
```

2) create a subdirectory `scripts` and clone this repo there
```shell
mkdir scripts
cd scripts
git clone git@github.com:mberaha/semihdp-scripts.git .
```

3) At the end of the "root" CMakeLists.txt file of bayesmix (located in bayesmix/CMakeLists.txt) add
the following line
```
add_subdirectory(scripts)
```

4) Now it's time to build! From the main `bayesmix` directory:
```shell
mkdir build
cd buid
cmake ..
make simu1
```
whill build the executable to run the first simulation. Other executables are `simu2` and `simu_many`.

5) Run the executable:
```shell
./simu1
```
