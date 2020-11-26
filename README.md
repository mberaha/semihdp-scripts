# semihdp-scripts

C++ scripts for reproducing the numerical results of the paper 
"The semi-hierarchical Dirichlet Process and its application to clustering homogeneous distributions"


### Installation

1) Install the `bayesmix` library: clone the Github repo https://github.com/bayesmix-dev/bayesmix in your favorite directory
```shell
mkdir my_fav_directory
cd my_fav_directory
git clone https://github.com/bayesmix-dev/bayesmix.git
```
Install the dependencies
```shell
cd bayesmix
./bash/install_libs.sh
./bash/compile_protos.sh
```
Compile it
Install the dependencies
```shell
mkdir build
cmake ..
make bayesmixlib
```
Now tell the environment there the bayesmix folder is located add the following line
```
export BAYESMIX_HOME=<my_fav_directory>/bayesmix
```
to your `~/.bashrc` file and call
```shell
source ~/.bashrc
```

2) Clone this repository 
