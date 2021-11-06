#include <src/algorithms/semihdp_sampler.h>
#include <src/collectors/memory_collector.h>
#include <src/includes.h>

#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <vector>

#include "utils.hpp"
#include "src/utils/io_utils.h"

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

std::vector<MatrixXd> read_data(std::string filename) {
   Eigen::MatrixXd data = load_csv<Eigen::MatrixXd>(filename);
   int ngroups = static_cast<int>(data.col(0).maxCoeff()) + 1;
   std::vector<std::vector<double>> tmp(ngroups);
   for (int i=0; i < data.rows(); i++) {
       int group = static_cast<int>(data(i, 0));
       double x = data(i, 1);
       tmp[group].push_back(x);
    }
    std::vector<MatrixXd> out(ngroups);
    for (int i=0; i < tmp.size(); i++) {
        Eigen::MatrixXd groupdata = Eigen::Map<Eigen::MatrixXd>(
            tmp[i].data(), tmp[i].size(), 1);
        out[i] = groupdata;
    }

   return out;
}


int main(int argc, char *argv[]) {
    std::string data_file = argv[1];
    std::string params_file = argv[2];
    std::string output_chains_file = argv[3];
    std::string output_latent_vars_file = argv[4];
    std::string dens_grid_file = argv[5];
    std::string output_dens_path = argv[6];

    int niter = 10000;
    int nburn = 10000;
    int thin = 5;
    std::string update_c = "full";

    std::vector<MatrixXd> data = read_data(data_file);
    Eigen::MatrixXd dens_grid = load_csv<Eigen::MatrixXd>(dens_grid_file);
    std::cout << "dens_grid: " << dens_grid.transpose() << std::endl;

    MemoryCollector chains;    
    chains = run_semihdp(data, output_chains_file, params_file,
                         niter, nburn, thin, update_c);

    std::cout << "chains_size: " << chains.get_size() << std::endl;

    std::vector<MatrixXd> densities = eval_uni_dens(chains, dens_grid, data.size());
    for (int i = 0; i < densities.size(); i++) {
        std::string dens_file = output_dens_path + "/group_" +
                                std::to_string(i) + ".csv";
        bayesmix::write_matrix_to_file(densities[i], dens_file);
    }

    Eigen::MatrixXd latent_vars = get_latent_vars(chains, data.size());
    bayesmix::write_matrix_to_file(latent_vars, output_latent_vars_file);
    return -1;
}