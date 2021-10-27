#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <src/algorithms/neal2_algorithm.hpp>
#include <src/algorithms/semihdp_sampler.hpp>
#include <src/collectors/file_collector.hpp>
#include <src/collectors/memory_collector.hpp>
#include <src/includes.hpp>
#include <src/utils/rng.hpp>
#include <stan/math/prim.hpp>
#include <vector>

void run_semihdp(const std::vector<MatrixXd> data, std::string chainfile);

#endif
