#ifndef UTILS_HPP
#define UTILS_HPP

#include <src/algorithms/neal2_algorithm.h>
#include <src/algorithms/semihdp_sampler.h>
#include <src/collectors/file_collector.h>
#include <src/collectors/memory_collector.h>
#include <src/includes.h>
#include <src/utils/rng.h>

#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <vector>

using namespace Eigen;

void run_semihdp(const std::vector<Eigen::MatrixXd> data,
                 std::string chainfile, std::string update_c = "full");

#endif
