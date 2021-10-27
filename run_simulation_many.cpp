// This scripts runs the simulations with four populations (Section 6.1)

#include <src/algorithms/neal2_algorithm.h>
#include <src/algorithms/semihdp_sampler.h>
#include <src/collectors/file_collector.h>
#include <src/collectors/memory_collector.h>
#include <src/includes.h>
#include <src/utils/rng.h>

#include <Eigen/Dense>
#include <chrono>
#include <stan/math/prim.hpp>
#include <vector>

using Eigen::MatrixXd;

MatrixXd generate_mixture(double m1, double s1, double m2, double s2, double w,
                          int n, std::mt19937_64& rng) {
  MatrixXd out(n, 1);
  for (int i = 0; i < n; i++) {
    if (stan::math::uniform_rng(0, 1, rng) < w) {
      out(i, 0) = stan::math::normal_rng(m1, s1, rng);
    } else {
      out(i, 0) = stan::math::normal_rng(m2, s2, rng);
    }
  }
  return out;
}

void run_semihdp2(const std::vector<MatrixXd> data, std::string chainfile,
                  std::string update_c = "full") {
  // compute overall mean
  double mu0 = std::accumulate(
      data.begin(), data.end(), 0,
      [&](int curr, const MatrixXd dat) { return curr + dat.sum(); });
  mu0 /= std::accumulate(
      data.begin(), data.end(), 0.0,
      [&](int curr, const MatrixXd dat) { return curr + dat.rows(); });
  auto hier = std::make_shared<NNIGHierarchy>();
  bayesmix::NNIGPrior hier_prior;
  hier_prior.mutable_fixed_values()->set_mean(mu0);
  hier_prior.mutable_fixed_values()->set_var_scaling(0.1);
  hier_prior.mutable_fixed_values()->set_shape(2.0);
  hier_prior.mutable_fixed_values()->set_scale(2.0);
  hier->get_mutable_prior()->CopyFrom(hier_prior);
  hier->initialize();

  // Collect pseudo priors
  std::vector<MemoryCollector> pseudoprior_collectors;
  pseudoprior_collectors.resize(data.size());
  bayesmix::DPPrior mix_prior;
  double totalmass = 1.0;
  mix_prior.mutable_fixed_value()->set_totalmass(totalmass);
#pragma omp parallel for
  for (int i = 0; i < data.size(); i++) {
    auto mixing = std::make_shared<DirichletMixing>();
    mixing->get_mutable_prior()->CopyFrom(mix_prior);
    mixing->set_num_components(5);
    auto hier = std::make_shared<NNIGHierarchy>();
    bayesmix::NNIGPrior hier_prior;
    hier_prior.mutable_fixed_values()->set_mean(data[i].mean());
    hier_prior.mutable_fixed_values()->set_var_scaling(0.1);
    hier_prior.mutable_fixed_values()->set_shape(2.0);
    hier_prior.mutable_fixed_values()->set_scale(2.0);
    hier->get_mutable_prior()->CopyFrom(hier_prior);

    Neal2Algorithm sampler;
    sampler.set_maxiter(2000);
    sampler.set_burnin(1000);
    sampler.set_mixing(mixing);
    sampler.set_data(data[i]);
    sampler.set_hierarchy(hier);
    sampler.run(&pseudoprior_collectors[i], false);
  }

  auto start = std::chrono::high_resolution_clock::now();
  int nburn = 5000;
  int niter = 5000;
  MemoryCollector collector;

  bayesmix::SemiHdpParams params;
  bayesmix::read_proto_from_file(
      "/home/mario/dev/bayesmix/resources/semihdp_params.asciipb", &params);
  params.set_rest_allocs_update(update_c);

  SemiHdpSampler sampler(data, hier, params);
  sampler.run(500, nburn, niter, 5, &collector, pseudoprior_collectors, true,
              200);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Finished running, duration: " << duration << std::endl;
  collector.write_to_file<bayesmix::SemiHdpState>(chainfile);
}

int main() {
  // Scenario VII
  std::vector<MatrixXd> data(100);

  auto& rng = bayesmix::Rng::Instance().get();

  for (int i = 0; i < 20; i++) {
    auto rng = bayesmix::Rng::Instance().get();
    data[i] = generate_mixture(-5, 1.0, 5, 1.0, 0.5, 100, rng);
  }
  for (int i = 20; i < 40; i++) {
    auto rng = bayesmix::Rng::Instance().get();
    data[i] = generate_mixture(-5.0, 1.0, 0.0, 1.0, 0.5, 100, rng);
  }
  for (int i = 40; i < 60; i++) {
    auto rng = bayesmix::Rng::Instance().get();
    data[i] = generate_mixture(0.0, 1.0, 5.0, 0.1, 0.5, 100, rng);
  }
  for (int i = 60; i < 80; i++) {
    auto rng = bayesmix::Rng::Instance().get();
    data[i] = generate_mixture(-10, 1.0, 0.0, 1.0, 0.5, 100, rng);
  }
  for (int i = 80; i < 100; i++) {
    auto rng = bayesmix::Rng::Instance().get();
    data[i] = generate_mixture(-10, 1.0, 0.0, 1.0, 0.1, 100, rng);
  }

  run_semihdp2(data,
               "/home/mario/PhD/exchangeability/semihdp-scripts/"
               "new_chains/s100.recordio",
               "metro_dist");
}