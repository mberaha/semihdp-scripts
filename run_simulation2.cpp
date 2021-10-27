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
                          int n) {
  auto& rng = bayesmix::Rng::Instance().get();
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
  int nadapt = 10000;
  int nburn = 5000;
  int niter = 5000;
  MemoryCollector collector;

  bayesmix::SemiHdpParams params;
  bayesmix::read_proto_from_file(
      "/home/mario/dev/bayesmix/resources/semihdp_params.asciipb", &params);
  params.set_rest_allocs_update(update_c);

  SemiHdpSampler sampler(data, hier, params);
  sampler.run(nadapt, nburn, niter, 5, &collector, pseudoprior_collectors,
              true, 200);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Finished running, duration: " << duration << std::endl;

  collector.write_to_file<bayesmix::SemiHdpState>(chainfile);
}

int main() {
  // Scenario IV std::vector<MatrixXd> data1(4);
  std::vector<MatrixXd> data1(4);
  std::cout << "data1.size(): " << data1.size() << std::endl;
  for (int i = 0; i < 4; i++) {
    data1[i] = MatrixXd::Zero(100, 1);
  }
  std::cout << data1[0].transpose() << std::endl;
  std::cout << "assigning stuff to data1" << std::endl;

  for (int j = 0; j < 100; j++) {
    for (int i = 0; i < 3; i++) {
      auto& rng = bayesmix::Rng::Instance().get();
      data1[i](j, 0) = stan::math::normal_rng(0.0, 1.0, rng);
    }
    auto& rng = bayesmix::Rng::Instance().get();
    data1[3](j, 0) = stan::math::skew_normal_rng(0.0, 1.0, 1.0, rng);
  }

  std::cout << data1[0].transpose() << std::endl;
  std::cout << data1[1].transpose() << std::endl;
  std::cout << data1[2].transpose() << std::endl;
  std::cout << data1[3].transpose() << std::endl;

  std::cout << "Data1 OK" << std::endl;
  run_semihdp2(data1,
               "/home/mario/PhD/exchangeability/semihdp-scripts/"
               "new_chains/s2e1_fullv2.recordio");
  // run_semihdp2(data1,
  //              "/home/mario/PhD/exchangeability/semihdp-scripts/"
  //              "new_chains/s2e1_metro_basev2.recordio",
  //              "metro_base");
  // run_semihdp2(data1,
  //              "/home/mario/PhD/exchangeability/semihdp-scripts/"
  //              "new_chains/s2e1_metro_distv2.recordio",
  //              "metro_dist");

  // // Scenario V
  // auto& rng = bayesmix::Rng::Instance().get();
  // std::vector<MatrixXd> data2(4);
  // for (int i = 0; i < 4; i++) {
  //   data2[i].resize(100, 1);
  // }

  // for (int j = 0; j < 100; j++) {
  //   data2[0](j, 0) = stan::math::normal_rng(0, 1, rng);
  //   data2[3](j, 0) = stan::math::normal_rng(0, 1, rng);
  //   data2[1](j, 0) = stan::math::normal_rng(0, std::sqrt(2.25), rng);
  //   data2[2](j, 0) = stan::math::normal_rng(0, std::sqrt(0.25), rng);
  // }

  // std::cout << "Data2 OK" << std::endl;
  // run_semihdp2(data2,
  //              "/home/mario/PhD/exchangeability/semihdp-scripts/"
  //              "new_chains/s2e2.recordio");

  // // // Scenario VI
  // std::vector<MatrixXd> data3(4);
  // data3[0] = generate_mixture(0, 1, 5, 1, 0.5, 100);
  // data3[1] = generate_mixture(0, 1, 5, 1, 0.5, 100);
  // data3[2] = generate_mixture(0, 1, -5, 1, 0.5, 100);
  // data3[3] = generate_mixture(-5, 1, 5, 1, 0.5, 100);

  // std::cout << "Data3 OK" << std::endl;

  // run_semihdp2(data3,
  //              "/home/mario/PhD/exchangeability/semihdp-scripts/"
  //              "new_chains/s2e3.recordio");
}