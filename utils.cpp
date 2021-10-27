#include "utils.hpp"

using namespace Eigen;

void run_semihdp(const std::vector<MatrixXd> data, std::string chainfile) {
  // Collect pseudo priors
  std::vector<MemoryCollector> pseudoprior_collectors;
  pseudoprior_collectors.resize(data.size());
  bayesmix::DPPrior mix_prior;
  double totalmass = 1.0;
  mix_prior.mutable_fixed_value()->set_totalmass(totalmass);
  for (int i = 0; i < data.size(); i++) {
    auto mixing = std::make_shared<DirichletMixing>();
    mixing->set_prior(mix_prior);
    auto hier = std::make_shared<NNIGHierarchy>();
    bayesmix::NNIGPrior hier_prior;
    hier_prior.mutable_fixed_values()->set_mean(data[i].mean());
    hier_prior.mutable_fixed_values()->set_var_scaling(0.1);
    hier_prior.mutable_fixed_values()->set_shape(2.0);
    hier_prior.mutable_fixed_values()->set_scale(2.0);
    hier->set_prior(hier_prior);

    Neal2Algorithm sampler;
    sampler.set_maxiter(2000);
    sampler.set_burnin(1000);
    sampler.set_mixing(mixing);
    sampler.set_data(data[i]);
    sampler.set_initial_clusters(hier, 5);
    sampler.run(&pseudoprior_collectors[i]);

    std::cout << "collector.size(): " << pseudoprior_collectors[i].get_size()
              << std::endl;
  }

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
  hier->set_prior(hier_prior);

  int nburn = 10000;
  int niter = 10000;
  MemoryCollector shdp_collector;
  SemiHdpSampler sampler(data, hier);
  sampler.run(nburn, nburn, niter, 5, &shdp_collector, pseudoprior_collectors,
              true, 200);
  shdp_collector.write_to_file(chainfile);
}