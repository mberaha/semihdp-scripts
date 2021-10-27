#include "utils.hpp"

void run_semihdp(const std::vector<MatrixXd> data, std::string chainfile,
                 std::string update_c) {
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
  std::cout << hier_prior.DebugString() << std::endl;
  hier->get_mutable_prior()->CopyFrom(hier_prior);
  hier->initialize();
  std::cout << hier->get_mutable_prior()->DebugString() << std::endl;

  hier->sample_prior();

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

    std::cout << "collector.size(): " << pseudoprior_collectors[i].get_size()
              << std::endl;
  };

  int nburn = 10000;
  int niter = 10000;
  MemoryCollector shdp_collector;
  bayesmix::SemiHdpParams params;
  bayesmix::read_proto_from_file(
      "/home/mario/dev/bayesmix/resources/semihdp_params.asciipb", &params);
  params.set_rest_allocs_update(update_c);

  SemiHdpSampler sampler(data, hier, params);
  sampler.run(nburn, nburn, niter, 5, &shdp_collector, pseudoprior_collectors,
              true, 200);
  sampler.print_debug_string();
  shdp_collector.write_to_file<bayesmix::SemiHdpState>(chainfile);
}