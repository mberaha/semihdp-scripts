#include "utils.hpp"
#include "src/utils/proto_utils.h"
#include "src/utils/eigen_utils.h"


MemoryCollector run_semihdp(const std::vector<MatrixXd> data, 
                 std::string chainfile,
                 std::string params_file,
                int niter, int nburn, int thin, std::string update_c) {
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
    sampler.set_verbose(false);
    sampler.run(&pseudoprior_collectors[i]);

    std::cout << "collector.size(): " << pseudoprior_collectors[i].get_size()
              << std::endl;
  };

  MemoryCollector shdp_collector;
  bayesmix::SemiHdpParams params;
  bayesmix::read_proto_from_file(params_file, &params);
  std::cout << params.DebugString() << std::endl;
  params.set_rest_allocs_update(update_c);

  SemiHdpSampler sampler(data, hier, params);
  sampler.run(1000, nburn, niter, thin, &shdp_collector, 
              pseudoprior_collectors, true, 200);
  sampler.print_debug_string();
  shdp_collector.write_to_file<bayesmix::SemiHdpState>(chainfile);
  return shdp_collector;
}

std::vector<Eigen::MatrixXd> eval_uni_dens(
        MemoryCollector& coll, const Eigen::VectorXd &xgrid, int ngroups) {
    std::vector<Eigen::MatrixXd> densities(ngroups);
    int nsteps = coll.get_size();
    int nx = xgrid.size();
    for (int i = 0; i < ngroups; i++) {
        densities[i] = Eigen::MatrixXd::Zero(nsteps, nx);
    }

    for (int i=0; i < nsteps; i++) {
        bayesmix::SemiHdpState state;
        coll.get_state(i, &state);
        for (int j=0; j < ngroups; j++) {
            auto rest = state.restaurants(state.c(j));
            const int* p = &(rest.n_by_clus())[0];
            Eigen::VectorXi cnts = Eigen::Map<const Eigen::VectorXi>(
                    p, rest.n_by_clus_size());
            Eigen::VectorXd weights = cnts;
            weights /= weights.sum();
            Eigen::VectorXd log_weights = weights.array().log();

            Eigen::MatrixXd curr(rest.theta_stars_size(), nx);

            for (int k=0; k < rest.theta_stars_size(); k++) {
                NNIGHierarchy hier;
                hier.set_state_from_proto(rest.theta_stars(k));
                curr.row(k) = hier.like_lpdf_grid(xgrid).transpose();
                curr.row(k) = curr.row(k).array() + log_weights(k); 
            }
            for (int l=0; l < nx; l++) {
                densities[j](i, l) = stan::math::log_sum_exp(curr.col(l));
            }
        }
    }
    return densities;
}

Eigen::MatrixXd get_latent_vars(
        MemoryCollector& coll, int ngroups) {
    
    int nsteps = coll.get_size();
    std::deque<Eigen::VectorXd> latent_vars;

    for (int l=0; l < nsteps; l++) {
        bayesmix::SemiHdpState state;
        coll.get_state(l, &state);
        for (int i=0; i < ngroups; i++) {
            auto rest = state.restaurants(state.c(i));
            auto clus_allocs = state.groups(i).cluster_allocs();
            for (int j=0; j < clus_allocs.size(); j++) {
                Eigen::VectorXd curr(4);
                curr(0) = l;
                curr(1) = i;
                curr(2) = rest.theta_stars(clus_allocs[j]).uni_ls_state().mean();
                curr(3) = rest.theta_stars(clus_allocs[j]).uni_ls_state().var();
                latent_vars.push_back(curr);
            }
        }
    }
    return bayesmix::stack_vectors(latent_vars);
}
