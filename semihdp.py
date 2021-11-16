import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from tempfile import TemporaryDirectory

SEMIHDP_HOME_DIR = os.path.dirname(os.path.realpath(__file__))
SEMIHDP_EXEC =  os.path.join(SEMIHDP_HOME_DIR, 'build/run_from_file')
BASE_CMD = SEMIHDP_EXEC + ' ' + "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}"
PARAMS_FILE = os.path.join(SEMIHDP_HOME_DIR, 'semihdp_params.asciipb')


def run_mcmc_from_files(data_path, dens_grid_path, output_path, 
                        niter, nburn, thin, update_c="full"):
    chainfile = os.path.join(output_path, "chains.recordio")
    c_file = os.path.join(output_path, "c.txt")
    latent_vars_file = os.path.join(output_path, "latent_vars.txt")
    dens_path = os.path.join(output_path, "dens/")
    os.makedirs(dens_path)
    cmd = BASE_CMD.format(
            data_path, PARAMS_FILE, chainfile, c_file,
            latent_vars_file, dens_grid_path, 
            dens_path, niter, nburn, thin, update_c)
        
    cmd = cmd.split(" ")
    subprocess.call(cmd, cwd=SEMIHDP_HOME_DIR)
    return  


def load_output(output_path, ngroups):
    c_file = os.path.join(output_path, "c.txt")
    latent_vars_file = os.path.join(output_path, "latent_vars.txt")
    dens_path = os.path.join(output_path, "dens/")

    c = np.loadtxt(c_file, delimiter=",")
    latent_vars = np.loadtxt(latent_vars_file, delimiter=",")
    log_dens = []
    for i in range(ngroups):
        fname = os.path.join(dens_path, "group_{0}.csv".format(i))
        log_dens.append(np.loadtxt(fname, delimiter=","))
    return c, latent_vars, log_dens                


def run_mcmc(data: list, dens_grid: np.array, niter=1000, nburn=1000, thin=10, update_c="full"):
    """
    Runs the semihpd sampler by calling the executable from a subprocess.
    Arguments
    ---------
    data: list of np.arrays, each entry is the data in one of the groups
    dens_grid: np.array, the grid on which to evaluate the density of all the groups
    niter: int, number of iterations to run the sampler
    nburn: int, number of burn-in iterations 
    thin: int, thinning factor
    update_c: str, either "full", "metro_base" or "metro_dist". 
        The update rule for the restourants allocations

    The sampler will be ran for niter + nburn iterations.

    Returns
    -------
    rest_allocs: np.array, of dimension [num_iter, num_groups]
        The parameters c_i's for each iteration
    latent_vars: np.array, of dimension [num_iter, 4] the colums are
        group_id, datum_id, mean (resp. variance) of the latent variable 
        associated to the observation
    log_dens: list of np.arrays, each entry is the evaluation of log_density of 
        one of the groups in each of the mcmc iterations
    """

    ngroups = len(data)
    data_ = np.vstack([
        np.column_stack([np.ones_like(x) * i, x]) for i, x in enumerate(data)])
    with TemporaryDirectory(prefix=SEMIHDP_HOME_DIR+"/") as tmpdir:
        data_path = os.path.join(tmpdir, 'data.txt')
        np.savetxt(data_path, data_, delimiter=",")
        grid_path = os.path.join(tmpdir, 'grid.txt')
        np.savetxt(grid_path, dens_grid, delimiter=",")

        run_mcmc_from_files(data_path, grid_path, tmpdir, niter, 
                            nburn, thin, update_c)

        return load_output(tmpdir, ngroups)

        
if __name__ == "__main__":
    data = [np.random.normal(0, 1, size=100),
            np.random.normal(0, 1, size=100)]
    dens_grid = np.linspace(-5, 5, 100)
    c, latent_vars, log_dens = run_mcmc(data, dens_grid)
    plt.plot(dens_grid, np.exp(np.mean(log_dens[0], axis=0)))
    plt.show()

