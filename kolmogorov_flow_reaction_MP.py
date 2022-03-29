"""
Dedalus script for 2D Kolmogorov Flow

This script uses a Fourier basis in the x and y directions with periodic boundary
conditions.  The equations are non-dimensional.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 kolmogorov_flow_fenep.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.

To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 kolmogorov_flow_fenep.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ ln -s snapshots/snapshots_s2.h5 restart.h5
    $ mpiexec -n 4 python3 kolmogorov_flow_fenep.py

The simulations should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Flow Parameters
Lx, Ly = (2*np.pi, 2*np.pi)
Nx = Ny = 256
Reynolds = 100.
n_K = 1
dtmax = 1.0e-3
tmax = 3.00
# Polymer parameters
reaction = True #If False the simulation runs a simple Kolmogorov flow. If True the momentum equation is modified and the polymer model is added
Peclet = 1000.
Lmax = 100.
k1 = 10
k2 = 1

# snapshot parameters
snapshots_directory = 'run00'
ifreq = 100 # snapshots, flow and simulations diagnostics are saved every ifreq iteration
# Create bases and domain
x_basis = de.Fourier('x', Nx, interval=(-Lx/2., Lx/2.), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(-Ly/2., Ly/2.), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Definition of variables and viscosities

problem = de.IVP(domain, variables=['p','u','v','A','B','R','S'])
problem.parameters['Peinv'] = 1./Peclet # viscosity for the momentum viscous term
problem.parameters['k1'] = k1 # viscosity for the polymer term in the momentum equation
problem.parameters['k2'] = k2 #Diffusivity for C_ij



problem.parameters['Reinv'] = 1./Reynolds

# definition of the flow forcing
problem.parameters['n_K'] = n_K
problem.substitutions['F'] = 'sin(n_K*y)'
# Definition of polymer parameters and part of the C_ij equations


problem.substitutions['rhsA'] = '-k1*A*B'
problem.substitutions['rhsB'] = '-k1*A*B - k2*B*R'
problem.substitutions['rhsR'] = 'k1*A*B - k2*B*R'
problem.substitutions['rhsS'] = 'k2*B*R'
# This part is common to both flows -> incompressibility
problem.add_equation("dx(u) + dy(v) = 0", condition=('nx!=0 or ny!=0'))
problem.add_equation('p = 0', condition=('nx==0 and ny==0'))

# Definition of flow equations
problem.add_equation("dt(u) - Reinv*(dx(dx(u)) + dy(dy(u))) + dx(p) = -(u*dx(u) + v*dy(u)) + F")
problem.add_equation("dt(v) - Reinv*(dx(dx(v)) + dy(dy(v))) + dy(p) = -(u*dx(v) + v*dy(v))")

problem.add_equation("dt(A) - Peinv*(dx(dx(A)) + dy(dy(A))) = -(u*dx(A) + v*dy(A)) + rhsA") #Report eq's
problem.add_equation("dt(B) - Peinv*(dx(dx(B)) + dy(dy(B))) = -(u*dx(B) + v*dy(B)) + rhsB")
problem.add_equation("dt(R) - Peinv*(dx(dx(R)) + dy(dy(R))) = -(u*dx(R) + v*dy(R)) + rhsR")
problem.add_equation("dt(S) - Peinv*(dx(dx(S)) + dy(dy(S))) = -(u*dx(S) + v*dy(S)) + rhsS")
# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():

    # Initial conditions
    x, y = domain.all_grids()
    u = solver.state['u']
    v = solver.state['v']
    
    A = solver.state['A']
    B = solver.state['B']
    R = solver.state['R']
    S = solver.state['S']



    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    # Laminar background + Taylor-green vortices + perturbations

    pert =  1e-2 * noise
    u['g'] = np.sin(n_K*y)+0.1*np.cos(n_K*x)*np.sin(n_K*y) + pert
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]
    pert =  1e-2 * noise
    v['g'] = -0.1*np.sin(n_K*x)*np.cos(n_K*y)+pert
    A['g'] = 0.5*(1+np.tanh(np.sin(n_K*y)))
    B['g'] = 0.5*(1-np.tanh(np.sin(n_K*y)))


    # Timestepping and output
    dt = dtmax
    stop_sim_time = tmax
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

    # Timestepping and output
    dt = last_dt
    stop_sim_time = tmax
    fh_mode = 'overwrite'

# Integration parameters
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler(snapshots_directory, sim_dt=0.25, max_writes=50, mode=fh_mode)
snapshots.add_system(solver.state)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=dtmax, threshold=0.05)
CFL.add_velocities(('u', 'v'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)

flow.add_property("sqrt(u*u + v*v) / Reinv", name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    pit_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % ifreq == 0:
            pitE_time = time.time()
            tpit = (pitE_time - pit_time)/ifreq
            pit_time = time.time()
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
            logger.info('cpu time/it = %e' % tpit)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
