import numpy as np
from sqgturb import SQG, rfft2, irfft2
import os
from tqdm import tqdm
from argparse import ArgumentParser
import uuid
import multiprocessing


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(
        description='Data Assimilation Experiment Configuration')
    parser.add_argument('--N', type=int, default=64, help='Grid size')
    parser.add_argument('--hrs', type=int, default=3,
                        help='Interval between frames in hours')
    parser.add_argument('--n_traj', type=int, default=1,
                        help='Number of trajectories')
    parser.add_argument('--n_times', type=int, default=100,
                        help='Number of time steps')
    parser.add_argument('--data_path', type=str, default='',
                        help='Directory to write the trajectories to')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Index of the first trajectory, used in the file names')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for the initial noise; trajectory i uses seed + i. '
                             'Unseeded when omitted.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        f"Generating data with N={args.N}, n_traj={args.n_traj}, n_times={args.n_times}, data_path={args.data_path}")

    if args.data_path:
        os.makedirs(args.data_path, exist_ok=True)

    for version in tqdm(range(args.start_index, args.start_index + args.n_traj)):
        # Run an SQG turbulence simulation and save it to netCDF and .npy.

        # model parameters.
        '''
        “dt" is directly related to “N" through the Courant-Friedrichs-Lewy condition — “dt" should become smaller if you increase “N",
        otherwise the numerical scheme becomes unstable.

        “diff_efold” is the numerical dissipation which ensures energy does not accumulate erroneously at the largest wavenumbers (smallest scales).

        As long the model does not crash, it will be ok to proceed.
        '''
        if args.N == 1024:
            N = 1024  # number of grid points in each direction (waves=N/2)
            dt = 40  # time step in seconds
            diff_efold = 900.  # time scale for hyperdiffusion at smallest resolved scale
        elif args.N == 512:
            N = 512
            dt = 90
            diff_efold = 1800.
        elif args.N == 256:
            N = 256
            dt = 90
            diff_efold = 86400./16.
        elif args.N == 192:
            N = 192
            dt = 300
            diff_efold = 86400./8.
        elif args.N == 128:
            N = 128
            dt = 600
            diff_efold = 86400./3.
        elif args.N == 96:
            N = 96
            dt = 900
            diff_efold = 86400./3.
        elif args.N == 64:
            N = 64
            dt = 1200
            diff_efold = 86400.
        elif args.N == 32:
            N = 32
            dt = 2400
            diff_efold = 86400. * 3
        elif args.N == 16:
            N = 16
            dt = 4800
            diff_efold = 86400. * 9

        norder = 8  # order of hyperdiffusion
        dealias = True  # dealiased with 2/3 rule?

        # Ekman damping coefficient r=dek*N**2/f, dek = ekman depth = sqrt(2.*Av/f))
        # Av (turb viscosity) = 2.5 gives dek = sqrt(5/f) = 223
        # for ocean Av is 1-5, land 5-50 (Lin and Pierrehumbert, 1988)
        # corresponding to ekman depth of 141-316 m over ocean.
        # spindown time of a barotropic vortex is tau = H/(f*dek), 10 days for
        # H=10km, f=0.0001, dek=100m.
        dek = 0  # applied only at surface if symmetric=False
        nsq = 1.e-4
        f = 1.e-4
        g = 9.8
        theta0 = 300
        H = 10.e3  # lid height
        r = dek*nsq/f
        U = 30  # jet speed
        Lr = np.sqrt(nsq)*H/f  # Rossby radius
        L = 20.*Lr
        # thermal relaxation time scale
        tdiab = 10.*86400  # in seconds
        # (if False, asymmetric equilibrium jet with zero wind at sfc)
        symmetric = True
        # parameter used to scale PV to temperature units.
        scalefact = f*theta0/g

        # create random noise
        if args.seed is not None:
            np.random.seed(args.seed + version)
        pv = np.random.normal(0, 100., size=(2, N, N)).astype(float)

        # add isolated blob on lid
        nexp = 20
        x = np.arange(0, 2.*np.pi, 2.*np.pi/N)
        y = np.arange(0., 2.*np.pi, 2.*np.pi/N)
        x, y = np.meshgrid(x, y)
        x = x.astype(float)
        y = y.astype(float)
        pv[1] = pv[1]+2000.*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
        # remove area mean from each level.
        for k in range(2):
            pv[k] = pv[k] - pv[k].mean()

        # get OMP_NUM_THREADS (threads to use) from environment.
        threads = int(os.getenv('OMP_NUM_THREADS', '1'))

        print(f'Available CPU threads: {multiprocessing.cpu_count()}')
        print(f'Using {threads} threads')
        print('Using %d threads' % threads)

        # single or double precision
        precision = 'single'  # pyfftw FFTs are twice as fast as double

        # initialize qg model instance
        model = SQG(pv, nsq=nsq, f=f, U=U, H=H, r=r, tdiab=tdiab, dt=dt,
                    diff_order=norder, diff_efold=diff_efold,
                    dealias=dealias, symmetric=symmetric, threads=threads,
                    precision=precision, tstart=0)

        hrs = args.hrs  # interval between frames in hours
        outputinterval = hrs*3600.  # interval between frames in seconds
        # Spin-up before saving starts (days); should be at least 100.
        n_days_min = 300
        tmin = n_days_min * 86400.
        n_days = args.n_times*hrs/24 + n_days_min  # number of days to simulate
        print(f"Simulating {n_days} days")
        tmax = n_days * 86400.
        nsteps = int(tmax/outputinterval)  # number of time steps to animate
        assert (
            tmax-tmin)/outputinterval == args.n_times, f"nsteps {(tmax-tmin)/outputinterval} != n_times {args.n_times}"

        print(
            f"Running SQG turbulence simulation, generating {(tmax-tmin)/outputinterval} frames, each {outputinterval/3600.} hours apart.")

        # set number of timesteps to integrate for each call to model.advance
        model.timesteps = int(outputinterval/model.dt)

        # Save data to a netCDF file.
        random_id = str(uuid.uuid4())[:4]
        savedata = f'{args.data_path}/sqg_N{N}_{hrs}hrly_steps_{args.n_times}_{version}_{random_id}'

        if savedata is not None:
            from netCDF4 import Dataset
            nc = Dataset(f'{savedata}.nc', mode='w',
                         format='NETCDF4_CLASSIC')
            nc.r = model.r
            nc.f = model.f
            nc.U = model.U
            nc.L = model.L
            nc.H = model.H
            nc.g = g
            nc.theta0 = theta0
            nc.nsq = model.nsq
            nc.tdiab = model.tdiab
            nc.dt = model.dt
            nc.diff_efold = model.diff_efold
            nc.diff_order = model.diff_order
            nc.symmetric = int(model.symmetric)
            nc.dealias = int(model.dealias)
            x = nc.createDimension('x', N)
            y = nc.createDimension('y', N)
            z = nc.createDimension('z', 2)
            t = nc.createDimension('t', None)
            pvvar =\
                nc.createVariable(
                    'pv', float, ('t', 'z', 'y', 'x'), zlib=True)
            pvvar.units = 'K'
            # pv scaled by g/(f*theta0) so du/dz = d(pv)/dy
            xvar = nc.createVariable('x', float, ('x',))
            xvar.units = 'meters'
            yvar = nc.createVariable('y', float, ('y',))
            yvar.units = 'meters'
            zvar = nc.createVariable('z', float, ('z',))
            zvar.units = 'meters'
            tvar = nc.createVariable('t', float, ('t',))
            tvar.units = 'seconds'
            xvar[:] = np.arange(0, model.L, model.L/N)
            yvar[:] = np.arange(0, model.L, model.L/N)
            zvar[0] = 0
            zvar[1] = model.H

        nout = 0

        levplot = 1
        t = 0.0
        initial_t = t

        # Total expected iterations, for the progress bar.
        if hasattr(tmax, 'size'):
            tmax_value = float(tmax.size)
        else:
            tmax_value = float(tmax)

        # Now calculate iterations with proper types
        total_iterations = int(
            (tmax_value - t) / (model.dt * model.timesteps)) + 1

        progress_bar = tqdm(total=total_iterations,
                            desc="Simulation progress", unit="steps")

        steps_completed = 0
        while t < tmax:
            model.advance()
            t = model.t
            pv = irfft2(model.pvspec)
            hr = t/3600.

            steps_completed += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"Hours": f"{hr:.2f}"})

            if savedata is not None and t >= tmin:
                pvvar[nout, :, :, :] = pv
                tvar[nout] = t
                nc.sync()
                if t >= tmax:
                    nc.close()
                nout = nout + 1

        progress_bar.close()

        nc = Dataset(f'{savedata}.nc', 'r')
        X = np.array(nc['pv'][:])
        nc.close()
        np.save(f'{savedata}.npy', X)
