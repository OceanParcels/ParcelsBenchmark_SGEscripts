"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4, AdvectionRK4_3D, DiffusionUniformKh
from parcels import FieldSet, ScipyParticle, JITParticle, Variable, StateCode, OperationCode, ErrorCode
# from parcels.particleset_benchmark import ParticleSet_Benchmark as BenchmarkParticleSet
# from parcels.particleset import ParticleSet as DryParticleSet
from parcels import BenchmarkParticleSetSOA, BenchmarkParticleSetAOS, BenchmarkParticleSetNodes
from parcels import ParticleSetSOA, ParticleSetAOS, ParticleSetNodes
from parcels import GenerateID_Service, SequentialIdGenerator, LibraryRegisterC  # noqa
from parcels.field import VectorField, NestedField, SummedField
# from parcels import plotTrajectoriesFile_loadedField
# from parcels import rng as random
from parcels.tools import logger
from parcels import ParcelsRandom
from datetime import timedelta as delta
import math
from argparse import ArgumentParser
import datetime
import numpy as np
import fnmatch
import sys
import gc
import os
import time as ostime
from glob import glob
# import matplotlib.pyplot as plt
from parcels.tools import perlin3d
from parcels.tools import perlin2d

try:
    from mpi4py import MPI
except:
    MPI = None
with_GC = False

pset_modes = ['soa', 'aos', 'nodes']
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
pset_types_dry = {'soa': {'pset': ParticleSetSOA},  # , 'pfile': ParticleFileSOA, 'kernel': KernelSOA
                  'aos': {'pset': ParticleSetAOS},  # , 'pfile': ParticleFileAOS, 'kernel': KernelAOS
                  'nodes': {'pset': ParticleSetNodes}}  # , 'pfile': ParticleFileNodes, 'kernel': KernelNodes
pset_types = {'soa': {'pset': BenchmarkParticleSetSOA},
              'aos': {'pset': BenchmarkParticleSetAOS},
              'nodes': {'pset': BenchmarkParticleSetNodes}}
method = {'rk4': AdvectionRK4, 'ee': AdvectionEE, 'rk45': AdvectionRK45, 'bm': AdvectionRK4}
global_t_0 = 0
Nparticle = int(math.pow(2,10)) # equals to Nparticle = 1024
#Nparticle = int(math.pow(2,11)) # equals to Nparticle = 2048
#Nparticle = int(math.pow(2,12)) # equals to Nparticle = 4096
#Nparticle = int(math.pow(2,13)) # equals to Nparticle = 8192
#Nparticle = int(math.pow(2,14)) # equals to Nparticle = 16384
#Nparticle = int(math.pow(2,15)) # equals to Nparticle = 32768
#Nparticle = int(math.pow(2,16)) # equals to Nparticle = 65536
#Nparticle = int(math.pow(2,17)) # equals to Nparticle = 131072
#Nparticle = int(math.pow(2,18)) # equals to Nparticle = 262144
#Nparticle = int(math.pow(2,19)) # equals to Nparticle = 524288

a = 9.6 * 1e3 # [a in km -> 10e3]
b = 4.8 * 1e3 # [b in km -> 10e3]
c = 1.0
tsteps = 122 # in steps
tstepsize = 6.0 # unitary
tscale = 12.0*60.0*60.0 # in seconds
# time[days] = (tsteps * tstepsize) * tscale
# gyre_rotation_speed = 60.0*24.0*60.0*60.0  # assume 1 rotation every 8.5 weeks
gyre_rotation_speed = (366.0*24.0*60.0*60.0)/2.0  # assume 1 rotation every 8.5 weeks
# scalefactor = (40.0 / (1000.0/ (60.0 * 60.0)))  # 40 km/h
scalefactor = ((4.0*1000) / (60.0*60.0))  # 4 km/h
vertical_scale = (800.0 / (24*60.0*60.0))  # 800 m/d
v_scale_small = 1.0/(40.0*1000.0*1000.0) # this is to adapt, cause 1 U = 1 m/s = 1 spatial unit / time unit; spatial scale; domain = 9600 m x 4800 m; earth: 40075 km x  200004 km


def DeleteParticle(particle, fieldset, time):
    particle.delete()


def RenewParticle(particle, fieldset, time):
    EA = fieldset.east_lim
    WE = fieldset.west_lim
    NO = fieldset.north_lim
    SO = fieldset.south_lim

    particle.lon = WE + ParcelsRandom.random() * (EA - WE)
    particle.lat = SO + ParcelsRandom.random() * (NO - SO)
    # if fieldset.isThreeD > 0.0:
    #     TO = fieldset.top
    #     BO = fieldset.bottom
    #     particle.depth = TO + ((BO-TO) / 0.75) + ((BO-TO) * -0.5 * ParcelsRandom.random())


def WrapClipParticle(particle, fieldset, time):
    EA = fieldset.east_lim
    WE = fieldset.west_lim
    dlon = EA - WE
    NO = fieldset.north_lim
    SO = fieldset.south_lim
    if particle.lon >= EA:
        particle.lon -= dlon
    if particle.lon < WE:
        particle.lon += dlon
    if particle.lat >= NO:
        particle.lat = NO - 0.001
    if particle.lat < SO:
        particle.lat = SO + 0.001


def reflect_top_bottom(particle, fieldset, time):
    span = fieldset.bottom - fieldset.top
    while particle.depth <= fieldset.top:
        particle.depth += 0.01 * span
    while particle.depth >= fieldset.bottom:
        particle.depth -= 0.01 * span


def wrap_top_bottom(particle, fieldset, time):
    span = fieldset.bottom - fieldset.top
    if particle.depth <= fieldset.top:
        particle.depth += span
    if particle.depth >= fieldset.bottom:
        particle.depth -= span


def clip_top_bottom(particle, fieldset, time):
    if particle.depth < fieldset.top:
        particle.depth = fieldset.top
    if particle.depth > fieldset.bottom:
        particle.depth = fieldset.bottom


periodicBC = WrapClipParticle


def perIterGC():
    gc.collect()


def doublegyre_from_numpy(xdim=960, ydim=480, periodic_wrap=False, write_out=False, steady=False, mesh='flat', simtime_days=None, diffusion=False):
    """Implemented following Froyland and Padberg (2009), 10.1016/j.physd.2009.03.002"""
    A = 0.3
    epsilon = 0.25
    omega = 2 * np.pi
    scalefac = scalefactor
    if 'flat' in mesh:  # and np.maximum(a, b) > 370.0 and np.maximum(a, b) < 100000:
        scalefac *= v_scale_small


    lon = np.linspace(-a*0.5, a*0.5, xdim, dtype=np.float32)
    lonrange = lon.max()-lon.min()
    sys.stdout.write("lon field: {}\n".format(lon.size))
    lat = np.linspace(-b*0.5, b*0.5, ydim, dtype=np.float32)
    latrange = lat.max() - lat.min()
    sys.stdout.write("lat field: {}\n".format(lat.size))
    totime = (tsteps * tstepsize) * tscale
    times = np.linspace(0., totime, tsteps, dtype=np.float64)
    sys.stdout.write("time field: {}\n".format(times.size))
    dx, dy = lon[2]-lon[1], lat[2]-lat[1]

    U = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
    V = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
    freqs = np.ones(times.size, dtype=np.float32)
    if not steady:
        for ti in range(times.shape[0]):
            time_f = times[ti] / gyre_rotation_speed
            # time_f = np.fmod((times[ti])/gyre_rotation_speed, 2.0)
            # time_f = np.fmod((times[ti]/gyre_rotation_speed), 2*np.pi)
            # freqs[ti] = omega * np.cos(time_f) * 2.0
            freqs[ti] *= omega * time_f
            # freqs[ti] *= time_f
    else:
        freqs = (freqs * 0.5) * omega

    for ti in range(times.shape[0]):
        freq = freqs[ti]
        # print(freq)
        for i in range(lon.shape[0]):
            for j in range(lat.shape[0]):
                x1 = ((lon[i]*2.0 + a) / a) # - dx / 2
                x2 = ((lat[j]*2.0 + b) / (2.0*b)) # - dy / 2
                f_xt = (epsilon * np.sin(freq) * x1**2.0) + (1.0 - (2.0 * epsilon * np.sin(freq))) * x1
                U[ti, j, i] = -np.pi * A * np.sin(np.pi * f_xt) * np.cos(np.pi * x2)
                V[ti, j, i] = np.pi * A * np.cos(np.pi * f_xt) * np.sin(np.pi * x2) * (2 * epsilon * np.sin(freq) * x1 + 1 - 2 * epsilon * np.sin(freq))
                # in 3D, make U and V shape a x^2 hyperbolic in depth
                # in 3 3D, the V-func will be also used for W

    U *= scalefac
    # U = np.transpose(U, (0, 2, 1))
    V *= scalefac
    # V = np.transpose(V, (0, 2, 1))

    data = {'U': U, 'V': V}
    dimensions = {'time': times, 'lon': lon, 'lat': lat}
    simtime_days = 365 if simtime_days is None else simtime_days
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_data(data, dimensions, mesh=mesh, transpose=False, time_periodic=delta(days=simtime_days))
    else:
        fieldset = FieldSet.from_data(data, dimensions, mesh=mesh, transpose=False, allow_time_extrapolation=True)

    Kh_zonal = None
    Kh_meridional = None
    if diffusion:
        Kh_zonal = np.random.uniform(9.5, 10.5)  # in m^2/s
        Kh_meridional = np.random.uniform(7.5, 12.5)  # in m^2/s
        if 'flat' in mesh:
            Kh_zonal, Kh_meridional = Kh_zonal * 100.0, Kh_meridional * 100.0  # because the mesh is flat

    if write_out:
        fieldset.write(filename=write_out)
    if diffusion:
        fieldset.add_constant_field("Kh_zonal", Kh_zonal, mesh=mesh)
        fieldset.add_constant_field("Kh_meridional", Kh_meridional, mesh=mesh)
    return a, b, c, lon, lat, times, fieldset


def fieldset_from_file(periodic_wrap=False, filepath=None, simtime_days=None, diffusion=False, chunk_level=0):
    """

    """
    if filepath is None:
        return None
    extra_fields = {}
    head_dir = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    fname, fext = os.path.splitext(basename)
    fext = ".nc" if len(fext) == 0 else fext
    flen = len(fname)
    field_fnames = glob(os.path.join(head_dir, fname+"*"+fext))
    for field_fname in field_fnames:
        field_fname, field_fext = os.path.splitext(os.path.basename(field_fname))
        field_indicator = field_fname[flen:]
        if field_indicator not in ["U", "V"]:
            extra_fields[field_indicator] = field_indicator
    if len(list(extra_fields.keys())) < 1:
        extra_fields = None

    a, b, c = 1.0, 1.0, 1.0
    simtime_days = 366 if simtime_days is None else simtime_days

    nemo_nchs = None
    if chunk_level > 1:
        nemo_nchs = {
            'U': {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('depthu', 25), 'time': ('time_counter', 1)},  #
            'V': {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('depthv', 25), 'time': ('time_counter', 1)},  #
        }
    elif chunk_level > 0:
        nemo_nchs = {
            'U': 'auto',
            'V': 'auto'
        }
    else:
        nemo_nchs = False

    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_parcels(os.path.join(head_dir, fname), extra_fields=extra_fields, time_periodic=delta(days=simtime_days), deferred_load=True, allow_time_extrapolation=False, chunksize=nemo_nchs)
        # return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', time_periodic=delta(days=366))
    else:
        fieldset = FieldSet.from_parcels(os.path.join(head_dir, fname), extra_fields=extra_fields, time_periodic=None, deferred_load=True, allow_time_extrapolation=True, chunksize=nemo_nchs)
        # return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', allow_time_extrapolation=True)
    times = fieldset.time_origin
    lon = fieldset.U.lon
    a = lon[len(lon)-1] - lon[0]
    lat = fieldset.V.lat
    b = lat[len(lat) - 1] - lat[0]
    if extra_fields is not None and "W" in extra_fields:
        depth = fieldset.W.depth
        c = depth[len(depth)-1] - depth[0]
        fieldset.add_constant("top", depth[0] + 0.001)
        fieldset.add_constant("bottom", depth[len(depth)-1] - 0.001)
    Kh_zonal = None
    Kh_meridional = None
    if diffusion:
        Kh_zonal = np.random.uniform(9.5, 10.5)  # in m^2/s
        Kh_meridional = np.random.uniform(7.5, 12.5)  # in m^2/s
        Kh_zonal, Kh_meridional = Kh_zonal * 100.0, Kh_meridional * 100.0  # because the mesh is flat
        fieldset.add_constant_field("Kh_zonal", Kh_zonal, mesh="flat")
        fieldset.add_constant_field("Kh_meridional", Kh_meridional, mesh="flat")
    return a, b, c, lon, lat, times, fieldset


class AgeParticle_JIT(JITParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0)


class AgeParticle_SciPy(ScipyParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0)


def UniformDiffusion(particle, fieldset, time):
    dWx = 0.
    dWy = 0.
    bx = 0.
    by = 0.

    if particle.state == StateCode.Evaluate:
        # dWt = 0.
        dWt = math.sqrt(math.fabs(particle.dt))
        dWx = ParcelsRandom.normalvariate(0, dWt)
        dWy = ParcelsRandom.normalvariate(0, dWt)
        bx = math.sqrt(2 * fieldset.Kh_zonal)
        by = math.sqrt(2 * fieldset.Kh_meridional)

    particle.lon += bx * dWx
    particle.lat += by * dWy


def initialize(particle, fieldset, time):
    if particle.initialized_dynamic < 1:
        np_scaler = math.sqrt(3.0 / 2.0)
        particle.life_expectancy = time + ParcelsRandom.uniform(.0, (fieldset.life_expectancy-time) * 2.0 / np_scaler)
        # particle.life_expectancy = time + ParcelsRandom.uniform(.0, (fieldset.life_expectancy-time)*math.sqrt(3.0 / 2.0))
        # particle.life_expectancy = time + ParcelsRandom.uniform(.0, fieldset.life_expectancy) * math.sqrt(3.0 / 2.0)
        # particle.life_expectancy = time+ParcelsRandom.uniform(.0, fieldset.life_expectancy) * ((3.0/2.0)**2.0)
        particle.initialized_dynamic = 1


def Age(particle, fieldset, time):
    if particle.state == StateCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > particle.life_expectancy:
        particle.delete()


age_ptype = {'scipy': AgeParticle_SciPy, 'jit': AgeParticle_JIT}

if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="performance_doublegyre.png", help="image file name of the plot")
    parser.add_argument("-b", "--backwards", dest="backwards", action='store_true', default=False, help="enable/disable running the simulation backwards")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-r", "--release", dest="release", action='store_true', default=False, help="continuously add particles via repeatdt (default: False)")
    parser.add_argument("-rt", "--releasetime", dest="repeatdt", type=int, default=720, help="repeating release rate of added particles in Minutes (default: 720min = 12h)")
    parser.add_argument("-a", "--aging", dest="aging", action='store_true', default=False, help="Removed aging particles dynamically (default: False)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-w", "--writeout", dest="write_out", action='store_true', default=False, help="write data in outfile")
    parser.add_argument("-d", "--delParticle", dest="delete_particle", action='store_true', default=False, help="switch to delete a particle (True) or reset a particle (default: False).")
    parser.add_argument("-A", "--animate", dest="animate", action='store_true', default=False, help="animate the particle trajectories during the run or not (default: False).")
    parser.add_argument("-V", "--visualize", dest="visualize", action='store_true', default=False, help="Visualize particle trajectories at the end (default: False). Requires -w in addition to take effect.")
    parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="2**6", help="number of particles to generate and advect (default: 2e6)")
    parser.add_argument("-sN", "--start_n_particles", dest="start_nparticles", type=str, default="96", help="(optional) number of particles generated per release cycle (if --rt is set) (default: 96)")
    parser.add_argument("-m", "--mode", dest="compute_mode", choices=['jit','scipy'], default="jit", help="computation mode = [JIT, SciPy]")
    parser.add_argument("-tp", "--type", dest="pset_type", default="soa", help="particle set type = [SOA, AOS, Nodes]")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    parser.add_argument("-3D", "--threeD", dest="threeD", action='store_true', default=False, help="make a 3D-simulation (default: False).")
    parser.add_argument("-im", "--interp_mode", dest="interp_mode", choices=['rk4','rk45', 'ee', 'bm'], default="rk4", help="interpolation mode = [rk4, rk45, ee (Eulerian Estimation), bm (Brownian Motion)]")
    parser.add_argument("-chs", "--chunksize", dest="chs", type=int, default=0, help="defines the chunksize level: 0=None, 1='auto', 2=fine tuned; default: 0")
    parser.add_argument("--dry", dest="dryrun", action="store_true", default=False, help="Start dry run (no benchmarking and its classes")
    args = parser.parse_args()

    pset_type = str(args.pset_type).lower()
    assert pset_type in pset_types
    ParticleSet = pset_types[pset_type]['pset']
    if args.dryrun:
        ParticleSet = pset_types_dry[pset_type]['pset']

    imageFileName=args.imageFileName
    periodicFlag=args.periodic
    backwardSimulation = args.backwards
    repeatdtFlag=args.release
    repeatRateMinutes=args.repeatdt
    time_in_days = args.time_in_days
    agingParticles = args.aging
    with_GC = args.useGC
    interp_mode = args.interp_mode
    Nparticle = int(float(eval(args.nparticles)))
    target_N = Nparticle
    addParticleN = 1
    gauss_scaler = 3.0 / 2.0
    cycle_scaler = 7.0 / 4.0
    start_N_particles = int(float(eval(args.start_nparticles)))

    # ======================================================= #
    # new ID generator things
    # ======================================================= #
    idgen = None
    c_lib_register = None
    if pset_type == 'nodes':
        idgen = GenerateID_Service(SequentialIdGenerator)
        idgen.setDepthLimits(0., 1.0)
        idgen.setTimeLine(0, delta(days=time_in_days).total_seconds())
        c_lib_register = LibraryRegisterC()

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        if mpi_comm.Get_rank() == 0:
            if agingParticles and not repeatdtFlag:
                sys.stdout.write("N: {} ( {} )\n".format(Nparticle, int(Nparticle * gauss_scaler)))
            else:
                sys.stdout.write("N: {}\n".format(Nparticle))
    else:
        if agingParticles and not repeatdtFlag:
            sys.stdout.write("N: {} ( {} )\n".format(Nparticle, int(Nparticle * gauss_scaler)))
        else:
            sys.stdout.write("N: {}\n".format(Nparticle))

    dt_minutes = 60
    nowtime = datetime.datetime.now()
    ParcelsRandom.seed(nowtime.microsecond)
    np.random.seed(nowtime.microsecond)

    a, b, c = 1.0, 1.0, 1.0
    use_3D = args.threeD

    branch = "benchmark"
    computer_env = "local/unspecified"
    scenario = "doublegyre"
    odir = ""
    if fnmatch.fnmatchcase(os.uname()[1], "science-bs*"):  # Gemini
        odir = "/scratch/{}/experiments/parcels_benchmarking/{}".format("ckehl", str(args.pset_type))
        computer_env = "Gemini"
    elif os.uname()[1] in ["lorenz.science.uu.nl",] or fnmatch.fnmatchcase(os.uname()[1], "node*"):  # Lorenz
        CARTESIUS_SCRATCH_USERNAME = 'ckehl'
        odir = "/storage/shared/oceanparcels/output_data/data_{}/experiments/parcels_benchmarking/{}".format(CARTESIUS_SCRATCH_USERNAME, str(args.pset_type))
        computer_env = "Lorenz"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        odir = "/scratch/shared/{}/experiments/parcels_benchmarking/{}".format(CARTESIUS_SCRATCH_USERNAME, str(args.pset_type))
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "int*.snellius.*") or fnmatch.fnmatchcase(os.uname()[1], "fcn*") or fnmatch.fnmatchcase(os.uname()[1], "tcn*") or fnmatch.fnmatchcase(os.uname()[1], "gcn*") or fnmatch.fnmatchcase(os.uname()[1], "hcn*"):  # Snellius
        SNELLIUS_SCRATCH_USERNAME = 'ckehluu'
        odir = "/scratch-shared/{}/experiments/parcels_benchmarking/{}".format(SNELLIUS_SCRATCH_USERNAME, str(args.pset_type))
        computer_env = "Snellius"
    else:
        odir = "/var/scratch/experiments/{}".format(str(args.pset_type))
    print("running {} on {} (uname: {}) - branch '{}' - (target) N: {} - argv: {}".format(scenario, computer_env, os.uname()[1], branch, target_N, sys.argv[1:]))

    if os.path.sep in imageFileName:
        head_dir = os.path.dirname(imageFileName)
        if head_dir[0] == os.path.sep:
            odir = head_dir
        else:
            odir = os.path.join(odir, head_dir)
            imageFileName = os.path.split(imageFileName)[1]
    pfname, pfext = os.path.splitext(imageFileName)
    if not os.path.exists(odir):
        os.makedirs(odir)

    func_time = []
    mem_used_GB = []

    fieldset = None
    field_fpath = None
    if args.write_out:
        field_fpath = os.path.join(odir,"doublegyre")
    if field_fpath is not None and os.path.exists(field_fpath+"U.nc"):
        a, b, c, lon, lat, times, fieldset = fieldset_from_file(periodic_wrap=periodicFlag, filepath=field_fpath+".nc", simtime_days=time_in_days, diffusion=(interp_mode == 'bm'), chunk_level=args.chs)
        use_3D &= hasattr(fieldset, "W")
    else:
        fout_path = False if field_fpath is None else field_fpath
        a, b, c, lon, lat, times, fieldset = doublegyre_from_numpy(xdim=960, ydim=480, periodic_wrap=periodicFlag, write_out=fout_path, simtime_days=time_in_days, diffusion=(interp_mode=='bm'))
        use_3D &= hasattr(fieldset, "W")
    fieldset.add_constant("east_lim", +a * 0.5)
    fieldset.add_constant("west_lim", -a * 0.5)
    fieldset.add_constant("north_lim", +b * 0.5)
    fieldset.add_constant("south_lim", -b * 0.5)
    fieldset.add_constant("isThreeD", 1.0 if use_3D else -1.0)
    fieldset.add_constant('life_expectancy', delta(days=time_in_days).total_seconds())
    fieldset.add_constant('gauss_scaler', gauss_scaler)
    nr_seasons = math.ceil(time_in_days / (366.0 / 4.0))
    days_per_season = math.floor(366.0 / 4.0)
    fieldset.add_constant('days_per_season', days_per_season)
    logger.info("a: {}; b: {}; c{}".format(a, b, c))

    if args.compute_mode == 'scipy':
        Nparticle = min(Nparticle, 2**10)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            global_t_0 = ostime.process_time()
    else:
        global_t_0 = ostime.process_time()

    simStart = None
    for f in fieldset.get_fields():
        if type(f) in [VectorField, NestedField, SummedField]:
            continue
        else:
            if backwardSimulation:
                simStart=f.grid.time_full[-1]
            else:
                simStart = f.grid.time_full[0]
            break

    if agingParticles:
        if not repeatdtFlag:
            Nparticle = int(Nparticle * gauss_scaler)
    if repeatdtFlag:
        addParticleN = Nparticle/2.0
        refresh_cycle = (delta(days=time_in_days).total_seconds() / (addParticleN/start_N_particles)) / cycle_scaler
        if agingParticles:
            refresh_cycle /= cycle_scaler
        repeatRateMinutes = int(refresh_cycle/60.0) if repeatRateMinutes == 720 else repeatRateMinutes

    if repeatdtFlag:
        pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()],
                           lon=np.random.rand(start_N_particles, 1) * (-a) + (a / 2.0),
                           lat=np.random.rand(start_N_particles, 1) * (-b) + (b / 2.0), time=simStart,
                           repeatdt=delta(minutes=repeatRateMinutes), idgen=idgen, c_lib_register=c_lib_register)
        if pset_type != 'nodes':
            psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()],
                                lon=np.random.rand(int(addParticleN), 1) * (-a) + (a / 2.0),
                                lat=np.random.rand(int(addParticleN), 1) * (-b) + (b / 2.0), time=simStart, idgen=idgen,
                                c_lib_register=c_lib_register)
            pset.add(psetA)
        else:
            lonlat_field = np.random.rand(int(addParticleN), 2)
            lonlat_field *= np.array([-a, -b])
            lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
            lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
            # time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
            time_field = simStart
            pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
            pset.add(pdata)

        # lonlat_field = np.random.rand(int(addParticleN), 2)
        # lonlat_field *= np.array([a, b])
        # lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
        # lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
        # # time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
        # time_field = simStart
        # pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
        # pset.add(pdata)
    else:
        pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()],
                           lon=np.random.rand(Nparticle, 1) * (-a) + (a / 2.0),
                           lat=np.random.rand(Nparticle, 1) * (-b) + (b / 2.0), time=simStart, idgen=idgen,
                           c_lib_register=c_lib_register)

    # if backwardSimulation:
    #     # ==== backward simulation ==== #
    #     if agingParticles:
    #         if repeatdtFlag:
    #             pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * (-a) + (a/2.0), lat=np.random.rand(start_N_particles, 1) * (-b) + (b/2.0), time=simStart, repeatdt=delta(minutes=repeatRateMinutes), idgen=idgen, c_lib_register=c_lib_register)
    #             if pset_type != 'nodes':
    #                 psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * (-a) + (a/2.0), lat=np.random.rand(int(addParticleN), 1) * (-b) + (b/2.0), time=simStart, idgen=idgen, c_lib_register=c_lib_register)
    #                 pset.add(psetA)
    #             else:
    #                 lonlat_field = np.random.rand(int(addParticleN), 2)
    #                 lonlat_field *= np.array([a, b])
    #                 lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
    #                 lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
    #                 # time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
    #                 time_field = simStart
    #                 pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
    #                 pset.add(pdata)
    #
    #             # lonlat_field = np.random.rand(int(addParticleN), 2)
    #             # lonlat_field *= np.array([a, b])
    #             # lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
    #             # lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
    #             # # time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
    #             # time_field = simStart
    #             # pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
    #             # pset.add(pdata)
    #         else:
    #             pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * (-a) + (a/2.0), lat=np.random.rand(Nparticle, 1) * (-b) + (b/2.0), time=simStart, idgen=idgen, c_lib_register=c_lib_register)
    #     else:
    #         if repeatdtFlag:
    #             pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * (-a) + (a/2.0), lat=np.random.rand(start_N_particles, 1) * (-b) + (b/2.0), time=simStart, repeatdt=delta(minutes=repeatRateMinutes), idgen=idgen, c_lib_register=c_lib_register)
    #             if pset_type != 'nodes':
    #                 psetA = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * (-a) + (a/2.0), lat=np.random.rand(int(addParticleN), 1) * (-b) + (b/2.0), time=simStart, idgen=idgen, c_lib_register=c_lib_register)
    #                 pset.add(psetA)
    #             else:
    #                 lonlat_field = np.random.rand(int(addParticleN), 2)
    #                 lonlat_field *= np.array([a, b])
    #                 lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
    #                 lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
    #                 # time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
    #                 time_field = simStart
    #                 pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
    #                 pset.add(pdata)
    #             # lonlat_field = np.random.rand(int(addParticleN), 2)
    #             # lonlat_field *= np.array([a, b])
    #             # lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
    #             # lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
    #             # # time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
    #             # time_field = simStart
    #             # pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
    #             # pset.add(pdata)
    #         else:
    #             pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * (-a) + (a/2.0), lat=np.random.rand(Nparticle, 1) * (-b) + (b/2.0), time=simStart, idgen=idgen, c_lib_register=c_lib_register)
    # else:
    #     # ==== forward simulation ==== #
    #     if agingParticles:
    #         if repeatdtFlag:
    #             pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * (-a) + (a/2.0), lat=np.random.rand(start_N_particles, 1) * (-b) + (b/2.0), time=simStart, repeatdt=delta(minutes=repeatRateMinutes), idgen=idgen, c_lib_register=c_lib_register)
    #             if pset_type != 'nodes':
    #                 psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * (-a) + (a/2.0), lat=np.random.rand(int(addParticleN), 1) * (-b) + (b/2.0), time=simStart, idgen=idgen, c_lib_register=c_lib_register)
    #                 pset.add(psetA)
    #             else:
    #                 lonlat_field = np.random.rand(int(addParticleN), 2)
    #                 lonlat_field *= np.array([a, b])
    #                 lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
    #                 lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
    #                 # time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
    #                 time_field = simStart
    #                 pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
    #                 pset.add(pdata)
    #             # lonlat_field = np.random.rand(int(addParticleN), 2)
    #             # lonlat_field *= np.array([a, b])
    #             # lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
    #             # lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
    #             # # time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
    #             # time_field = simStart
    #             # pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
    #             # pset.add(pdata)
    #         else:
    #             pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * (-a) + (a/2.0), lat=np.random.rand(Nparticle, 1) * (-b) + (b/2.0), time=simStart, idgen=idgen, c_lib_register=c_lib_register)
    #     else:
    #         if repeatdtFlag:
    #             pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * (-a) + (a/2.0), lat=np.random.rand(start_N_particles, 1) * (-b) + (b/2.0), time=simStart, repeatdt=delta(minutes=repeatRateMinutes), idgen=idgen, c_lib_register=c_lib_register)
    #             if pset_type != 'nodes':
    #                 psetA = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * (-a) + (a/2.0), lat=np.random.rand(int(addParticleN), 1) * (-b) + (b/2.0), time=simStart, idgen=idgen, c_lib_register=c_lib_register)
    #                 pset.add(psetA)
    #             else:
    #                 lonlat_field = np.random.rand(int(addParticleN), 2)
    #                 lonlat_field *= np.array([a, b])
    #                 lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
    #                 lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
    #                 # time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
    #                 time_field = simStart
    #                 pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
    #                 pset.add(pdata)
    #             # lonlat_field = np.random.rand(int(addParticleN), 2)
    #             # lonlat_field *= np.array([a, b])
    #             # lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
    #             # lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
    #             # # time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
    #             # time_field = simStart
    #             # pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
    #             # pset.add(pdata)
    #         else:
    #             pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * (-a) + (a/2.0), lat=np.random.rand(Nparticle, 1) * (-b) + (b/2.0), time=simStart, idgen=idgen, c_lib_register=c_lib_register)

    output_file = None
    out_fname = "benchmark_doublegyre"
    if args.write_out:
        if MPI and (MPI.COMM_WORLD.Get_size()>1):
            out_fname += "_MPI" + "_n{}".format(MPI.COMM_WORLD.Get_size())
            pfname += "_MPI" + "_n{}".format(MPI.COMM_WORLD.Get_size())
        else:
            out_fname += "_noMPI"
            pfname += "_noMPI"
        if periodicFlag:
            out_fname += "_p"
            pfname += '_p'
        out_fname += "_n"+str(Nparticle)
        pfname += "_n"+str(Nparticle)
        out_fname += '_%dd' % (time_in_days, )
        pfname += '_%dd' % (time_in_days, )
        if use_3D:
            out_fname += "_3D"
            pfname += "_3D"
        if backwardSimulation:
            out_fname += "_bwd"
            pfname += "_bwd"
        else:
            out_fname += "_fwd"
            pfname += "_fwd"
        if repeatdtFlag:
            out_fname += "_add"
            pfname += "_add"
        if agingParticles:
            out_fname += "_age"
            pfname += "_age"
        if with_GC:
            out_fname += "_wGC"
            pfname += "_wGC"
        else:
            out_fname += "_woGC"
            pfname += "_woGC"
        output_file = pset.ParticleFile(name=os.path.join(odir, out_fname+".nc"), outputdt=delta(hours=24))
    imageFileName = pfname + pfext

    delete_func = RenewParticle
    if args.delete_particle:
        delete_func=DeleteParticle
    postProcessFuncs = None
    callbackdt = None
    if with_GC:
        postProcessFuncs = [perIterGC, ]
        callbackdt = delta(hours=12)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank == 0:
            starttime = ostime.process_time()
    else:
        starttime = ostime.process_time()

    kernels = pset.Kernel(method[interp_mode] if not use_3D else AdvectionRK4_3D, delete_cfiles=True)
    if interp_mode=='bm':
        kernels += pset.Kernel(DiffusionUniformKh, delete_cfiles=True)
    kernels += pset.Kernel(periodicBC, delete_cfiles=True)
    if agingParticles:
        kernels += pset.Kernel(initialize, delete_cfiles=True)
        kernels += pset.Kernel(Age, delete_cfiles=True)

    if backwardSimulation:
        # ==== backward simulation ==== #
        if args.animate:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=callbackdt, moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=callbackdt)
    else:
        # ==== forward simulation ==== #
        if args.animate:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=callbackdt, moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=callbackdt)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank == 0:
            endtime = ostime.process_time()
    else:
        endtime = ostime.process_time()

    if args.write_out and not args.dryrun:
        output_file.close()

    if not args.dryrun:
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            # mpi_comm.Barrier()
            size_Npart = len(pset.nparticle_log)
            Npart = pset.nparticle_log.get_param(size_Npart-1)
            Npart = mpi_comm.reduce(Npart, op=MPI.SUM, root=0)
            if mpi_comm.Get_rank() == 0:
                if size_Npart > 0:
                    sys.stdout.write("final # particles: {}\n".format( Npart ))
                sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime-starttime))
                avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
                sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time*1000.0))
        else:
            size_Npart = len(pset.nparticle_log)
            Npart = pset.nparticle_log.get_param(size_Npart-1)
            if size_Npart > 0:
                sys.stdout.write("final # particles: {}\n".format( Npart ))
            sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
            avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
            sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

        if MPI:
            mpi_comm = MPI.COMM_WORLD
            Nparticles = mpi_comm.reduce(np.array(pset.nparticle_log.get_params()), op=MPI.SUM, root=0)
            Nmem = mpi_comm.reduce(np.array(pset.mem_log.get_params()), op=MPI.SUM, root=0)
            if mpi_comm.Get_rank() == 0:
                pset.plot_and_log(memory_used=Nmem, nparticles=Nparticles, target_N=target_N, imageFilePath=imageFileName, odir=odir, xlim_range=[0, 730], ylim_range=[0, 150])
        else:
            pset.plot_and_log(target_N=target_N, imageFilePath=imageFileName, odir=odir, xlim_range=[0, 730], ylim_range=[0, 150])

    del pset
    if idgen is not None:
        idgen.close()
        del idgen
    if c_lib_register is not None:
        c_lib_register.clear()
        del c_lib_register
