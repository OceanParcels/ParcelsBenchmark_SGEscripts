# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:31:22 2017

@author: nooteboom
"""
import itertools

from parcels import FieldSet, JITParticle, AdvectionRK4_3D, Field, Variable, StateCode, OperationCode, ErrorCode
from parcels import BenchmarkParticleSetSOA, BenchmarkParticleSetAOS, BenchmarkParticleSetNodes
from parcels import ParticleSetSOA, ParticleSetAOS, ParticleSetNodes
from parcels import GenerateID_Service, SequentialIdGenerator, LibraryRegisterC  # noqa

from argparse import ArgumentParser
from datetime import timedelta as delta
from datetime import datetime
import numpy as np
import math
from glob import glob
import sys
import pandas as pd

import os
import time as ostime
import fnmatch

# import dask
import gc

try:
    from mpi4py import MPI
except:
    MPI = None

import warnings
import xarray as xr
warnings.simplefilter("ignore", category=xr.SerializationWarning)

global_t_0 = 0
odir = ""
minlat = -78.694
maxlat = -0.005
minlon = -179.5
maxlon = -179.99

pset_modes = ['soa', 'aos', 'nodes']
pset_types_dry = {'soa': {'pset': ParticleSetSOA},  # , 'pfile': ParticleFileSOA, 'kernel': KernelSOA
                  'aos': {'pset': ParticleSetAOS},  # , 'pfile': ParticleFileAOS, 'kernel': KernelAOS
                  'nodes': {'pset': ParticleSetNodes}}  # , 'pfile': ParticleFileNodes, 'kernel': KernelNodes
pset_types = {'soa': {'pset': BenchmarkParticleSetSOA},
              'aos': {'pset': BenchmarkParticleSetAOS},
              'nodes': {'pset': BenchmarkParticleSetNodes}}

def set_nemo_fieldset(ufiles, vfiles, wfiles, tfiles, pfiles, dfiles, ifiles, bfile,
                      mesh_mask='/scratch/ckehl/experiments/palaeo-parcels/NEMOdata/domain/coordinates.nc',
                      chunk_level=0, periodicFlag=False, period_t_days=None):
    bfile_array = bfile
    if not isinstance(bfile_array, list):
        bfile_array = [bfile, ]
    filenames = {'U': {'lon': mesh_mask,
                       'lat': mesh_mask,
                       'depth': [ufiles[0]],
                       'data': ufiles},
                 'V': {'lon': mesh_mask,
                       'lat': mesh_mask,
                       'depth': [ufiles[0]],
                       'data': vfiles},
                 'W': {'lon': mesh_mask,
                       'lat': mesh_mask,
                       'depth': [ufiles[0]],
                       'data': wfiles},
                 'S': {'lon': mesh_mask,
                       'lat': mesh_mask,
                       'data': tfiles},
                 'T': {'lon': mesh_mask,
                       'lat': mesh_mask,
                       'data': tfiles},
                 'NO3': {'lon': mesh_mask,
                         'lat': mesh_mask,
                         'depth': [ufiles[0]],
                         # 'depth': [pfiles[0]],
                         'data': pfiles},
                 'ICE': {'lon': mesh_mask,
                         'lat': mesh_mask,
                         'data': ifiles},
                 'ICEPRES': {'lon': mesh_mask,
                             'lat': mesh_mask,
                             'data': ifiles},
                 'CO2': {'lon': mesh_mask,
                         'lat': mesh_mask,
                         'data': dfiles},
                 'PP': {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        # 'depth': [dfiles[0]],
                        'data': dfiles},
                 }
    if mesh_mask:
        filenames['mesh_mask'] = mesh_mask
    variables = {'U': 'uo',  # 3D
                 'V': 'vo',  # 3D
                 'W': 'wo',  # 3D
                 'S': 'sss',  # 2D - 'salin' = 3D
                 'T': 'sst',  # 2D - 'potemp' = 3D
                 # 'O2': 'OXY',
                 'NO3': 'DIN',  # 3D
                 'ICE': 'sit',  # 2D
                 'ICEPRES': 'ice_pres',  # 2D
                 'CO2': 'TCO2',  # 2D
                 'PP': 'TPP3',  # 3D
                 # 'B': 'Bathymetry',
                 }

    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'T': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'S': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'NO3': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},
                  'ICE': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'ICEPRES': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'CO2': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'PP': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},
                  }
    bfiles = {'lon': mesh_mask, 'lat': mesh_mask, 'data': bfile_array}
    bvariables = ('B', 'Bathymetry')
    bdimensions = {'lon': 'glamf', 'lat': 'gphif'}
    bchs = False

    nchs = None
    if chunk_level > 1:
        nchs = {
            'U':       {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('depthu', 80), 'time': ('time_counter', 1)},
            'V':       {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('depthv', 80), 'time': ('time_counter', 1)},
            'W':       {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('depthw', 80), 'time': ('time_counter', 1)},
            'T':       {'lon': ('x', 128), 'lat': ('y', 96), 'time': ('time_counter', 1)},
            'S':       {'lon': ('x', 128), 'lat': ('y', 96), 'time': ('time_counter', 1)},
            'NO3':     {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('deptht', 80), 'time': ('time_counter', 1)},
            'PP':      {'lon': ('x', 128), 'lat': ('y', 96), 'time': ('time_counter', 1)},
            'ICE':     {'lon': ('x', 128), 'lat': ('y', 96), 'time': ('time_counter', 1)},
            'ICEPRES': {'lon': ('x', 128), 'lat': ('y', 96), 'time': ('time_counter', 1)},
            'CO2':     {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('deptht', 80), 'time': ('time_counter', 1)}
        }
    elif chunk_level > 0:
        nchs = {
            'U':       'auto',
            'V':       'auto',
            'W':       'auto',
            'T':       'auto',
            'S':       'auto',
            'NO3':     'auto',
            'PP':      'auto',
            'ICE':     'auto',
            'ICEPRES': 'auto',
            'CO2':     'auto'
        }
    else:
        nchs = False
    # dask.config.set({'array.chunk-size': '16MiB'})

    if mesh_mask: # and isinstance(bfile, list) and len(bfile) > 0:
        if not periodicFlag:
            fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=True, chunksize=nchs)
            Bfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', chunksize=bchs)
        else:
            fieldset = FieldSet.from_nemo(filenames, variables, dimensions, time_periodic=delta(days=10*366 if period_t_days is not None else period_t_days), chunksize=nchs)
            Bfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', chunksize=bchs)
        fieldset.add_field(Bfield, 'B')
        fieldset.U.vmax = 10
        fieldset.V.vmax = 10
        fieldset.W.vmax = 10
        return fieldset
    else:
        filenames.pop('B')
        variables.pop('B')
        dimensions.pop('B')
        if not periodicFlag:
            fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True, chunksize=nchs)
        else:
            fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, time_periodic=delta(days=10*366 if period_t_days is not None else period_t_days), chunksize=nchs)
        fieldset.U.vmax = 10
        fieldset.V.vmax = 10
        fieldset.W.vmax = 10
        return fieldset
        

def periodicBC(particle, fieldSet, time):
    if particle.lon > 180.0:
        particle.lon -= 360.0
    if particle.lon < -180.0:
        particle.lon += 360.0

def Sink(particle, fieldset, time):
    if(particle.depth > fieldset.dwellingdepth):
        particle.depth = particle.depth + fieldset.sinkspeed * particle.dt
    elif(particle.depth <= fieldset.dwellingdepth and particle.depth>1):
        particle.depth = fieldset.surface
        # == Comment: bad idea to do 'time+dt' here cause those data are (likely) not loaded right now == #
        # particle.temp = fieldset.T[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        # particle.salin = fieldset.S[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        # particle.PP = fieldset.PP[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        # particle.NO3 = fieldset.NO3[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        # particle.ICE = fieldset.ICE[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        # particle.ICEPRES = fieldset.ICEPRES[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        # particle.CO2 = fieldset.CO2[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.temp = fieldset.T[time, fieldset.surface, particle.lat, particle.lon]
        particle.salin = fieldset.S[time, fieldset.surface, particle.lat, particle.lon]
        particle.PP = fieldset.PP[time, fieldset.surface, particle.lat, particle.lon]
        particle.NO3 = fieldset.NO3[time, fieldset.surface, particle.lat, particle.lon]
        particle.ICE = fieldset.ICE[time, fieldset.surface, particle.lat, particle.lon]
        particle.ICEPRES = fieldset.ICEPRES[time, fieldset.surface, particle.lat, particle.lon]
        particle.CO2 = fieldset.CO2[time, fieldset.surface, particle.lat, particle.lon]
        particle.delete()

def Age(particle, fieldset, time):
    if particle.state == StateCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    # if particle.age > fieldset.maxage:
    #     particle.delete()

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def perIterGC():
    gc.collect()

def initials(particle, fieldset, time):
    if particle.age==0.:
        particle.depth = fieldset.B[time, fieldset.surface, particle.lat, particle.lon]
        if(particle.depth  > 5800.):
            particle.age = (particle.depth - 5799.)*fieldset.sinkspeed
            particle.depth = 5799.        
        particle.lon0 = particle.lon
        particle.lat0 = particle.lat
        particle.depth0 = particle.depth

class DinoParticle(JITParticle):
    temp = Variable('temp', dtype=np.float32, initial=np.nan)
    age = Variable('age', dtype=np.float32, initial=0.)
    salin = Variable('salin', dtype=np.float32, initial=np.nan)
    lon0 = Variable('lon0', dtype=np.float32, initial=0.)
    lat0 = Variable('lat0', dtype=np.float32, initial=0.)
    depth0 = Variable('depth0',dtype=np.float32, initial=0.)
    PP = Variable('PP',dtype=np.float32, initial=np.nan)
    NO3 = Variable('NO3',dtype=np.float32, initial=np.nan)
    ICE = Variable('ICE',dtype=np.float32, initial=np.nan)
    ICEPRES = Variable('ICEPRES',dtype=np.float32, initial=np.nan)
    CO2 = Variable('CO2',dtype=np.float32, initial=np.nan)


if __name__ == "__main__":
    parser = ArgumentParser(description="Example of particle advection for the palaeo-plankton case")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="benchmark_palaeo.png", help="image file name of the plot")
    parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="-1", help="number of particles (per 3 days of simulation) to generate and advect (default: 66)")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-sp", "--sinking_speed", dest="sp", type=float, default=11.0, help="set the simulation sinking speed in [m/day] (default: 11.0)")
    parser.add_argument("-dd", "--dwelling_depth", dest="dd", type=float, default=10.0, help="set the dwelling depth (i.e. ocean surface depth) in [m] (default: 10.0)")
    parser.add_argument("-w", "--writeout", dest="write_out", action='store_true', default=False, help="write data in outfile")
    # parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=365, help="runtime in days (default: 365)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=str, default="1*366", help="runtime in days (default: 1*366)")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    parser.add_argument("-pr", "--profiling", dest="profiling", action='store_true', default=False, help="tells that the profiling of the script is activates")
    parser.add_argument("-tp", "--type", dest="pset_type", default="SoA", help="particle set type = [SOA, AOS, Nodes]")
    parser.add_argument("-chs", "--chunksize", dest="chs", type=int, default=0, help="defines the chunksize level: 0=None, 1='auto', 2=fine tuned; default: 0")
    parser.add_argument("--dry", dest="dryrun", action="store_true", default=False, help="Start dry run (no benchmarking and its classes")
    args = parser.parse_args()

    pset_type = str(args.pset_type).lower()
    assert pset_type in pset_types
    ParticleSet = pset_types[pset_type]['pset']
    if args.dryrun:
        ParticleSet = pset_types_dry[pset_type]['pset']

    sp = args.sp # The sinkspeed m/day
    dd = args.dd  # The dwelling depth
    imageFileName=args.imageFileName
    periodicFlag=args.periodic
    with_GC = args.useGC
    time_in_days = int(float(eval(args.time_in_days)))
    time_in_years = int(float(time_in_days)/365.0)
    Nparticle = int(float(eval(args.nparticles)))
    if Nparticle > 1:
        sx = int(math.sqrt(Nparticle))
        sy = sx
        Nparticle = sx * sy

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

    branch = "soa_benchmark"
    computer_env = "local/unspecified"
    scenario = "palaeo-parcels"
    headdir = ""
    odir = ""
    dirread_pal = ""
    datahead = ""
    dirread_top = ""
    dirread_top_bgc = ""
    basefile_str = {}
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        # headdir = "/scratch/{}/experiments/palaeo-parcels".format(os.environ['USER'])
        headdir = "/scratch/{}/experiments/palaeo-parcels".format("ckehl")
        odir = os.path.join(headdir, "BENCHres", str(args.pset_type))
        dirread_pal = os.path.join(headdir, 'NEMOdata')
        datahead = "/data/oceanparcels/input_data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        dirread_top_bgc = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        basefile_str = {
            'U': 'ORCA0083-N06_2000????d05U.nc',
            'V': 'ORCA0083-N06_2000????d05V.nc',
            'W': 'ORCA0083-N06_2000????d05W.nc',
            'T': 'ORCA0083-N06_2000????d05T.nc',
            'P': 'ORCA0083-N06_2000????d05P.nc',
            'D': 'ORCA0083-N06_2000????d05D.nc',
            'I': 'ORCA0083-N06_2000????d05I.nc',
            'B': 'bathymetry_ORCA12_V3.3.nc'
        }
        computer_env = "Gemini"
    elif os.uname()[1] in ["lorenz.science.uu.nl", ] or fnmatch.fnmatchcase(os.uname()[1], "node*"):  # Lorenz
        CARTESIUS_SCRATCH_USERNAME = 'ckehl'
        headdir = "/storage/shared/oceanparcels/output_data/data_{}/experiments/palaeo-parcels".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "BENCHres", str(args.pset_type))
        dirread_pal = os.path.join(headdir, 'NEMOdata')
        datahead = "/storage/shared/oceanparcels/input_data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        dirread_top_bgc = os.path.join(datahead, 'NEMO-MEDUSA_BGC/ORCA0083-N006/')
        basefile_str = {
            'U': 'ORCA0083-N06_2004????d05U.nc',
            'V': 'ORCA0083-N06_2004????d05V.nc',
            'W': 'ORCA0083-N06_2004????d05W.nc',
            'T': 'ORCA0083-N06_2004????d05T.nc',
            'P': 'ORCA0083-N06_2004????d05P.nc',
            'D': 'ORCA0083-N06_2004????d05D.nc',
            'I': 'ORCA0083-N06_2004????d05I.nc',
            'B': 'bathymetry_ORCA12_V3.3.nc'
        }
        computer_env = "Lorenz"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch/shared/{}/experiments/palaeo-parcels".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "BENCHres", str(args.pset_type))
        dirread_pal = os.path.join(headdir, 'NEMOdata')
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        dirread_top_bgc = os.path.join(datahead, 'NEMO-MEDUSA_BGC/ORCA0083-N006/')
        basefile_str = {
            'U': 'ORCA0083-N06_2000????d05U.nc',
            'V': 'ORCA0083-N06_2000????d05V.nc',
            'W': 'ORCA0083-N06_2000????d05W.nc',
            'T': 'ORCA0083-N06_2000????d05T.nc',
            'P': 'ORCA0083-N06_2000????d05P.nc',
            'D': 'ORCA0083-N06_2000????d05D.nc',
            'I': 'ORCA0083-N06_2000????d05I.nc',
            'B': 'bathymetry_ORCA12_V3.3.nc'
        }
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "int*.snellius.*") or fnmatch.fnmatchcase(os.uname()[1], "fcn*") or fnmatch.fnmatchcase(os.uname()[1], "tcn*") or fnmatch.fnmatchcase(os.uname()[1], "gcn*") or fnmatch.fnmatchcase(os.uname()[1], "hcn*"):  # Snellius
        SNELLIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch-shared/{}/experiments/palaeo-parcels".format(SNELLIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "BENCHres", str(args.pset_type))
        dirread_pal = os.path.join(headdir, 'NEMOdata')
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        dirread_top_bgc = os.path.join(datahead, 'NEMO-MEDUSA_BGC/ORCA0083-N006/')
        basefile_str = {
            'U': 'ORCA0083-N06_2000????d05U.nc',
            'V': 'ORCA0083-N06_2000????d05V.nc',
            'W': 'ORCA0083-N06_2000????d05W.nc',
            'T': 'ORCA0083-N06_2000????d05T.nc',
            'P': 'ORCA0083-N06_2000????d05P.nc',
            'D': 'ORCA0083-N06_2000????d05D.nc',
            'I': 'ORCA0083-N06_2000????d05I.nc',
            'B': 'bathymetry_ORCA12_V3.3.nc'
        }
        computer_env = "Snellius"
    else:
        headdir = "/var/scratch/nooteboom"
        odir = os.path.join(headdir, "BENCHres", str(args.pset_type))
        dirread_pal = headdir
        datahead = "/data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        dirread_top_bgc = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        basefile_str = {
            'U': 'ORCA0083-N06_2000????d05U.nc',
            'V': 'ORCA0083-N06_2000????d05V.nc',
            'W': 'ORCA0083-N06_2000????d05W.nc',
            'T': 'ORCA0083-N06_2000????d05T.nc',
            'P': 'ORCA0083-N06_2000????d05P.nc',
            'D': 'ORCA0083-N06_2000????d05D.nc',
            'I': 'ORCA0083-N06_2000????d05I.nc',
            'B': 'bathymetry_ORCA12_V3.3.nc'
        }

    # print("running {} on {} (uname: {}) - branch '{}' - headdir: {}; odir: {} - argv: {}".format(scenario, computer_env, os.uname()[1], branch, headdir, odir, sys.argv[1:]))
    # dirread_pal = '/projects/0/palaeo-parcels/NEMOdata/'

    if os.path.sep in imageFileName:
        head_dir = os.path.dirname(imageFileName)
        if head_dir[0] == os.path.sep:
            odir = head_dir
        else:
            odir = os.path.join(odir, head_dir)
            imageFileName = os.path.split(imageFileName)[1]
    pfname, pfext = os.path.splitext(imageFileName)


    outfile = 'grid_dd' + str(int(dd))
    outfile += '_sp' + str(int(sp))
    if periodicFlag:
        outfile += '_p'
        pfname += '_p'
    if args.write_out:
        pfname += '_w'
    if time_in_years != 1:
        outfile += '_%dy' % (time_in_years, )
        pfname += '_%dy' % (time_in_years, )
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_size = mpi_comm.Get_size()
        outfile += '_n' + str(mpi_size)
        pfname += '_n' + str(mpi_size)
    if args.profiling:
        outfile += '_prof'
        pfname += 'prof'
    if with_GC:
        outfile += '_wGC'
        pfname += '_wGC'
    else:
        outfile += '_woGC'
        pfname += '_woGC'
    outfile += "_N"+str(Nparticle)
    pfname += "_N"+str(Nparticle)
    outfile += '_%s' % ('nochk' if args.chs==0 else ('achk' if args.chs==1 else 'dchk'))
    pfname += '_%s' % ('nochk' if args.chs==0 else ('achk' if args.chs==1 else 'dchk'))
    imageFileName = pfname + pfext
    dirwrite = os.path.join(odir, "sp%d_dd%d" % (int(sp),int(dd)))
    if not os.path.exists(dirwrite):
        os.mkdir(dirwrite)

    # ==== TODO: here, make pre-runs to first sample the depth, then execute the simulation ==== #
    timesz = np.array([datetime(2000, 12, 25) - delta(days=x) for x in range(0,int(365),3)])
    latlonstruct = pd.read_csv(os.path.join(headdir,"TF_locationsSurfaceSamples_forPeter.csv"))
    latsz = np.array(latlonstruct.Latitude.tolist())
    lonsz = np.array(latlonstruct.Longitude.tolist())
    numlocs = np.logical_and(latsz<1000, lonsz<1000)
    latsz = latsz[numlocs]
    lonsz = lonsz[numlocs]
    assert ~(np.isnan(latsz)).any(), 'locations should not contain any NaN values'
    stored_Nsamples = latsz.shape[0]
    if Nparticle > 1:
        Nsamples = int(Nparticle/timesz.shape[0] * 1.5)
        indices = np.random.randint(0, stored_Nsamples, Nsamples)
        lonloc = np.array(lonsz)
        latloc = np.array(latsz)
        lonsz = lonloc[indices]
        latsz = latloc[indices]

    dep = dd * np.ones(latsz.shape)

    times = np.empty(shape=(0))
    depths = np.empty(shape=(0))
    lons = np.empty(shape=(0))
    lats = np.empty(shape=(0))
    for i in range(len(timesz)):
        lons = np.append(lons,lonsz)
        lats = np.append(lats, latsz)
        depths = np.append(depths, np.zeros(len(lonsz), dtype=np.float32))
        times = np.append(times, np.full(len(lonsz),timesz[i]))

    print("running {} on {} (uname: {}) - branch '{}' - (target) N: {} - argv: {}".format(scenario, computer_env, os.uname()[1], branch, lons.shape[0], sys.argv[1:]))


    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            # global_t_0 = MPI.Wtime()
            global_t_0 = ostime.process_time()
    else:
        # global_t_0 = ostime.time()
        global_t_0 = ostime.process_time()

    ufiles = sorted(glob(os.path.join(dirread_top, 'means', basefile_str['U'])))
    vfiles = sorted(glob(os.path.join(dirread_top, 'means', basefile_str['V'])))
    wfiles = sorted(glob(os.path.join(dirread_top, 'means', basefile_str['W'])))
    tfiles = sorted(glob(os.path.join(dirread_top, 'means', basefile_str['T'])))
    pfiles = sorted(glob(os.path.join(dirread_top_bgc, 'means', basefile_str['P'])))
    dfiles = sorted(glob(os.path.join(dirread_top_bgc, 'means', basefile_str['D'])))
    ifiles = sorted(glob(os.path.join(dirread_top, 'means', basefile_str['I'])))
    bfile = os.path.join(dirread_top, 'domain', basefile_str['B'])

    fieldset = set_nemo_fieldset(ufiles, vfiles, wfiles, tfiles, pfiles, dfiles, ifiles, bfile, os.path.join(dirread_pal, "domain/coordinates.nc"), chunk_level=args.chs, periodicFlag=periodicFlag, period_t_days=time_in_days)
    fieldset.add_periodic_halo(zonal=True) 
    fieldset.add_constant('dwellingdepth', np.float(dd))
    fieldset.add_constant('sinkspeed', sp/86400.)
    fieldset.add_constant('maxage', 300000.*86400)
    fieldset.add_constant('surface', 2.5)

    print("|lon| = {}; |lat| = {}; |depth| = {}, |times| = {}, |grids| = {}".format(lonsz.shape[0], latsz.shape[0], dep.shape[0], times.shape[0], fieldset.gridset.size))

    # ==== Set min/max depths in the fieldset ==== #
    fs_depths = fieldset.U.depth

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=DinoParticle, lon=lons.tolist(), lat=lats.tolist(), depth=depths.tolist(), time=times, idgen=idgen, c_lib_register=c_lib_register)

    """ Kernel + Execution"""
    postProcessFuncs = None
    callbackdt = None
    if with_GC:
        postProcessFuncs = [perIterGC, ]
        callbackdt = delta(days=30)
    output_fpath = None
    pfile = None
    if args.write_out and not args.dryrun:
        output_fpath = os.path.join(dirwrite, outfile)
        pfile = pset.ParticleFile(output_fpath, convert_at_end=True, write_ondelete=True)
    kernels = pset.Kernel(initials) + Sink + Age + pset.Kernel(AdvectionRK4_3D) + Age

    starttime = 0
    endtime = 0
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank == 0:
            # starttime = ostime.time()
            # starttime = MPI.Wtime()
            starttime = ostime.process_time()
    else:
        # starttime = ostime.time()
        starttime = ostime.process_time()

    # pset.execute(kernels, runtime=delta(days=365*9), dt=delta(minutes=-20), output_file=pfile, verbose_progress=False,
    # recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, postIterationCallbacks=postProcessFuncs)
    # postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12)
    # pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(hours=-12), output_file=pfile, verbose_progress=False, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, postIterationCallbacks=postProcessFuncs, callbackdt=callbackdt)
    pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(hours=-12), output_file=pfile,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
                 postIterationCallbacks=postProcessFuncs, callbackdt=callbackdt)
    
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank == 0:
            # endtime = ostime.time()
            # endtime = MPI.Wtime()
            endtime = ostime.process_time()
    else:
        #endtime = ostime.time()
        endtime = ostime.process_time()

    if args.write_out:
        pfile.close()


    if not args.dryrun:
        if MPI:
            mpi_comm = MPI.COMM_WORLD
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
            # mpi_comm.Barrier()
            Nparticles = mpi_comm.reduce(np.array(pset.nparticle_log.get_params()), op=MPI.SUM, root=0)
            Nmem = mpi_comm.reduce(np.array(pset.mem_log.get_params()), op=MPI.SUM, root=0)
            if mpi_comm.Get_rank() == 0:
                pset.plot_and_log(memory_used=Nmem, nparticles=Nparticles, target_N=1, imageFilePath=imageFileName, odir=odir)
        else:
            pset.plot_and_log(target_N=1, imageFilePath=imageFileName, odir=odir)

    del pset
    if idgen is not None:
        idgen.close()
        del idgen
    if c_lib_register is not None:
        c_lib_register.clear()
        del c_lib_register

    print('Execution finished')
