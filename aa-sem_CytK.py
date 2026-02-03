import os
import argparse
import pickle
from pathlib import Path
from glob import glob
from natsort import natsorted

import numpy as np
import MDAnalysis as mda
from scipy.spatial import KDTree
from scipy.interpolate import interp1d, RegularGridInterpolator
import subprocess
import time
import psutil
from resource import getrusage, RUSAGE_SELF
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(f'%(asctime)s %(name)s: %(levelname)s: %(message)s')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

from glob import glob
from natsort import natsorted

from fenics import BoxMesh, cells, MeshFunction, refine, Point, Mesh, XDMFFile
from sem_new import AbstractSEM, shared_ndarray, voxel_distance_from_nearest_bead_adnan

parser = argparse.ArgumentParser(prog=__file__,
				 description='Run SEM.')
parser.add_argument('-n','--num-procs', type=int,
		    default=1,
                    help='Number of parallel threads')
parser.add_argument('-pqr','--pqr_file',type=str,
                    help='PQR input file')
parser.add_argument('-dcd','--dcd_file',type=str,
                    help='DCD input file')
parser.add_argument('--outname',type=str,
                    help='Output directory prefix')
parser.add_argument('--length',type=float,default=45,
                   help='Length in XY dimension')
parser.add_argument('--stride',type=int,default=1,
                    help='Stride for DCD file')
#parser.add_argument('-box_x','--box_size',type=float,default=50.0,
#                    help='Coordinate of x_max for SEM box')
# parser.add_argument('-s','--system-name', type=str,
#                     choices=('cyl','slit','hourglass'),
# 		    default='cyl',
#                     help='Which system?')
# parser.add_argument('-i','--system-index', type=int,
# 		    default=0,
#                     help='Index of selected system')
# parser.add_argument('-c','--concentration', type=float,
# 		    default=1000,
#                     help='Bulk KCl concentration (mM)')
parser.add_argument('--voltage', type=float,
		    default=100,
                    help='Applied bias (mV)')

args = parser.parse_args()

system_name = args.outname
sem_sel = "protein"
_peak_memory=0
#Memory logging
def log_memory(stage):
        global _peak_memory
        timestamp = time.strftime("%H:%M:%S")
        mem = psutil.virtual_memory()
        process = psutil.Process()
        rss_gb = process.memory_info().rss/1024**3
    # Track peak memory
        if rss_gb > _peak_memory:
              _peak_memory = rss_gb
    
        print(f"[{timestamp} {stage:25s} |"
          f"System: {mem.used/1024**3:5.1f}GB used, {mem.cached/1024**3:5.1f}GB cached |"
          f"Process: {rss_gb:5.1f}GB RSS | Peak: {_peak_memory:5.1f}GB]")

class ConductivityMap():

    """ Custom conductivity map from Sarthak """
    def __init__(self):
        self.bulk_conductivity = 20.0840 # S/m
#        self.bulk_conductivity =16.5# S/m 1M KCL, MD simulation
#        self.bulk_conductivity =10.5# S/m 1M KCL, Experimental 
        minr = 0.13
        self.cutoff = maxr = 0.41
        self.slope = 1.0/(maxr-minr)
        self.intercept = -minr*self.slope

    def __call__(self, dists):
        # points on line (min,0) (max,1)
        result = self.slope*dists + self.intercept
        result[result<0] = 0.0000001
        result[result>1] = 1.0
        return result * self.bulk_conductivity

def _get_grids():
    """ Since we are not analyzing a trajectory inthis script, we'll
    just find the conductance grid here """
    resolution = 1
    z_shift =44
    xy_max = args.length/2
    xy_min = -xy_max
    z_min = 1.2*xy_min+z_shift-54
    z_max = 1.2*xy_max+z_shift+54
    min_ = np.array((xy_min, xy_min,z_min))
    max_ = np.array((xy_max, xy_max, z_max))
    xyz = tuple(0.1*np.arange(a,b,resolution) for a,b in zip(min_,max_)) # nm
    X,Y,Z = np.meshgrid( *xyz, indexing='ij' )
    R2 = (X**2+Y**2)
    bulk = np.ones([len(a) for a in xyz])
    bulk[(Z < 1.7) & (Z > -2.5) & (R2 > 1.2)] = 0.00001
    return bulk, xyz
    
class MySEM(AbstractSEM):

    def __init__(self,
                 system_name,
                 domain, voltage, conductivity_model,
                 # dcd_prefix,
                 # far_from_pore,
                 **kwargs):

        self.domain = domain
        self.voltage = voltage
        self.system_name = system_name
        
        for k,v in kwargs.items():
            setattr(self,k,v)

        self.dist_to_cond = conductivity_model
        self.bulk_conductivity = conductivity_model.bulk_conductivity
        
        # self.stride = stride
        # self.prefix = dcd_prefix

        # dcds = natsorted( glob(f'{self.prefix}/output/run_in-cont??.dcd') )
        # self.universe = u = mda.Universe(f'{self.prefix}/run_in.psf', f'{self.prefix}/output/run_in.dcd', *dcds )
        
        super().__init__()      # sets up logger for cls.info(...), etc

    @classmethod
    def get_base_conductance(cls):
        ## Find points where there is a gradient in base_conductance
        try:
            base_conductance = cls.base_conductance
        except:
            base_conductance = cls.base_conductance_shm.get_array() #.astype(np.half)

            xyz = [np.linspace(r0,r1,int((r1-r0)//dr)) for r0,r1,dr in cls.domain]
            centers = [0.5*(a[1:]+a[:-1]) for a in xyz]
            X,Y,Z = np.meshgrid(*centers, indexing='ij')
            Rsq = X**2+Y**2
            base_conductance = np.ones(X.shape) * cls.dist_to_cond.bulk_conductivity
            sl = (Rsq > cls.pore_radius**2) & (np.abs(Z) < cls.membrane_thickness/2)
            base_conductance[sl] = 0
            cls.base_conductance = base_conductance
            cls.base_conductance_edges = xyz
        return base_conductance

    @classmethod
    def generate_base_mesh(cls):
        desired_finest_mesh_resolution = 2.5 # nm
        desired_finest_mesh_resolution = 0.2 # nm
        gradient_resolution = 2*desired_finest_mesh_resolution

        cls.desired_finest_mesh_resolution = desired_finest_mesh_resolution
        cls.intial_mesh_points = initial_points = 32

        ## Find points where there is a gradient in base_conductance
        try:
            base_conductance = cls.base_conductance
        except:
            base_conductance = cls.base_conductance_shm.get_array() #.astype(np.half)
            cls.base_conductance = base_conductance

        cls.info('generate_base_mesh: Generating low resolution mesh')
        ## First create a low resolution mesh that will be further refined
        x,y,z = (cls.x,cls.y,cls.z)
                        
        cls.debug( [(a[0]+10*np.abs(np.spacing(a[0])),a[-1]-10*np.abs(np.spacing(a[-1])))
                  for a in (x,y,z)] )

        cls.info('generate_base_mesh: Loading base conductance and creating meshgrid')

        cls.info(f'Memory used: {getrusage(RUSAGE_SELF).ru_maxrss}')

        ## Find points where there is a gradient in base_conductance        
        cls.debug(f'base_cond hash: {hash((base_conductance.shape,base_conductance.mean()))}')
            
        cls.trace('generate_base_mesh: Here')
        xyz = cls.base_conductance_edges

        subsample = [int(np.ceil( gradient_resolution / (a[1]-a[0]) )) for a in xyz]
        cls.trace(f'generate_base_mesh: subsampling gradient with spacing {subsample}')
        sl = tuple(slice(None,None,s) for s in subsample)
        # dsl = tuple(slice(None,-1,s) for s in subsample)
        sls = tuple( tuple(slice(None,None if i == j else -1,s)
                           for j,s in enumerate(subsample))
                     for i in range(3) )

        cls.trace(f'Memory used: {getrusage(RUSAGE_SELF).ru_maxrss}')

        # _xyz = [a[:-1][_sl].astype(np.half) for a,_sl in zip(xyz,sl)]
        _xyz = [a[:-1][_sl] for a,_sl in zip(xyz,sl)]
        cls.trace(f'generate_base_mesh: Here3 {[a.shape for a in _xyz]}')

        X,Y,Z = np.meshgrid(*[a[:-1][_sl] for a,_sl in zip(xyz,sl)], indexing='ij', copy=False)

        cls.info(f"generate_base_mesh: Finding high gradient points")
        gradient = np.max([np.abs(np.diff(base_conductance,axis=i)[_sl])
                           for i,_sl in enumerate(sls)],axis=0)
        gradient = gradient/cls.bulk_conductivity
        cls.info(f"generate_base_mesh: Found high gradient points")

        # cls.debug(f"generate_base_mesh: gradiant,X,Y,Z shapes: {[a.shape for a in (gradient,X,Y,Z)]}")
        sl = (gradient > 0.02)
        high_gradient_points = [r for r in zip(X[sl],Y[sl],Z[sl])]
        cls.info(f"generate_base_mesh: Building KDTree of {len(high_gradient_points)} high gradient points")
        tree = KDTree( high_gradient_points )
        cls.trace(f'Memory used: {getrusage(RUSAGE_SELF).ru_maxrss}')

        cls.info(f"generate_base_mesh: Finding high gradient plane")

        cls.debug(f"{_xyz[2]}")
        R_sq = (X**2+Y**2)
        high_grad_zs = np.unique( Z[sl & (R_sq > cls.far_from_pore**2) & (R_sq < 4*cls.far_from_pore**2)] )
        # (X < xyz[0][10*subsample[0]]) & (Y < xyz[1][10*subsample[1]])] )

        cls.debug(f"generate_base_mesh: {high_grad_zs}")
        cls.info(f"generate_base_mesh: Found {len(high_grad_zs)}/{len(_xyz[2])} high gradient points")

        ## Assign some stuff
        cls.high_grad_zs = high_grad_zs
        
        _cache_filename = f'__cache_basemesh_{cls.prefix.replace("../10-coarse-grained-","").replace("/","_")}.xdmf'

        if not Path(_cache_filename).exists():

            mesh = BoxMesh( Point(np.array([a[0]+10*np.abs(np.spacing(a[0]))  for a in (x,y,z)])),
                            Point(np.array([a[-1]-10*np.abs(np.spacing(a[-1])) for a in (x,y,z)])),
                            *[initial_points for i in range(3)] )

            def _get_refinement_steps():
                spacing0 = np.max([a[-1]-a[0] for a in xyz])/initial_points
                spacing1 = desired_finest_mesh_resolution
                refinement_steps = np.log(spacing0/spacing1) / np.log(2)
                return int(np.ceil(refinement_steps))
            refinement_steps = _get_refinement_steps()

            for step in range(refinement_steps):
                cls.info(f'generate_base_mesh: Beginning refinement step {step+1}/{refinement_steps}')
                markers = MeshFunction('bool', mesh, mesh.topology().dim(), False)

                cls.trace(f'generate_base_mesh: Finding mesh cell positions')

                ## Get each cell's approximate center
                verts = [np.array(cell.get_vertex_coordinates()).reshape(-1,3)
                              for cell in cells(mesh)]
                r = np.array([v.mean(axis=0) for v in verts])
                # cls.trace(f'generate_base_mesh: zmin/zmax: {(np.min(r[:,2]),np.max(r[:,2]))}')
                cls.trace(f'generate_base_mesh: Found {len(r)} cell positions')

                ## Get a cutoff that accounts roughly for each cell's size
                max_vert_dists = np.sqrt(np.array([ np.max( ((v-r[i,:])**2).sum(axis=-1) )
                                                    for i,v in enumerate(verts) ]))
                cutoffs = np.array([v * 1.0 + 0.5*gradient_resolution for v in max_vert_dists])
                cutoff = np.max(cutoffs)

                # part_cutoffs = np.array([v * 1.0 + 0.5*cls.dist_to_cond.cutoff for v in max_vert_dists])
                # part_cutoff = np.max(part_cutoffs)

                cls.info(f'generate_base_mesh: min/max cell sizes: {np.min(max_vert_dists):0.2f}/{np.max(max_vert_dists):0.2f}')

                cls.trace(f'generate_base_mesh: Searching tree for cells within cutoff {cutoff:0.2f}')
                d,idx = tree.query( r, distance_upper_bound=cutoff )

                # part_d,idx = part_tree.query( r, distance_upper_bound=part_cutoff )

                cls.trace(f'generate_base_mesh: Determining which cells to refine') # steps until next cls.info are quite fasto
                assert(d.shape == cutoffs.shape )
                cond1 = (d < cutoffs) # cells near conductance grid gradient
                # cond2 = (part_d < part_cutoffs) # cells near protein positions

                ## cond2: large cells near barrier in plane
                sl = (max_vert_dists > 2)
                cond2 = np.zeros(d.shape, dtype=bool)
                cond2[sl] = sum([np.abs(r[sl][:,2]-z) < cutoffs[sl] for z in high_grad_zs],
                                cond2[sl])
                
                cls.trace(f'generate_base_mesh: Marking cells for refinement') # this step is fast
                for cell, mark in zip( cells(mesh), (cond1 | cond2) ):
                    markers[cell] = mark

                cls.info(f'generate_base_mesh: Refining {(cond1|cond2).sum()}/{len(r)} mesh cells (cond1/cond2: {cond1.sum()}/{cond2.sum()})')
                mesh = refine(mesh,markers)
                
            with XDMFFile(_cache_filename) as fh:
                fh.write(mesh)
        else:
            mesh = Mesh()
            with XDMFFile(_cache_filename) as fh:
                fh.read(mesh)
            
        return mesh

    @classmethod
    def generate_mesh_highres(cls):
        cls.generate_base_mesh()
        desired_finest_mesh_resolution = cls.desired_finest_mesh_resolution
        initial_points = cls.initial_mesh_points

        base_conductance = cls.base_conductance
        u = cls.universe

        cls.info('generate_mesh: Generating low resolution mesh')
        ## First create a low resolution mesh that will be further refined
        x,y,z = (cls.x,cls.y,cls.z)
            
            
        cls.debug( [(a[0]+10*np.abs(np.spacing(a[0])),a[-1]-10*np.abs(np.spacing(a[-1])))
                  for a in (x,y,z)] )

        cls.info('generate_mesh: Loading base conductance and creating meshgrid')

        cls.info(f'Memory used: {getrusage(RUSAGE_SELF).ru_maxrss}')

        ## Find points where there is a gradient in base_conductance        
        cls.debug(f'base_cond hash: {hash((base_conductance.shape,base_conductance.mean()))}')
            
        cls.trace('generate_mesh: Here')
        xyz = cls.base_conductance_edges

        subsample = [int(np.ceil( gradient_resolution / (a[1]-a[0]) )) for a in xyz]
        cls.trace(f'generate_mesh: subsampling gradient with spacing {subsample}')
        sl = tuple(slice(None,None,s) for s in subsample)
        # dsl = tuple(slice(None,-1,s) for s in subsample)
        sls = tuple( tuple(slice(None,None if i == j else -1,s)
                           for j,s in enumerate(subsample))
                     for i in range(3) )

        cls.trace(f'Memory used: {getrusage(RUSAGE_SELF).ru_maxrss}')

        # _xyz = [a[:-1][_sl].astype(np.half) for a,_sl in zip(xyz,sl)]
        _xyz = [a[:-1][_sl] for a,_sl in zip(xyz,sl)]
        cls.trace(f'generate_mesh: Here3 {[a.shape for a in _xyz]}')

        X,Y,Z = np.meshgrid(*[a[:-1][_sl] for a,_sl in zip(xyz,sl)], indexing='ij', copy=False)

        cls.info(f"generate_mesh: Finding high gradient points")
        gradient = np.max([np.abs(np.diff(base_conductance,axis=i)[_sl])
                           for i,_sl in enumerate(sls)],axis=0)
        gradient = gradient/cls.bulk_conductivity
        cls.info(f"generate_mesh: Found high gradient points")

        # cls.debug(f"generate_mesh: gradiant,X,Y,Z shapes: {[a.shape for a in (gradient,X,Y,Z)]}")
        sl = (gradient > 0.02)
        high_gradient_points = [r for r in zip(X[sl],Y[sl],Z[sl])]
        cls.info(f"generate_mesh: Building KDTree of {len(high_gradient_points)} high gradient points")
        tree = KDTree( high_gradient_points )
        cls.trace(f'Memory used: {getrusage(RUSAGE_SELF).ru_maxrss}')

        cls.info(f"generate_mesh: Finding high gradient plane")

        cls.debug(f"{_xyz[2]}")
        R_sq = (X**2+Y**2)
        high_grad_zs = np.unique( Z[sl & (R_sq > cls.far_from_pore**2) & (R_sq < 4*cls.far_from_pore**2)] )
        # (X < xyz[0][10*subsample[0]]) & (Y < xyz[1][10*subsample[1]])] )

        cls.debug(f"generate_mesh: {high_grad_zs}")
        cls.info(f"generate_mesh: Found {len(high_grad_zs)}/{len(_xyz[2])} high gradient points")

        ## Assign some stuff
        cls.high_grad_zs = high_grad_zs
        

        _cache_filename = f'__cache_mesh_{cls.prefix.replace("../10-coarse-grained-","").replace("/","_")}.xdmf'

        if not Path(_cache_filename).exists():

            mesh = BoxMesh( Point(np.array([a[0]+10*np.abs(np.spacing(a[0]))  for a in (x,y,z)])),
                            Point(np.array([a[-1]-10*np.abs(np.spacing(a[-1])) for a in (x,y,z)])),
                            *[initial_points for i in range(3)] )

            
            ## Get kdtree encoding all protein coordinates
            cls.info(f"generate_mesh: Loading particle coordinate for KDTree")

            cls.info(f'Memory used: {getrusage(RUSAGE_SELF).ru_maxrss}')

            stride = cls.stride
            sel = u.select_atoms(sem_sel)
            particle_coords = np.vstack([0.1*sel.positions for ts in u.trajectory[::stride*10]])

            cls.info(f"generate_mesh: Building KDTree of {len(particle_coords)} near-particle points")
            part_tree = KDTree( particle_coords )

            def _get_refinement_steps():
                spacing0 = np.max([a[-1]-a[0] for a in xyz])/initial_points
                spacing1 = desired_finest_mesh_resolution
                refinement_steps = np.log(spacing0/spacing1) / np.log(2)
                return int(np.ceil(refinement_steps))
            refinement_steps = _get_refinement_steps()

            for step in range(refinement_steps):
                cls.info(f'generate_mesh: Beginning refinement step {step+1}/{refinement_steps}')
                markers = MeshFunction('bool', mesh, mesh.topology().dim(), False)

                cls.trace(f'generate_mesh: Finding mesh cell positions')

                ## Get each cell's approximate center
                verts = [np.array(cell.get_vertex_coordinates()).reshape(-1,3)
                              for cell in cells(mesh)]
                r = np.array([v.mean(axis=0) for v in verts])
                # cls.trace(f'generate_mesh: zmin/zmax: {(np.min(r[:,2]),np.max(r[:,2]))}')
                cls.trace(f'generate_mesh: Found {len(r)} cell positions')

                ## Get a cutoff that accounts roughly for each cell's size
                max_vert_dists = np.sqrt(np.array([ np.max( ((v-r[i,:])**2).sum(axis=-1) )
                                                    for i,v in enumerate(verts) ]))
                cutoffs = np.array([v * 1.5 + 1.5*gradient_resolution for v in max_vert_dists])
                cutoff = np.max(cutoffs)

                part_cutoffs = np.array([v * 1.5 + 1.5*cls.dist_to_cond.cutoff for v in max_vert_dists])
                part_cutoff = np.max(part_cutoffs)

                cls.info(f'generate_mesh: min/max cell sizes: {np.min(max_vert_dists):0.2f}/{np.max(max_vert_dists):0.2f}')

                cls.trace(f'generate_mesh: Searching tree for cells within cutoff {cutoff:0.2f}')
                d,idx = tree.query( r, distance_upper_bound=cutoff )

                part_d,idx = part_tree.query( r, distance_upper_bound=part_cutoff )

                cls.info(f'generate_mesh: Determining which cells to refine')
                assert(d.shape == cutoffs.shape )
                cond1 = (d < cutoffs) # cells near conductance grid gradient
                cond2 = (part_d < part_cutoffs) # cells near protein positions

                ## cond3: large cells near barrier in plane
                sl = (max_vert_dists > 2)
                cond3 = np.zeros(d.shape, dtype=bool)
                cond3[sl] = sum([np.abs(r[sl][:,2]-z) < cutoffs[sl] for z in high_grad_zs],
                                cond3[sl])
                
                cls.info(f'generate_mesh: Marking cells for refinement')
                for cell, mark in zip(cells(mesh), ( (cond1 | cond2) | cond3 )):
                    markers[cell] = mark

                cls.info(f'generate_mesh: Refining {(cond1|cond2|cond3).sum()}/{len(r)} mesh cells (cond1/cond2/cond3: {cond1.sum()}/{cond2.sum()}/{cond3.sum()})')
                mesh = refine(mesh,markers)
                
            with XDMFFile(_cache_filename) as fh:
                fh.write(mesh)
        else:
            mesh = Mesh()
            with XDMFFile(_cache_filename) as fh:
                fh.read(mesh)
                
            # verts = [np.array(cell.get_vertex_coordinates()).reshape(-1,3)
            #          for cell in cells(mesh)]

            # r = np.array([v.mean(axis=0) for v in verts])
            # np.savetxt('mesh_centers.txt', r)
            # max_vert_dists = np.sqrt(np.array([ np.max( ((v-r[i,:])**2).sum(axis=-1) )
            #                                     for i,v in enumerate(verts) ]))
            # np.savetxt('mesh_sizes.txt', max_vert_dists)

            # for frame in (0,20,40,60):
            #     cond = cls.get_conductance('',frame, r)
            #     np.savetxt(f'mesh_cond_{frame:04d}.txt', cond)

            
        return mesh

    @classmethod
    def get_distances_from_atom_surfaces(cls, coordinates, particle_positions, particle_radii, cutoff):
        max_dist = np.max(particle_radii) + cutoff
        dists = np.ones(coordinates[...,0].shape) * cutoff
        log_memory(f"Distance calc start")
        ## Search distance with KDTrees
        for rad in np.unique(particle_radii):
            sl = (particle_radii == rad)
            cls.info(f'  Calculating distances from {sl.sum()} atoms with radius {rad}')
            log_memory(f"Building KDTree for radius {rad}")
            atom_tree = KDTree(particle_positions[sl])
            dists_from_subset = atom_tree.query( coordinates , distance_upper_bound = cutoff+rad)[0] - rad
            log_memory(f"KDTree query complete for radius {rad}")
            sl = (dists_from_subset < dists)
            dists[sl] = dists_from_subset[sl]
        log_memory(f"Distance calc complete")
        return dists
        
    
    @classmethod
    def get_conductance(cls, key, frame, coordinates):
        u = cls.universe
        num_frames = len(u.trajectory)//cls.stride
        cls.info(f"get_conductance_grid: frame = {frame+1} / {num_frames}")

        ## Note, we're ignoring the argument 'key' in this implementation
        u.trajectory[frame]
        part_coords = 0.1*u.select_atoms(sem_sel).positions
        part_radii = 0.1*u.select_atoms(sem_sel).radii
        log_memory(f"Frame {frame} particles extracted")
        try:
            if cls.dist_to_cond is None:
                raise Exception
            cls.base_cond_interp
        except:
            cls.warn("MySEM class dist_to_cond attribute not found... creating")
            cls.dist_to_cond = ConductivityMap()
            cls.bulk_conductivity = cls.dist_to_cond.bulk_conductivity # TODO: try to remove this line
            cond = cls.base_conductance_shm.get_array()
            x,y,z = cls.base_conductance_edges
            cls.base_cond_interp = RegularGridInterpolator((x,y,z),cond,bounds_error=False)
        log_memory(f"Frame {frame} before interpolation")
        cond = cls.base_cond_interp(coordinates)
        log_memory(f"Frame {frame} after interpolation")
        ## values outside of interpolation domain will be nan, fix this here
        _z = cls.base_conductance_edges[2]
        z0 = _z[len(_z)//2]
   
        cond[np.isnan(cond) & (coordinates[:,2] < 1.7) & (coordinates[:,2] > -2.5)] = 0
        cond[np.isnan(cond) & ((coordinates[:,2] >= 1.7) | (coordinates[:,2] <= -2.5))] = 1 # cls.dist_to_cond.bulk_conductivity

        cutoff = cls.dist_to_cond.cutoff

        
        # cls.info(f"get_conductance_grid: Building KDTree of {part_coords.size} points")
        # tree = KDTree( part_coords )
        # d,idx = tree.query( coordinates, distance_upper_bound=cutoff )

        cls.info(f"get_conductance_grid: Performing KDTree search of {coordinates.size} query points")
        log_memory(f"Frame {frame} before distance calc")
        d = cls.get_distances_from_atom_surfaces(coordinates, part_coords, part_radii, cutoff)
        log_memory(f"Frame {frame} after distance calc")
        assert(len(coordinates) == len(d) )

        sl = d < cutoff
        modulation = cls.dist_to_cond( d[sl] ) / cls.dist_to_cond.bulk_conductivity
        cond[sl] = cond[sl] * modulation

        cond = cond * cls.dist_to_cond.bulk_conductivity # finally convert (already done elsewhere
        
        cls.info(f"get_conductance_grid: {np.sum(cond<0)} negative conductance values out of {cond.size}:")
        cond[cond<0] = 0        # zero any negative values
        log_memory(f"Frame {frame} complete")
        cls.info(f"get_conductance_grid: frame = {frame} DONE")
        return cond
    
    def run(self, num_procs=4):

        ## The spawn function will bind any keyword arguments to the class, making mp.RawArray objects available as shared memory;
        ## Basically, you should put anything you'll need later in the functions that process this stuff
        output_period = 1e4
        timestep = 20e-6
        
        ## Expand the SEM domain...
        x,y,z = self.domain

        spawn = self.spawn(num_procs)

        with spawn as pool:

            key = ''

            stride = self.stride
            complete_last_step = len(u.trajectory[::stride])-1

            dirname = f'{self.prefix}_{self.voltage}mV'
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            last_step = -1
            bc= self.bulk_conductivity
            assert(bc is not None)
            lines_per_file = 100

            # self.__class__.worker_init(voltage=self.voltage, x=x, y=y, z=z,
            #                            args_dict = dict(dist_to_cond = self.dist_to_cond,
            #                                             base_conductance_shm = base_conductance_shm,
            #                                             base_conductance_edges = base_conductance_edges,
            #                                             protein_coords = self.protein_coords,
            #                                             protein_offsets = self.protein_offsets,
            #                                             num_frames = len(self.protein_offsets)
            #                                             )
            #                            )
            # self.__class__.run_task('',1,bc,1)
            
            # print(self.protein_offsets.shape)
            
            for i in range(complete_last_step//(lines_per_file)):
                first_step =     i*lines_per_file*stride
                last_step  = (i+1)*lines_per_file*stride
                filename = dirname+"/{:07d}.txt".format(first_step)
                if os.path.exists(filename): continue
                log_memory(f"Batch {i} start (frames {first_step}-{last_step})")
                args = [(key,f,bc, output_period * timestep * stride) for f in range(first_step,last_step,stride)]
                results = pool.starmap(self.__class__.run_task, args)
                log_memory(f"Batch {i} complete")
                results = np.array(results)
                np.savetxt(filename, results)
                log_memory(f"Batch {i} saved")
		

            if last_step < complete_last_step:
                filename = dirname+"/last.txt".format(last_step)
                args = [(key,f,bc, output_period * timestep * stride) for f in range(last_step,complete_last_step,stride)]
                results = pool.starmap(self.__class__.run_task, args)

                results = np.array(results)
                np.savetxt(filename, results)

if __name__ == '__main__':
    __spec__ = None
    log_memory("Script start")
    base_conductance, base_conductance_edges = _get_grids()
    log_memory("Base grids created")
    base_conductance_shm = shared_ndarray(base_conductance)
    log_memory("Shared memory created")
    # from matplotlib import pyplot as plt
    # plt.imshow( base_conductance[len(base_conductance)//2,:,:].T )
    # plt.savefig( f'conductivity_side-{system_name}.png')
    # plt.imshow( base_conductance[:,:,base_conductance.shape[2]//2].T )
    # plt.savefig( f'conductivity_top-{system_name}.png' )

    x,y,z = domain = base_conductance_edges
    # domain = [np.arange( a[0], a[-1], a[1]-a[0] ) for i,a in enumerate(base_conductance_edges)]
    domain = [ (a[0],a[-1],a[1]-a[0]) for i,a in enumerate(base_conductance_edges)]
    # domain = [np.arange( (1.25+0.25*(i==2))*a[0], (1.25+0.25*(i==2))*a[-1], a[1]-a[0])
    #           for i,a in enumerate(base_conductance_edges)]
    print(domain)
    
    prefix = f'for_chris/{system_name}'
    # u = mda.Universe(f'{prefix}.pdb', f'{prefix}.1.dcd')
    u = mda.Universe(args.pqr_file, args.dcd_file)

    sem = MySEM(domain=domain,
                voltage=args.voltage,
                conductivity_model = ConductivityMap(),
                system_name=f'{system_name}',
                base_conductance_shm = base_conductance_shm,
                base_conductance_edges = base_conductance_edges,
                far_from_pore = 20,
                stride = args.stride,
                prefix = system_name, # for output
                universe = u,
                )

    from matplotlib import pyplot as plt
    r = np.array([(_x,0,_z) for _x in x for _z in z])

    sem.__class__.worker_init(
        args_dict = dict(voltage = sem.voltage,
                         domain = sem.domain,
                         stride = sem.stride,
                         dist_to_cond = sem.dist_to_cond,
                         base_conductance_shm = base_conductance_shm,
                         base_conductance_edges = base_conductance_edges,
                         universe = u)
    )

    for frame in (0,):        
        log_memory(f"Processing test frame {frame}")
        cond = sem.get_conductance('',frame, r).reshape((len(x),len(z))) * base_conductance[:,base_conductance.shape[1]//2,:]
        log_memory(f"Test frame {frame} complete")
        plt.imshow( cond )
        plt.savefig( f'conductivity_side-{system_name}-{frame:04d}.png')
        np.savetxt(f'conductivity_side-{system_name}-{frame:04d}.dat', cond)
    log_memory("Before main SEM run")
    sem.run( num_procs=args.num_procs )
    log_memory("SEM run complete")
