import logging
from abc import ABC, abstractmethod

import time
import sys
import os

if __name__ != '__main__':
    from fenics import *
    try:
        ## mshr is deprecated, so we expect it to fail
        from mshr import *
    except:
        from gmsh import *

import ctypes
import multiprocessing as mp
import multiprocessing.dummy as mp_dummy
 
import MDAnalysis as mda
from gridData import Grid
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

## https://github.com/python/cpython/issues/87115
__spec__ = None                 # not sure if it's needed, but may help with multiprocessing/pdb bug



#helper functions 

class shared_ndarray():
    """ Utility class to pack and unpack numpy array for read-only access """
    
    def __init__(self, a, type_ = None):
        """a: numpy ndarray"""
        if type_ is None:
            type_ = 'd' # ctypes.c_double
        self._size_obj = mp.RawArray('i', [int(i) for i in a.shape])
        self._obj = mp.RawArray(type_,a.flatten())

    @property    
    def shape(self):
        return np.frombuffer(self._size_obj, dtype = 'i')
        
    def get_array(self):
        """returns numpy ndarray; uses frombuffer, so should be "read only" unless copied """
        return np.frombuffer(self._obj).reshape(self.shape)

    def __len__(self):
        return self.shape[0]

class DummyPool():

    def __init__(self, processes=None, initializer=None, initargs=None):
        self.initializer = initializer
        self.initargs = initargs
    
    def starmap(self, fn, args):
        """ Implement pool method for spawn(...,dummy_threads=True) """
        return [fn(*arg) for arg in args]

    def __enter__(self):
        self.initializer(*self.initargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    
    
class AbstractSEM(ABC):

    """ Abstract class that can be specialized with a custom
    get_conductance() function to generate SEM currents with a
    voltage applied across the z-axis. 

    Runs in parallel, but requires substantial subclassing/specialization.

    May want to break up design a bit into parts: Mesher, 

    Some of the code below may be fragile, so be cautious making changes.

    We use classmethods to allow data to be shared among threads; any
    attributes from AbstractSEM that we don't want to automatically be
    copied to each worker class *must* be explicity excluded with the
    excluded_attr list

    To test: 
Maximum size with a given amount of available memory (test shared memory)

Consistency of captured results between versions


 """
    _tol_boundary = 1e-6

    count = 0
    def __init__(self, excluded_attr = None):
        self.__class__.class_init()
        self.__class__.info('Creating SEM object.')
        
        if excluded_attr is None:
            excluded_attr = []
        self.excluded_attr = set(excluded_attr + ['excluded_attr'])

    
    """ Set up class object """
    @classmethod
    def class_init(cls):
        cls.logger = logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        cls.count += 1
        # if (cls.count > 1):
        #     raise Exception("SEM class must be used as a singleton due to multiprocessing library")
        cls.__system_assembled = False

    """ Create worker pool context manager """
    def spawn(self,num_procs, dummy_threads=None):
        if dummy_threads is None: dummy_threads == (num_procs == 1)
        self.__class__.info(f'{self.__class__}.spawn: creating pool of {num_procs} threads')
        _pool_creator = DummyPool if dummy_threads else mp.get_context(method='spawn').Pool
        initkwargs = {k:v for k,v in self.__dict__.items() if k not in self.excluded_attr}
        return _pool_creator(processes=num_procs, initializer=self.__class__.worker_init, initargs=[initkwargs])

        
    """ Intitializes workers """
    @classmethod
    def worker_init(cls, args_dict):
        """ Voltage must be given in mV """
        cls.class_init()

        cls.logger = logging.getLogger(f'sem_thread({mp.current_process().name.split("-")[-1]})')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s %(name)s: %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        cls.logger.setLevel(logging.DEBUG)
        cls.logger.addHandler(handler)

        cls.start = time.time()
        if args_dict is not None:
            for k,v in args_dict.items():
                setattr(cls,k,v)

        # x,y,z = [np.arange(*a) for a in cls.domain]
        xyz = [np.linspace(r0,r1,int((r1-r0)//dr)) for r0,r1,dr in cls.domain]
        for a in xyz:
            assert(len(a) > 2)
        x,y,z = xyz
        cls.x = x
        cls.y = y
        cls.z = z
        cls.setup_vars = cls.setup_fenics()
        cls._universe = dict()

    """ Convenient logging functions """
    @classmethod
    def info(cls, *args, **kwargs):
        cls.logger.info(*args,**kwargs)
    @classmethod
    def debug(cls, *args, **kwargs):
        cls.logger.debug(*args,**kwargs)
    @classmethod
    def trace(cls, *args, **kwargs):
        cls.logger.log(5, *args, **kwargs)
    @classmethod
    def warn(cls, *args, **kwargs):
        cls.logger.warn(*args,**kwargs)

    """ Initialize fenics solver """
    @classmethod
    def setup_fenics(cls, plot_mesh=False):
        x,y,z = cls.x, cls.y, cls.z
        cls.info(f"setup_fenics: {[len(a) for a in (x,y,z)]}")
        # numx,numy,numz = [len(a) for a in (x,y,z)]
        # Dx,Dy,Dz = [len(a) for a in (x,y,z)]
        # x,y,z = [np.arange(n)*d+o for n,d,o in zip(dim,delta,origin)]
        
        solver = KrylovSolver("gmres", "amg")
        #solver = KrylovSolver("gmres", "none")
        #solver = LinearSolver("mumps")
        parameters['krylov_solver']['nonzero_initial_guess'] = True
        parameters["krylov_solver"]["monitor_convergence"] = True
        #solver.parameters["linear_solver"] = "mumps"
        #PETScOptions.set('ksp_rtol', '.05')
        solver.parameters["relative_tolerance"] = 1e-8
        solver.parameters["maximum_iterations"] = 20000
        #solver.parameters["monitor_convergence"] = True

        cls.info("setup_fenics: Generating mesh")
        mesh = cls.generate_mesh()
        # cls.info("setup_fenics: done generating mesh")
        
        if plot_mesh:
            from matplotlib import pyplot as plt
            plt.plot(mesh, 'My mesh', interactive=True)
            plt.show()

        # tol = 1e-14
        cls.info("setup_fenics: Compiling boundary conditions")
        ground = CompiledSubDomain('on_boundary && near(x[2],zwall,tol)',   tol=cls._tol_boundary, zwall = cls.z[0])
        terminal = CompiledSubDomain('on_boundary and near(x[2],zwall,tol)',tol=cls._tol_boundary, zwall = cls.z[-1])

        
        boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        ground.mark(boundary_parts,1)
        terminal.mark(boundary_parts,2)

        cls.info("setup_fenics: Creating 'Measure' object")
        ds = Measure('ds',domain=mesh, subdomain_data=boundary_parts)

        cls.info("setup_fenics: Making function space")
        #make Function Space
        V = FunctionSpace(mesh, 'P', 1)
        F = FunctionSpace(mesh, 'CG', 1)

        bc_ground = DirichletBC(V, Constant(0), ground)
        cls.info(f'setup_fenics: {cls}.voltage = {cls.voltage}')
        bc_terminal = DirichletBC(V, Constant(cls.voltage*1e-3), terminal)
        bcs = [bc_ground, bc_terminal]

        ## Define functions for variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        sig = Function(F)
        f= Constant(0)
        DE = sig*dot(grad(u), grad(v))*dx - f*v*dx
        a, L = lhs(DE), rhs(DE)

        u1 = Function(V)

        _flux = dot(Constant((0,0,1)),sig*nabla_grad(u1))

        if 0:
            ## Get flux everywhere
            # p 128 https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7.pdf#%5B%7B%22num%22%3A302%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22XYZ%22%7D%2C0%2C666%2Cnull%5D
            degree = V.ufl_element().degree()
            W = VectorFunctionSpace(mesh, 'P', degree) 
            flux_u = project( sig*grad(u1), W ) # this failed for me with: Unable to successfully call PETSc function 'MatXIJSetPreallocation'
        
        flux_in = _flux*ds(1)
        flux_out = _flux*ds(2)
        cls.info("setup_fenics: done")

        return solver, mesh, V, F, u1, sig, a, L, bcs, flux_in, flux_out

    """ Create a mesh """
    @classmethod
    def generate_mesh(cls, check_mesh=True):
        """ Generate a mesh, specialize as needed """
        x,y,z = (cls.x,cls.y,cls.z)
        # mesh = BoxMesh( Point(np.array([a[0]  for a in (x,y,z)])),
        #                 Point(np.array([a[-1] for a in (x,y,z)])),
        #                 *[len(a) for a in (x,y,z)] )
        mesh = BoxMesh( Point(np.array([a[0]+10*np.abs(np.spacing(a[0]))  for a in (x,y,z)])),
                        Point(np.array([a[-1]-10*np.abs(np.spacing(a[-1])) for a in (x,y,z)])),
                        *[len(a) for a in (x,y,z)] )

        if check_mesh:
            ## This check is really a programming assertion and should ultimately default to False
            coor = mesh.coordinates()
            for i,(a,dim) in enumerate(zip((x,y,z),'x y z'.split())):
                bad_vals = (coor[:,i] < a[0]).sum()
                if bad_vals > 0:
                    raise Exception(f'Mesh vertices lay outside of provided domain on left side of {dim}-axis ({a[0]})')
                bad_vals = (coor[:,i] > a[-1]).sum()
                if bad_vals > 0:
                    raise Exception(f'Mesh vertices lay outside of provided domain on right side of {dim}-axis ({a[-1]})')
        return mesh

    @classmethod
    def mesh_to_cell_vertices(cls, mesh):
        """ Return numpy array of vertex positions for each cell in mesh """
        vertex_coords = mesh.coordinates()
        cell_vertex_ids = mesh.cells()
        verts = np.stack([vertex_coords[:,i][cell_vertex_ids] for i in range(3)], axis=-1)
                
        # cls.trace(f'{vertex_coords[:2]} : {cell_vertex_ids[:2]}')

        """ ## Test using (slow) direct method
        verts0 = [np.array(cell.get_vertex_coordinates()).reshape(-1,3)
                  for cell in cells(mesh)]
        verts0 = np.array(verts0)
        cls.debug(f'{verts.shape} : {verts0.shape}')
        assert( np.all( np.isclose(verts,verts0) ) )
        """
        return verts
    
    """ Compute conductance. key and frame arguments could probably be better """
    @classmethod
    @abstractmethod
    def get_conductance(cls, key, frame, coordinates):
        """ return conductance for a system optionally referenced by 'key' at 'frame' at coordinates specified in Nx3 array (in angstroms)"""
        return cls.base_conductance


    """ A thread task; @classmethod is a bit of a hack for shared memory access """
    @classmethod
    def run_task(cls, key, frame, bulk_conductance, time_per_frame=1):
        cls.info(f'run_task: frame: {frame}')
        
        cond_time1 = time.time()
        solver, mesh, V, F, u1, sig, a, L, bcs, flux_in, flux_out = cls.setup_vars

        # cond,(x,y,z) = cls.get_conductance_grid(key, frame)
        # try:
        #     cond = cls.get_conductance_grid()
        # except:
        #     t = frame * time_per_frame
        #     cls.warn(f'Could not obtain conductance grid at frame {frame}...skipping')
        #     return t, np.nan, np.nan

        cond_time2 = time.time() 
        cls.info(f'run_task: time for cond_gen: {cond_time2-cond_time1}')

        ######### FEniCS section
        fenics_time1 = time.time()

        def _loadFunc(mesh, F, sig):
            vec = sig.vector()
            values = vec.get_local()

            dofmap = F.dofmap()
            my_first, my_last = dofmap.ownership_range()

            n = F.dim()
            d = mesh.geometry().dim()
            F_dof_coordinates = F.tabulate_dof_coordinates()
            F_dof_coordinates.resize((n,d))

            unowned = dofmap.local_to_global_unowned()
            dofs = filter(lambda dof: dofmap.local_to_global_index(dof) not in unowned,
                        range(my_last-my_first))
            coords = F_dof_coordinates[list(dofs)]

            ## Get conductances at coords
            values[:] = cls.get_conductance(key,frame,coords)

            vec.set_local(values)
            vec.apply('insert')

        _loadFunc(mesh,F,sig)
        if True: # not cls.__system_assembled:
            A,bb = assemble_system(a,L,bcs)
            solver.set_operator(A)
            cls.__bb = bb
            cls.__system_assembled = True

        # Solve Problem
        solver.solve( u1.vector(),cls.__bb)
        
       
        fi = assemble(flux_in)
        fo = assemble(flux_out)
        cls.info(f'run_task: flux: {fi-fo} ({fi},{fo})')

        fenics_time2 = time.time()
        cls.info(f'run_task: Time for fenics: {fenics_time2-fenics_time1}')
        cls.info(f'run_task: Total time: {fenics_time2 - cond_time1} {fenics_time2 - cls.start}')
        cls.info(f'run_task: done with {frame}.')

        t = frame * time_per_frame

        # import pdb
        # pdb.set_trace()
        
        return t, fo, fo-fi

    @classmethod
    def get_voltage_map(cls):
        solver, mesh, V, F, u1, sig, a, L, bcs, flux_in, flux_out = cls.setup_vars

        vtd = vertex_to_dof_map(V)
        coor = mesh.coordinates()
        
        def _convert_array(arr,coor,vtd):
            
            """ Legacy function that copies values from arr into a
            regular grid using a floor function to determine where in
            the grid each vertex is located """
            xp = int((coor[-1][0]-coor[0][0])*10+1)
            yp = int((coor[-1][1]-coor[0][1])*10+1)
            zp = int((coor[-1][2]-coor[0][2])*10+1)
            a = np.zeros((xp,yp,zp))
            for i, dum in enumerate(coor):
                x = int((dum[0]-coor[0][0])*10)
                y = int((dum[1]-coor[0][1])*10)
                z = int((dum[2]-coor[0][2])*10)
                a[x,y,z] = arr[vtd[i]]           
            return a

        arr = u1.vector().get_local()
        a = _convert_array(arr, coor, vtd)
        return a

    """ Analyze the subclass """
    @abstractmethod
    def run():
        """ ## EXAMPLE:
        ## The spawn function will bind any keyword arguments to the class, making mp.RawArray objects available as shared memory;
        ## Basically, you should put anything you'll need later in the functions that process this stuff
        spawn = self.spawn(num_procs,
                           voltage=self.voltage, x=x, y=y, z=z,
                           dist_to_cond = self.dist_to_cond,
                           base_conductance_shm = base_conductance_shm,
                           base_conductance_fine_shm = base_conductance_fine_shm,
                           base_conductance_edges = base_conductance_edges,
                           base_conductance_fine_edges = base_conductance_fine_edges,
                           )
        with spawn as pool:
            for replica in range(24):
                indir = '../run{:02d}.rad39'.format(replica)
                sim_type = 'run_in'
                psf = '{}/{}.psf'.format(indir,sim_type)
                globstrings = ['{}/output/{}{}.dcd'.format(indir,sim_type,suff)
                                for suff in ('','-cont??')]
                dcds = [d for s in globstrings for d in natsorted(glob(s))] 
                key = (psf,tuple(dcds))

                output_period=1e4 * 1 # fix10
                timestep = 40e-6 # ns
                u = mda.Universe(psf,*dcds)

                complete_last_step = len(u.trajectory[::stride])*stride-stride

                dirname = "{}_{}mV_rep{}_skip{}".format(sim_type,int(self.voltage),replica,stride)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                last_step = 0
                bc= self.bulk_conductance
                assert(bc is not None)
                lines_per_file = 20
                for i in range(complete_last_step//(lines_per_file*stride)):
                    first_step =     i*lines_per_file*stride
                    last_step  = (i+1)*lines_per_file*stride
                    filename = dirname+"/{:07d}.txt".format(first_step)
                    if os.path.exists(filename): continue

                    args = [(key,f,bc, output_period * timestep * stride) for f in range(first_step,last_step,stride)]
                    results = pool.starmap(self.run_task, args)
                    results = np.array(results)
                    np.savetxt(filename, results)

                if last_step < complete_last_step:
                    filename = dirname+"/last.txt".format(last_step)
                    args = [(key,f,bc, output_period * timestep * stride) for f in range(last_step,complete_last_step,stride)]
                    results = pool.starmap(self.run_task, args)
                    results = np.array(results)
                    np.savetxt(filename, results)
        """

## Useful utilities
# def voxel_distance_from_nearest_bead(bead_positions, edge_centers, cutoff=cutoff):

#     assert(len(edge_centers) == 3)
#     r_e = edge_centers           # should be three 1D arrays
#     shape = [len(a) for a in r_e]
#     dists= np.ones(shape) * cutoff # create array of distances initialized to maximum

#     r0_v = np.array( [a[0] for a in r_e] )
#     dr_v = np.array( [a[1]-a[0] for a in r_e] )
    

#     # x_init, y_init, z_init = initials[0], initials[1], initials[2]
#     # x_fin, y_fin, z_fin = finals[0], finals[1], finals[2]

#     # dnax = pos[0]
#     # dnay = pos[1]
#     # dnaz = pos[2]

#     origin_to_beads = (bead_positions-r0_v[None,:])
#     home = np.around( origin_to_beads / dr_v[None,:] ).astype(int)

#     # neighborhood = np.array([np.arange(2*cutoff/dr)*dr-cutoff for dr in dr_v])
#     DX,DY,DZ = np.meshgrid(*[np.arange(2*cutoff/dr)*dr-cutoff for dr in dr_v], indexing='ij')

    
#     dr_i0 = np.around( (bead_positions-cutoff) / dr_v[None,:] )
#     dr_i1 = np.around( (bead_positions+cutoff) / dr_v[None,:] )

#     # neighborhood = np.arange( N
#     # distance_neighborhood = home

                              
    
#     r0_v = np.maximum(np.around( (dnax-cutoff)/Dx0 )*Dx0, x_init*np.ones_like(dnax))
    
#     xi = np.maximum(np.around((dnax-cutoff)/Dx0)*Dx0,x_init*np.ones_like(dnax))
#     xi = np.where( np.abs(xi-dnax) < cutoff , np.maximum(xi-Dx0,x_init*np.ones_like(dnax)), xi)
#     xi_index=np.around((xi-x_init)/Dx0).astype(int)
#     xf = np.minimum(np.around((dnax+cutoff)/Dx0)*Dx0,x_fin*np.ones_like(dnax))
#     xf = np.where( np.abs(xf-dnax) < cutoff , np.minimum(xf+Dx0,x_fin*np.ones_like(dnax)), xf)
#     # The plus one makes sure the indices include the final one.
#     xf_index=np.around((xf-x_init)/Dx0).astype(int)+1

#     yi = np.maximum(np.around((dnay-cutoff)/Dy0)*Dy0,y_init*np.ones_like(dnay))
#     yi = np.where( np.abs(yi-dnay) < cutoff , np.maximum(yi-Dy0,y_init*np.ones_like(dnay)), yi)
#     yi_index=np.around((yi-y_init)/Dy0).astype(int)
#     yf = np.minimum(np.around((dnay+cutoff)/Dy0)*Dy0,y_fin*np.ones_like(dnay))
#     yf = np.where( np.abs(yf-dnay) < cutoff , np.minimum(yf+Dy0,y_fin*np.ones_like(dnay)), yf)
#     yf_index=np.around((yf-y_init)/Dy0).astype(int)+1

#     zi = np.maximum(np.around((dnaz-cutoff)/Dz0)*Dz0,z_init*np.ones_like(dnaz))
#     zi = np.where( np.abs(zi-dnaz) < cutoff , np.maximum(zi-Dz0,z_init*np.ones_like(dnaz)), zi)
#     zi_index=np.around((zi-z_init)/Dz0).astype(int)
#     zf = np.minimum(np.around((dnaz+cutoff)/Dz0)*Dz0,z_fin*np.ones_like(dnaz))
#     zf = np.where( np.abs(zf-dnaz) < cutoff , np.minimum(zf+Dz0,z_fin*np.ones_like(dnaz)), zf)
#     zf_index=np.around((zf-z_init)/Dz0).astype(int)+1

#     for i in range(0, len(dnax)):
#         positions = np.array([[[[xx,yy,zz] for zz in (z[zi_index[i]:zf_index[i]])] for yy in (y[yi_index[i]:yf_index[i]])] for xx in (x[xi_index[i]:xf_index[i]])])
#         try:
#             distances = np.sqrt(np.sum(np.square(positions-pos.T[i]),axis=3))
#         except:
#             continue
#         dist_section = dist[xi_index[i]:xf_index[i],yi_index[i]:yf_index[i],zi_index[i]:zf_index[i]]
#         # the second condition does two things. First, it ensures that the walls where conductivity is zero remain as such. It does this because dist was first defined to be zero outside of the capillary, and so any value of d calculated above is guaranteed to not be less than zero. Therefore, the points outside the capillary always have a value of zero in the dist array. Secondly, it only rewrites the distance array if the presently calculated distance d is less than any previously calculated distance. In this way, it takes the minimum value of distance to the DNA beads.
#         dist[xi_index[i]:xf_index[i],yi_index[i]:yf_index[i],zi_index[i]:zf_index[i]] = np.where((distances < cutoff) & (distances < dist_section),distances,dist_section)
#     return dist

def voxel_distance_from_nearest_bead_adnan(bead_positions, edge_centers, cutoff):

    assert(len(edge_centers) == 3)
    x,y,z = r_e = edge_centers           # should be three 1D arrays
    x_init, y_init, z_init = [a[0] for a in r_e]
    x_fin, y_fin, z_fin = [a[-1] for a in r_e]
    Dx0,Dy0,Dz0 = [a[1]-a[0] for a in r_e]

    dnax = bead_positions[:,0]
    dnay = bead_positions[:,1]
    dnaz = bead_positions[:,2]

    dist = np.ones([len(a) for a in r_e])*cutoff
    
    xi = np.maximum(np.around((dnax-cutoff)/Dx0)*Dx0,x_init*np.ones_like(dnax))
    xi = np.where( np.abs(xi-dnax) < cutoff , np.maximum(xi-Dx0,x_init*np.ones_like(dnax)), xi)
    xi_index=np.around((xi-x_init)/Dx0).astype(int)
    xf = np.minimum(np.around((dnax+cutoff)/Dx0)*Dx0,x_fin*np.ones_like(dnax))
    xf = np.where( np.abs(xf-dnax) < cutoff , np.minimum(xf+Dx0,x_fin*np.ones_like(dnax)), xf)
    # The plus one makes sure the indices include the final one.
    xf_index=np.around((xf-x_init)/Dx0).astype(int)+1

    yi = np.maximum(np.around((dnay-cutoff)/Dy0)*Dy0,y_init*np.ones_like(dnay))
    yi = np.where( np.abs(yi-dnay) < cutoff , np.maximum(yi-Dy0,y_init*np.ones_like(dnay)), yi)
    yi_index=np.around((yi-y_init)/Dy0).astype(int)
    yf = np.minimum(np.around((dnay+cutoff)/Dy0)*Dy0,y_fin*np.ones_like(dnay))
    yf = np.where( np.abs(yf-dnay) < cutoff , np.minimum(yf+Dy0,y_fin*np.ones_like(dnay)), yf)
    yf_index=np.around((yf-y_init)/Dy0).astype(int)+1

    zi = np.maximum(np.around((dnaz-cutoff)/Dz0)*Dz0,z_init*np.ones_like(dnaz))
    zi = np.where( np.abs(zi-dnaz) < cutoff , np.maximum(zi-Dz0,z_init*np.ones_like(dnaz)), zi)
    zi_index=np.around((zi-z_init)/Dz0).astype(int)
    zf = np.minimum(np.around((dnaz+cutoff)/Dz0)*Dz0,z_fin*np.ones_like(dnaz))
    zf = np.where( np.abs(zf-dnaz) < cutoff , np.minimum(zf+Dz0,z_fin*np.ones_like(dnaz)), zf)
    zf_index=np.around((zf-z_init)/Dz0).astype(int)+1

    for i in range(0, len(dnax)):
        positions = np.array([[[[xx,yy,zz] for zz in (z[zi_index[i]:zf_index[i]])] for yy in (y[yi_index[i]:yf_index[i]])] for xx in (x[xi_index[i]:xf_index[i]])])
        try:
            distances = np.sqrt(np.sum(np.square(positions-bead_positions[i]),axis=3))
        except:
            continue
        dist_section = dist[xi_index[i]:xf_index[i],yi_index[i]:yf_index[i],zi_index[i]:zf_index[i]]
        # the second condition does two things. First, it ensures that the walls where conductivity is zero remain as such. It does this because dist was first defined to be zero outside of the capillary, and so any value of d calculated above is guaranteed to not be less than zero. Therefore, the points outside the capillary always have a value of zero in the dist array. Secondly, it only rewrites the distance array if the presently calculated distance d is less than any previously calculated distance. In this way, it takes the minimum value of distance to the DNA beads.
        dist[xi_index[i]:xf_index[i],yi_index[i]:yf_index[i],zi_index[i]:zf_index[i]] = np.where((distances < cutoff) & (distances < dist_section),distances,dist_section)
    return dist

class ConductivityMapAdnan():
        
    ## TODO: refactor some of MySEM.get_conductance_grid() into this
    def __init__(self, concentration, type_='KCl', temperature=25):
        """ distsances are in nm """
        self.cutoff = 2.6 # may be overridden below
        self.concentration = concentration
        self.bulk_conductance = None
        self._interp_functions = None
        
        if type_ != 'KCl':
            raise NotImplementedError

        if temperature != 25:
            ## Note, the numbers below are not necessarily for 25 C.
            raise NotImplementedError

        if concentration == 100: # mM
            ## From Chris's 2010 PRL, SI 10c, 170mM Na+ for the positive ions, and 170mM Cl- for the negative ions
            ## Note: the bulk concentration is actuall pretty close to 200 mM
            rs=np.linspace(0,26,1000)/10
            def _cl_number_density(rs): # I think it is Cl, have not checked carefully
                ## TODO: Check
                center=10.5/10
                slope=8/10
                ## units "208 mM" "1.25e-4 particles/AA**3"
                saturated=1.25e-4 ## ions/AA**3
                func = saturated*.5*(1+np.tanh((rs-center)/slope))
                return (func[-1]/(func[-1]-func[0])*(func-func[0]))

            ## num_densities here are given as particles/AA**3
            # bulk_mobilities = tuple((55e-5,57e-5))
            ## units "(1/AA**3) * ((e AA/s)/(mV/nm))" "S/m"
            conv = 1.6021765e-05
            bulk_mobilities = tuple(x*conv for x in (55.0,57.0)) ## CM: not sure where Adnan originally obtained these numbers, but if I drop the e-5, I get a sensible bulk conductance of ~2 S/m

            num_densities = [interp1d(np.concatenate([[0],np.arange(0.3,30,2)])/10, np.array([0, 0,1.3,5,4.9,3.75,6.3,7.5,5,3,2,1.8,1.6,1.5,1.35,1.25])*1e-4, kind='cubic'),
                             interp1d(rs,_cl_number_density(rs),kind='cubic')
            ]
            mobilities = [interp1d(
                np.concatenate([[0],np.arange(0.5,27,1)])/10,
                bulk_mobilities[0]*np.concatenate([[0,0,0,0,0],np.array([.01,.05,.09,.12,.23,.36,.61,.75,.83,.95,1,1,1,1,1,1,1,1,1,1,1,1,1])]),
                kind='cubic'),
                          interp1d(
                np.concatenate([[0],np.arange(0.5,27,1)])/10,
                bulk_mobilities[1]*np.concatenate([[0,0,0,0,0],np.array([.01,.05,.09,.12,.23,.36,.61,.75,.83,.95,1,1,1,1,1,1,1,1,1,1,1,1,1])]),
                kind='cubic')]

            ## Adnan's code does the interpolation twice, pretty sure we can do it once, but will need to test            
            self._interp_functions = tuple((interp1d(rs, num_densities[0](rs) * mobilities[0](rs), kind='cubic'),
                                           interp1d(rs, num_densities[1](rs) * mobilities[1](rs), kind='cubic')))

        elif (concentration == 1000) or (concentration == 4000):
            if concentration == 1000:
                ## https://www-origin.horiba.com/uk/application/material-property-characterization/water-analysis/water-quality-electrochemistry-instrumentation/accessories-and-consumables/standard-solution-internal-solution/conductivity-standard-solutions/1/
                BULK = BULK1M = 11.18 # S/m ± 1% Calibration Solution, 1M KCl at 25°C 250ml in dosing bottle with certificate
                # self.bulk_conductance = BULK

            elif concentration == 4000:
                BULK = BULK4M = 17.3 # S/m bulk conductivity of 1M KCl
                # self.bulk_conductance = BULK
                raise NotImplementedError

            self.cutoff = 2.55 # may be overridden below
            self._interp_functions = tuple((
                interp1d(np.concatenate([np.linspace(0,.4,9),np.linspace(.55,2.55,21)]),
                         BULK1M/(.29+.35)*np.array([0,0,0,0,0,0,0,0,0,
                                                    0.035,.05,.04,.045,.07,.115,.2,.305, .36,.35,.33,.33,.315,.3,.305,.3,.295,.295,.285,.29,.29]),kind='cubic'),
                interp1d(np.concatenate([np.linspace(0,.4,9),np.linspace(.55,2.55,21)]),
                         BULK1M/(.29+.35)*np.array([0,0,0,0,0,0,0,0,0,
                                                    0, 0,.005,.01 ,.02,.035,.06,.1,.16,.215,.265, .295, .31,.32, .325, .34, .35,.34, .35, .35,.35]),kind='cubic')))
        else:
            print(concentration)
            # data = ConductivityMap._get_KCl_data()
            raise NotImplementedError

        self.bulk_conductance = self(self.cutoff)

    def __call__(self, dists):
        ## TODO combine interp functions
        na_fn,cl_fn = self._interp_functions
        return na_fn(dists) + cl_fn(dists)

class ConductivityMapAdnan2():
    ## I think this is the final model that Adnan used for 170 mM NaCl (sort of)
    ## adapted from /data/server2/adnanch/2capillary/final_50.invited/sem_analysis.py

    ## TODO: refactor some of MySEM.get_conductance_grid() into this
    def __init__(self, concentration, type_='KCl', temperature=25):
        """ distsances are in nm """
        self.cutoff = 2.6 # may be overridden below
        self.concentration = concentration
        self.bulk_conductance = None
        self._interp_functions = None
        
        if type_ != 'KCl':
            raise NotImplementedError

        if temperature != 25:
            ## Note, the numbers below are not necessarily for 25 C.
            raise NotImplementedError

        if concentration == 170: # mM
            ## From Chris's 2010 PRL, SI 10c, 170mM Na+ for the positive ions, and 170mM Cl- for the negative ions
            ## Note: the bulk concentration is actuall pretty close to 200 mM
            rs=np.linspace(0,26,1000)/10
            def _cl_number_density(rs): # I think it is Cl, have not checked carefully
                ## TODO: Check
                center=10.5/10
                slope=8/10
                saturated=1.25e-4*10**30  ## ions/m**3
                func = saturated*.5*(1+np.tanh((rs-center)/slope))
                return (func[-1]/(func[-1]-func[0])*(func-func[0]))

            # arrays have positive ion values first and negative second
            qs = (1.6*10**(-19),1.6*10**(-19))

            # 0: bulk mobility values could be taken from BELK2016 s5c. But experimental values in m^2/Vs for NaCl are used I got them from googling, but can be found in https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2010WR009551
            # 0: normalized mobility curves taken from Figure from BELK2016 3d, 2M KCl, K+ at T=295K
            bulk_mobilities = (5.19e-8,7.91e-8)

            num_densities = [interp1d(np.concatenate([[0],np.arange(0.3,30,2)])/10, np.array([0, 0,1.3,5,4.9,3.75,6.3,7.5,5,3,2,1.8,1.6,1.5,1.35,1.25])*1e-4*10**30, kind='cubic'),
                             interp1d(rs,_cl_number_density(rs),kind='cubic')
            ]
            
            mobilities = [interp1d(
                np.concatenate([[0],np.arange(0.5,27,1)])/10,
                bulk_mobilities[0]*np.concatenate([[0,0,0,0,0],np.array([.01,.05,.09,.12,.23,.36,.61,.75,.83,.95,1,1,1,1,1,1,1,1,1,1,1,1,1])]),
                kind='cubic'),
                          interp1d(
                np.concatenate([[0],np.arange(0.5,27,1)])/10,
                bulk_mobilities[1]*np.concatenate([[0,0,0,0,0],np.array([.01,.05,.09,.12,.23,.36,.61,.75,.83,.95,1,1,1,1,1,1,1,1,1,1,1,1,1])]),
                kind='cubic')]

            ## Adnan's code does the interpolation twice, pretty sure we can do it once, but will need to test            
            self._interp_functions = tuple((interp1d(rs, qs[0]*num_densities[0](rs) * mobilities[0](rs), kind='cubic'),
                                           interp1d(rs, qs[1]*num_densities[1](rs) * mobilities[1](rs), kind='cubic')))

            # print("WARNING: DEBUG")
            # # self._interp_functions = tuple((mobilities[0],mobilities[1]))
            # self._interp_functions = tuple((num_densities[0],num_densities[1]))
            
        else:
            print(concentration)
            # data = ConductivityMap._get_KCl_data()
            raise NotImplementedError

        self.bulk_conductance = self(self.cutoff)

    def __call__(self, dists):
        ## TODO combine interp functions
        na_fn,cl_fn = self._interp_functions
        return na_fn(dists) + cl_fn(dists)

class ConductivityMapSingleStranded():

    ## TODO: refactor some of MySEM.get_conductance_grid() into this
    def __init__(self, concentration, type_='KCl', temperature=25):
        """ distsances are in nm """
        self.cutoff = 2.6 # may be overridden below
        self.concentration = concentration
        self.bulk_conductance = None
        self._interp_functions = None
        
        if type_ != 'KCl':
            raise NotImplementedError

        if temperature != 25:
            ## Note, the numbers below are not necessarily for 25 C.
            raise NotImplementedError

        if concentration == 170: # mM
            ## From Chris's 2010 PRL, SI 10c, 170mM Na+ for the positive ions, and 170mM Cl- for the negative ions
            ## Note: the bulk concentration is actuall pretty close to 200 mM
            rs=np.linspace(0,26,1000)/10
            def _cl_number_density(rs): # I think it is Cl, have not checked carefully
                ## TODO: Check
                center=10.5/10
                slope=8/10
                saturated=1.25e-4*10**30## ions/m**3
                func = saturated*.5*(1+np.tanh((rs-center)/slope))
                return (func[-1]/(func[-1]-func[0])*(func-func[0]))

            # arrays have positive ion values first and negative second
            qs = (1.6*10**(-19),1.6*10**(-19))

            # 0: bulk mobility values could be taken from BELK2016 s5c. But experimental values in m^2/Vs for NaCl are used I got them from googling, but can be found in https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2010WR009551
            # 0: normalized mobility curves taken from Figure from BELK2016 3d, 2M KCl, K+ at T=295K
            bulk_mobilities = (5.19e-8,7.91e-8)

            num_densities = [interp1d(np.concatenate([[0],np.arange(0.3,30,2)])/10, np.array([0, 0,1.3,5,4.9,3.75,6.3,7.5,5,3,2,1.8,1.6,1.5,1.35,1.25])*1e-4*10**30, kind='cubic'),
                             interp1d(rs,_cl_number_density(rs),kind='cubic')
            ]
            
            mobilities = [interp1d(
                np.concatenate([[0],np.arange(0.5,27,1)])/10,
                bulk_mobilities[0]*np.concatenate([[0,0,0,0,0],np.array([.01,.05,.09,.12,.23,.36,.61,.75,.83,.95,1,1,1,1,1,1,1,1,1,1,1,1,1])]),
                kind='cubic'),
                          interp1d(
                np.concatenate([[0],np.arange(0.5,27,1)])/10,
                bulk_mobilities[1]*np.concatenate([[0,0,0,0,0],np.array([.01,.05,.09,.12,.23,.36,.61,.75,.83,.95,1,1,1,1,1,1,1,1,1,1,1,1,1])]),
                kind='cubic')]

            ## Adnan's code does the interpolation twice, pretty sure we can do it once, but will need to test            
            self._interp_functions = tuple((interp1d(rs, qs[0]*num_densities[0](rs) * mobilities[0](rs), kind='cubic'),
                                           interp1d(rs, qs[1]*num_densities[1](rs) * mobilities[1](rs), kind='cubic')))

            # print("WARNING: DEBUG")
            # # self._interp_functions = tuple((mobilities[0],mobilities[1]))
            # self._interp_functions = tuple((num_densities[0],num_densities[1]))
            
        else:
            print(concentration)
            # data = ConductivityMap._get_KCl_data()
            raise NotImplementedError

        self.bulk_conductance = self(self.cutoff)

    def __call__(self, dists):
        ## TODO combine interp functions
        na_fn,cl_fn = self._interp_functions
        return (na_fn(dists) + cl_fn(dists))/(na_fn(2.6)+cl_fn(2.6))
    
class ConductivityMapSingleStranded_allatom():

    ## TODO: refactor some of MySEM.get_conductance_grid() into this
    def __init__(self, concentration, type_='KCl', temperature=25):
        """ distsances are in nm """
        self.cutoff = 2.6 # may be overridden below
        self.concentration = concentration
        self.bulk_conductance = None
        self._interp_functions = None
        
        if type_ != 'KCl':
            raise NotImplementedError

        if temperature != 25:
            ## Note, the numbers below are not necessarily for 25 C.
            raise NotImplementedError

        if concentration == 170: # mM
            ## From Chris's 2010 PRL, SI 10c, 170mM Na+ for the positive ions, and 170mM Cl- for the negative ions
            ## Note: the bulk concentration is actuall pretty close to 200 mM
            rs=np.linspace(0,26,1000)/10
            def _cl_number_density(rs): # I think it is Cl, have not checked carefully
                ## TODO: Check
                center=10.5/10 
                slope=8/10
                saturated=1.25e-4*10**30 *2 ## ions/m**3
                func = saturated*.5*(1+np.tanh((rs-center)/slope))
                return (func[-1]/(func[-1]-func[0])*(func-func[0]))

            # arrays have positive ion values first and negative second
            qs = (1.6*10**(-19),1.6*10**(-19))

            # 0: bulk mobility values could be taken from BELK2016 s5c. But experimental values in m^2/Vs for NaCl are used I got them from googling, but can be found in https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2010WR009551
            # 0: normalized mobility curves taken from Figure from BELK2016 3d, 2M KCl, K+ at T=295K
            bulk_mobilities = (5.19e-8,7.91e-8)

            num_densities = [interp1d(np.concatenate([[0],np.arange(0.3,30,2)])/10, np.array([0, 0,1.3,5,4.9,3.75,6.3,7.5,5,3,2,1.8,1.6,1.5,1.35,1.25])*1e-4*10**30, kind='cubic'),
                             interp1d(rs,_cl_number_density(rs),kind='cubic')
            ]
            
            mobilities = [interp1d(
                np.concatenate([[0],np.arange(0.5,27,1)])/10,
                bulk_mobilities[0]*np.concatenate([[0,0,0,0,0],np.array([.01,.05,.09,.12,.23,.36,.61,.75,.83,.95,1,1,1,1,1,1,1,1,1,1,1,1,1])]),
                kind='cubic'),
                          interp1d(
                np.concatenate([[0],np.arange(0.5,27,1)])/10,
                bulk_mobilities[1]*np.concatenate([[0,0,0,0,0],np.array([.01,.05,.09,.12,.23,.36,.61,.75,.83,.95,1,1,1,1,1,1,1,1,1,1,1,1,1])]),
                kind='cubic')]

            ## Adnan's code does the interpolation twice, pretty sure we can do it once, but will need to test            
            self._interp_functions = tuple((interp1d(rs, qs[0]*num_densities[0](rs) * mobilities[0](rs), kind='cubic'),
                                           interp1d(rs, qs[1]*num_densities[1](rs) * mobilities[1](rs), kind='cubic')))

            # print("WARNING: DEBUG")
            # # self._interp_functions = tuple((mobilities[0],mobilities[1]))
            # self._interp_functions = tuple((num_densities[0],num_densities[1]))
            
        else:
            print(concentration)
            # data = ConductivityMap._get_KCl_data()
            raise NotImplementedError

        self.bulk_conductance = self(self.cutoff)

    def __call__(self, dists):
        ## TODO combine interp functions
        na_fn,cl_fn = self._interp_functions
        return (na_fn(dists) + cl_fn(dists))/(na_fn(2.6)+cl_fn(2.6))
        
    
    
class ConductivityMap():

    def _get_KCl_conductance_data(concentration, temperature = 25, interpolation='logarithmic'):
        ## https://hbcp.chemnetbase.com/faces/documents/05_07/05_07_0001.xhtml
        """
STANDARD KCl SOLUTIONS FOR CALIBRATING ELECTRICAL CONDUCTIVITY CELLS

This table presents recommended electrical conductivity (κ) values for aqueous potassium chloride solutions with molalities of 0.01 mol kg-1, 0.1 mol kg-1, and 1.0 mol kg-1 at temperatures from 0 °C to 50 °C. The values, which are based on measurements at the National Institute of Standards and Technology, provide primary standards for the calibration of conductivity cells. The measurements at 0.01 and 0.1 molal are described in Ref. 1, while those at 1.0 molal are in Ref. 2. Temperatures are given on the ITS-90 scale. The uncertainty in the conductivity is about 0.03% for the 0.01 molal values and about 0.04% for the 0.1 and 1.0 molal values.

Column definitions for the table are as follows.
Column heading 	Definition
t 	Temperature, in °C
κ(concentration) 	Electrical conductivity, in units μS m-1; in columns 2-4, concentration values are indicated in parentheses; in the last column, the value is for water saturated with atmospheric CO2

Conductivity values in the last column were subtracted from the original measurements to give the values in the preceding columns for KCl (aq). 

The assistance of Kenneth W. Pratt is appreciated.
References

    Wu, Y. C., Koch, W. F., and Pratt, K. W., J. Res. Natl. Inst. Stand. Technol. 96, 191, 1991. [https://doi.org/10.6028/jres.096.008]
    Wu, Y. C., Koch, W. F., Feng, D., Holland, L. A., Juhasz, E., Arvay, E., and Tomek, A., J. Res. Natl. Inst. Stand. Technol. 99, 241, 1994. [https://doi.org/10.6028/jres.099.019]
    Pratt, K. W., Koch, W. F., Wu, Y. C., and Berezansky, P. A., Pure Appl. Chem. 73, 1783, 2001. [https://doi.org/10.1351/pac200173111783]
"""

        
        ## Row	Name            	Formula	CAS Reg. No.	Mol. Wt.	t/ºC	κ(0.01 m)/μS m-1	κ(0.1 m)/μS m-1	κ(1.0 m)/μS m-1	κ(H2O)/μS m-1
        txt = """1	Potassium chloride	KCl	7447-40-7	74.551	0	772.92	7116.85	63488	0.58
2	Potassium chloride	KCl	7447-40-7	74.551	5	890.96	8183.70	72030	0.68
3	Potassium chloride	KCl	7447-40-7	74.551	10	1013.95	9291.72	80844	0.79
4	Potassium chloride	KCl	7447-40-7	74.551	15	1141.45	10437.1	89900	0.89
5	Potassium chloride	KCl	7447-40-7	74.551	18	1219.93	11140.6	nan	0.95
6	Potassium chloride	KCl	7447-40-7	74.551	20	1273.03	11615.9	99170	0.99
7	Potassium chloride	KCl	7447-40-7	74.551	25	1408.23	12824.6	108620	1.10
8	Potassium chloride	KCl	7447-40-7	74.551	30	1546.63	14059.2	118240	1.20
9	Potassium chloride	KCl	7447-40-7	74.551	35	1687.79	15316.0	127970	1.30
10	Potassium chloride	KCl	7447-40-7	74.551	40	1831.27	16591.0	137810	1.40
11	Potassium chloride	KCl	7447-40-7	74.551	45	1976.62	17880.6	147720	1.51
12	Potassium chloride	KCl	7447-40-7	74.551	50	2123.43	19180.9	157670	1.61"""

         
        temperature_col = 5
        conductance_cols = [6,7,8]
        concentrations = np.array([0.01, 0.1, 1.]) * 1e3

        split_lines = [l.split('\t') for l in txt.split('\n')]
        temperatures = [float(l[temperature_col]) for l in split_lines]
        row = temperatures.index(float(temperature))
        assert( row > 0 )

        line = split_lines[row]
        conds = [float(line[i])*1e-4 for i in conductance_cols]

        right = np.where(concentrations >= concentration)[0][0]
        left = right-1
        if left < 0:
            raise ValueError("concentration is too small")
            
        if right == len(concentrations):
            raise ValueError("concentration is too large")
        
        ## Linearly interpolate in logarithmic space
        c = concentration
        c_l,c_r = [concentrations[i] for i in (left,right)]
        cond_l, cond_r = [conds[i] for i in (left,right)]

        if interpolation == 'logarithmic':
            x = (np.log(c)-np.log(c_l))/(np.log(c_r)-np.log(c_l))   # interpolation parameter
            cond = np.exp((np.log(cond_r) - np.log(cond_l)) * x + np.log(cond_l))
        elif interpolation == 'linear':
            x = (c-c_l)/(c_r-c_l)   # interpolation parameter
            cond = (cond_r - cond_l) * x + cond_l
        else:
            raise NotImplementedError
            
        return cond
        
    ## TODO: refactor some of MySEM.get_conductance_grid() into this
    def __init__(self, concentration, type_='KCl', temperature=25):
        """ distsances are in nm """
        self.cutoff = 2.6 # may be overridden below
        self.concentration = concentration
        self.bulk_conductance = None
        self._interp_functions = None
        
        if type_ != 'KCl':
            raise NotImplementedError

        if temperature != 25:
            raise NotImplementedError

        BULK = ConductivityMap._get_KCl_conductance_data( concentration )
        # for c in [100, 200, 400, 600, 800, 1000]:
        #     print(c, ConductivityMap._get_KCl_conductance_data(c))
        # for c in [100, 200, 400, 600, 800, 1000]:
        #     print(c, ConductivityMap._get_KCl_conductance_data(c,interpolation='linear'))
        
        if concentration == 1000:
            ## https://www-origin.horiba.com/uk/application/material-property-characterization/water-analysis/water-quality-electrochemistry-instrumentation/accessories-and-consumables/standard-solution-internal-solution/conductivity-standard-solutions/1/
            BULK1M = 11.18 # S/m ± 1% Calibration Solution, 1M KCl at 25°C 250ml in dosing bottle with certificate
            # self.bulk_conductance = BULK

        elif concentration == 4000:
            BULK4M = 17.3 # S/m bulk conductivity of 4M LiCl
            # self.bulk_conductance = BULK
            raise NotImplementedError

        self.cutoff = 2.55 # may be overridden below
        self._interp_functions = tuple((
            interp1d(np.concatenate([np.linspace(0,.4,9),np.linspace(.55,2.55,21)]),
                     BULK/(.29+.35)*np.array([0,0,0,0,0,0,0,0,0,
                                                0.035,.05,.04,.045,.07,.115,.2,.305, .36,.35,.33,.33,.315,.3,.305,.3,.295,.295,.285,.29,.29]),kind='cubic'),
            interp1d(np.concatenate([np.linspace(0,.4,9),np.linspace(.55,2.55,21)]),
                     BULK/(.29+.35)*np.array([0,0,0,0,0,0,0,0,0,
                                                0, 0,.005,.01 ,.02,.035,.06,.1,.16,.215,.265, .295, .31,.32, .325, .34, .35,.34, .35, .35,.35]),kind='cubic')))


        self.bulk_conductance = self(self.cutoff)

    def __call__(self, dists):
        ## TODO combine interp functions
        na_fn,cl_fn = self._interp_functions
        return na_fn(dists) + cl_fn(dists)

## TODO: Use python testing framework to test things, eg. result of blank inputs

if __name__ == '__main__':
    cm = ConductivityMap(400)
