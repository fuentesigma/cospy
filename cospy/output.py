'''
Created on 15 May 2015

@author: Pablo Galaviz
'''
import logging
import h5py
import numpy as np 
import csv
from mpi4py import MPI
import time

class Output(object):
    '''
    classdocs
    '''


    def __init__(self, params, project_directory):
        '''
        Constructor
        '''

        self.start_time=time.time()

        logging.info("---------- output setup ----------")

        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_size = self.mpi_comm.Get_size() # total number of nodes
        self.mpi_rank = self.mpi_comm.Get_rank() # current node 

     
        self.verbose = params.getboolean('verbose', False)

        self.debug = params.getboolean('debug', False)
                
        self.delta_t = abs(params.getfloat('delta output',0))

        self.delta_a = abs(params.getfloat('delta analysis',0))

        self._fields = params.get('output fields').replace(' ','').split(',')

        self.h5file = h5py.File(project_directory+'/output.h5','w', driver='mpio', comm=self.mpi_comm)
        self.h5file.atomic = True
#        self.h5file = h5py.File(project_directory+'/output'+str(self.mpi_rank)+'.h5')


        self.csv_file = project_directory+'/scalars.csv'

        self.iteration = 0
        self.it = 0

        self.next_output = 0

        self.next_analysis = 0

        self._fields_in_grid=list()

    def init(self, grid, problem):

        
        for fname in self._fields:
            if grid.has_field(fname) :
                self.h5file.create_group(fname)
                self._fields_in_grid.append(fname)
                logging.info('Output set for field: %s', fname)
                         
        
        self.init_axis(grid)
        
        self.problem_analysis = getattr(problem, 'analysis',None)

        if self.problem_analysis == None:
            #logging.warning('Problem does not implements "analysis" method')
            self.problem_analysis = lambda grid, t : dict()


    def init_axis(self, grid):

        x, y, z  = grid.get_coordinates( cglobal=True)
        coordinates = self.h5file.create_group('coordinates')
    
        coordinates['x'] = x
        coordinates['y'] = y
        coordinates['z'] = z
            

            
        
    def update(self, grid,t, force=False):


        self.it += 1

        
        if t >= self.next_output or t == 0 or force:
            if self.verbose:
                logging.info('Output at time: %f - iteration %d',t, self.it)

            
            self.save( grid, t)
            
            self.next_output += self.delta_t
            self.iteration += 1
        
        if t >= self.next_analysis or t == 0 or force:
            if self.verbose:
                logging.info('Analysis at time: %f - iteration %d',t, self.it)
                        
            self.next_analysis += self.delta_a

            self.analysis(grid, t)


        
        self.mpi_comm.barrier()
        
    
    
        
    def save(self, grid, t):
    
    
        for fname in self._fields_in_grid:

         #   grid[fname,0]=self.mpi_rank

         #   grid.sync_field(fname)

            
            u = grid.get_field(fname)            
            
            u_name = fname+'/'+str(self.iteration)
                
#            dset = self.h5file.create_dataset(u_name, data=u, compression="gzip", compression_opts=9 )
            dset = self.h5file.create_dataset(u_name, grid._gx[0].shape )
            index=grid._global_index[0]
            ii=index[0]
            fi=index[1]
            dset[ii[0]:fi[0],ii[1]:fi[1],ii[2]:fi[2]]=u 
            
            dset.attrs['time'] = t
                
            if self.debug:
                for level in np.arange(1,grid._mg_levels):
                    u = grid[fname,level]
            
                    u_name = fname+'_'+str(level)+'/'+str(self.iteration)
                
#                    dset = self.h5file.create_dataset(u_name, data=u, compression="gzip", compression_opts=9 )
                    dset = self.h5file.create_dataset(u_name, grid._gx[level].shape )
                    index=grid._global_index[level]
                    ii=index[0]
                    fi=index[1]
            
                    dset[ii[0]:fi[0],ii[1]:fi[1],ii[2]:fi[2]]=u 
                    dset.attrs['time'] = t
                    dset.attrs['level'] = level
                    
                                                        

    def analysis(self, grid, t):
        
        scalars = self.problem_analysis(grid, t)
        

        for fname in self._fields_in_grid:
 
            u = grid.get_field(fname)
            umax=np.zeros(1)                      
            
            self.mpi_comm.Reduce(u.max(),umax,op=MPI.MAX)
            umin=np.zeros(1)
            self.mpi_comm.Reduce(u.min(),umin,op=MPI.MIN)
            scalars[fname+'_max'] = umax[0]
            scalars[fname+'_min'] = umin[0]

        
        if self.mpi_rank == 0:
            
            fieldnames = ['t','wall_time']
        
            fieldnames.extend(list(scalars.keys()))
        
            scalars['t'] = t
            scalars['wall_time'] = time.time()-self.start_time

            
            
            with open(self.csv_file, 'a+') as f:
                dict_writer = csv.DictWriter(f, fieldnames=fieldnames,delimiter ='\t')
                if t==0:
                    f.write('#')
                    dict_writer.writeheader()
                dict_writer.writerow(scalars)           
                        
 


