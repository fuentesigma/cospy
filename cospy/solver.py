'''
Created on 15 May 2015

@author: Jesus Fuentes and Pablo Galaviz
'''
from equations import * 
import logging
from mpi4py import MPI

import pygsl._numobj as Numeric
import pygsl
from pygsl import  odeiv, Float

import cospy.elliptic as cp_elliptic

import numpy as np

class Solver(object):
    '''
    classdocs
    '''


    def __init__(self, params,multigrid_params):
        '''
        Constructor
        '''


        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_size = self.mpi_comm.Get_size() # total number of nodes
        self.mpi_rank = self.mpi_comm.Get_rank() # current node 

        
        logging.info("---------- problem setup ----------")

        self.problem_name = params.get('problem')
        
        logging.info("Problem name: %s", self.problem_name)
        
        self.ft = params.getfloat('final time', 1)
        logging.info("Final evolution time: %s", self.ft)
        
        self.eq_source = self.problem_name.replace(' ', '_')
        
        if self.eq_source == None:
            logging.error("Equation source for %s are undefined.",self.problem_name)
            logging.error("Please provide the source in the parameters file.")
            exit(1)

        self.problem = eval ('cospy.'+self.eq_source+'.Problem(params)')
        
        self.multigrid_params = multigrid_params
                
        self.t = 0  
        self.dt = 0.01  
        self.status = None
            
    def initial_data(self, grid):

        fields=self.problem.get_fields()

        
        grid.make_grid(fields)
        

        self.problem.set_initial_data(grid)

        
 
        dimension=grid.number_of_mol_variables
        
        stepper = odeiv.step_rk8pd
        
        step = stepper(dimension, self.problem.rhs,self.problem.rhs_jac, grid)
        control = odeiv.control_y_new(step, 1e-6, 1e-6)
        
        self.evolve  = odeiv.evolve(step, control, dimension)
        
        self.status = 'initial data done'
        
        self.multigrid = cp_elliptic.Elliptic(grid, self.problem.mg_problem,params=self.multigrid_params)
                
        self.multigrid.solve()
        
        
        return self.t, self.status
    
    
    def update(self,grid):
        
        logging.info('Time %f', self.t )


        self.t, self.dt, y = self.evolve.apply(self.t,self.ft , self.dt, grid.get_mol_variables())

        logging.debug(" dt = %f", self.dt)

        self.multigrid.solve()

        
        self.status = 'continue'

        
        if self.t >= self.ft:
            self.status = 'done'
        
        
        return self.t, self.status

        
    
