'''
Created on 15 May 2015

@author: Pablo Galaviz
'''
import logging
import cospy.grid_cpu as cp_grid_cpu
import cospy.grid_mpi as cp_grid_mpi
import cospy.grid_gpu as cp_grid_gpu

def Grid( params):
    
    logging.info("---------- grid setup ----------")
    
    grid_type =  params.get('grid type','CPU_grid')

    if grid_type == 'CPU_grid':
        grid = cp_grid_cpu.Grid(params)
    elif grid_type == 'MPI_grid':
        grid = cp_grid_mpi.Grid(params)
    elif grid_type == 'GPU_grid':
        grid = cp_grid_gpu.Grid(params)
    else:
        logging.error("Undefined grid type")
        exit(1)

    return grid


        