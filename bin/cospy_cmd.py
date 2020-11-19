#!/usr/bin/env python2.7

'''
Created on 13 May 2015

@author: Pablo Galaviz
'''

import shutil
import argparse
import configparser
import logging
import sys
import cospy.utils as cp_utils
import errno
import cospy.grid as cp_grid
import cospy.output as cp_output
import cospy.solver as cp_solver
from mpi4py import MPI
import time

if __name__ == '__main__':

    start_time = time.time()
    
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size() # total number of nodes
    mpi_rank = mpi_comm.Get_rank() # current node 
    mpi_root = 0 

    
    if mpi_size > 1:
        logFormatter = logging.Formatter('Cospy %(levelname)s [%(asctime)s] cpu '+str(mpi_rank)+' | %(message)s')
    else:
        logFormatter = logging.Formatter('Cospy %(levelname)s [%(asctime)s] | %(message)s')

    rootLogger = logging.getLogger()
    
    rootLogger.setLevel(logging.INFO)
    
    epilog_text=" =============================\n     Author: Pablo Galaviz    \n     pablo.galaviz@me.com  \n =============================\n   License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html> \n   This is free software: you are free to change and redistribute it.\n   There is NO WARRANTY, to the extent permitted by law."

    

    parser = argparse.ArgumentParser(description='Cospy, cosmology amr solver', epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('input_file', help='Input parameter file.',metavar='intput.par')
    parser.add_argument('-s','--silent',action='store_false' ,help='Starts in silent mode, no message will be output.')
    
    parser.add_argument('-nx','--grid_size_x',type=int ,help='Overwrites grid size x')
    parser.add_argument('-ny','--grid_size_y',type=int ,help='Overwrites grid size y')
    parser.add_argument('-nz','--grid_size_z',type=int ,help='Overwrites grid size z')

    parser.add_argument('-lx','--domain_size_x',type=float ,help='Overwrites domain size x')
    parser.add_argument('-ly','--domain_size_y',type=float ,help='Overwrites domain size y')
    parser.add_argument('-lz','--domain_size_z',type=float ,help='Overwrites domain size z')

    parser.add_argument('--project_directory',type=str ,help='Overwrites project directory')
    parser.add_argument('--problem',type=str ,help='Overwrites problem')
    parser.add_argument('-ft','--final_time',type=float ,help='Overwrites final time')

    parser.add_argument('--pre_cycles',type=int ,help='Overwrites pre cycles')
    parser.add_argument('--post_cycles',type=int ,help='Overwrites post cycles')

    parser.add_argument('-fgt','--fine_grid_tolerance',type=float ,help='Overwrites fine grid tolerance')
    parser.add_argument('-cgt','--coarse_grid_tolerance',type=float ,help='Overwrites coarse grid tolerance')

    parser.add_argument('-fdo','--finite_difference_order',type=int ,help='Overwrites finite difference order')
    parser.add_argument('-do','--delta_output',type=float ,help='Overwrites delta output')

    parser.add_argument('-d','--debug_output',action='store_true' ,help='Output debug log')

    args = parser.parse_args()
    config = configparser.ConfigParser()

    if args.debug_output:
        rootLogger.setLevel(logging.DEBUG)


    if  args.silent and mpi_rank == mpi_root:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)


    
    if cp_utils.test_file_exist(args.input_file, must_exist=True):
        config.read(args.input_file, encoding='utf-8')
    
    try:

        grid_settings=cp_utils.overwrite_settings(config['Grid'],args)
        problem_settings=cp_utils.overwrite_settings(config['Problem'],args)
        project_settings=cp_utils.overwrite_settings(config['Project'],args)
        output_settings=cp_utils.overwrite_settings(config['Output'],args)
        multigrid_settings=cp_utils.overwrite_settings(config['Multigrid'],args)
    except:
        logging.exception("Not valid parameter file")
        exit(errno.EINVAL)
     
               
    project_directory = cp_utils.expand_path(project_settings.get('project directory'), 'results') 
    
    if mpi_rank == mpi_root:
        cp_utils.validate_and_make_directory(project_directory)
    
        shutil.copy(args.input_file, project_directory )

    mpi_comm.Barrier()  
    if mpi_size > 1:
        fileHandler = logging.FileHandler(project_directory+"/Cospy_cpu"+str(mpi_rank)+".log",mode='w')
    else:
        fileHandler = logging.FileHandler(project_directory+"/Cospy.log",mode='w')
        
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    cp_utils.display_welcome()        
    
    logging.info('Project directory: %s', project_directory)



    grid = cp_grid.Grid(grid_settings)
    
        
    solver = cp_solver.Solver(problem_settings,multigrid_settings)
    output = cp_output.Output(output_settings, project_directory)
    
     

    t, status = solver.initial_data(grid)
    output.init(grid,solver.problem)
    

    status = 'continue'

    while(status == 'continue'):
        
        output.update(grid,t)
        t, status = solver.update(grid)


    output.update(grid,t, force=True)

    total_time = time.time()-start_time 


    logging.info("All done. Total execution time: "+str(total_time))



