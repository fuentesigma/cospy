'''
Created on 13 May 2015

@author: Jesus Fuentes and Pablo Galaviz
'''
import os 
import logging
import errno
import shutil
import numpy as np 
import svn.local
from mpi4py import MPI


def test_file_exist(data_file, must_exist=False):

    if os.path.exists(data_file):
        return True
    elif must_exist:
        logging.error('File %s does not exist!', data_file)
        exit(errno.ENOENT)
    else :
        return False
    
def diff_O2(f,dxyz,axis):
    return 0.5*(np.roll(f, 1, axis) - np.roll(f, -1, axis))/dxyz[axis]

def right_diff(f,dxyz,axis):
    return 0.5*( -3*f + 4* np.roll(f, 1, axis) - np.roll(f, 2, axis) )/(dxyz[axis]**2)

def left_diff(f,dxyz,axis):
    return 0.5*( 3*f - 4* np.roll(f, -1, axis) + np.roll(f, -2, axis) )/(dxyz[axis]**2)

def diff2_O2(f,dxyz,axis):
    return ( np.roll(f, 1, axis) - 2 * f + np.roll(f, -1, axis))/(dxyz**2)

def ddiff2_du_O2(f,dxyz,axis):
    return - 2 /(dxyz**2)

def lap_O2(f,dxyz,i=slice(None),j=slice(None),k=slice(None)):
    dx=dxyz[0][i,j,k]
    dy=dxyz[1][i,j,k]
    dz=dxyz[2][i,j,k]    
    return diff2_O2(f,dx, 0) + diff2_O2(f,dy, 1) + diff2_O2(f,dz, 2)

def dlap_du_O2(f,dxyz,i=slice(None),j=slice(None),k=slice(None)):
    dx=dxyz[0][i,j,k]
    dy=dxyz[1][i,j,k]
    dz=dxyz[2][i,j,k]    
    return ddiff2_du_O2(f,dx, 0) + ddiff2_du_O2(f,dy, 1) + ddiff2_du_O2(f,dz, 2)




def diff_O4(f,dxyz,axis):
  
    return (np.roll(f, -2, axis) + 8*( np.roll(f, 1, axis) - np.roll(f, -1, axis) ) - np.roll(f, 2, axis))/(12*dxyz[axis])
 


def diff2_O4(f,dxyz,axis):



    d=( -(np.roll(f, 2, axis) + np.roll(f, -2, axis)) 
        + 16 * (np.roll(f, 1, axis) + np.roll(f, -1, axis)) 
        - 30 * f  )/(12*dxyz**2)
    
    if np.any(np.isnan(d)):
        print(f)
        exit()

    return d 

def ddiff2_du_O4(f,dxyz,axis):
    d = -2.5 /dxyz**2
    return 2*d 

def lap_O4(f,dxyz,i=slice(None),j=slice(None),k=slice(None)):
    dx=dxyz[0][i,j,k]
    dy=dxyz[1][i,j,k]
    dz=dxyz[2][i,j,k]    
    return diff2_O4(f,dx, 0) + diff2_O4(f,dy, 1) + diff2_O4(f,dz, 2)

def dlap_du_O4(f,dxyz,i=slice(None),j=slice(None),k=slice(None)):
    dx=dxyz[0][i,j,k]
    dy=dxyz[1][i,j,k]
    dz=dxyz[2][i,j,k]    
    
    return ddiff2_du_O4(f,dx, 0) + ddiff2_du_O4(f,dy, 1) + ddiff2_du_O4(f,dz, 2)





def validate_and_make_directory(_directory):

    _directory = _directory.replace('//', '/')
    
    if os.path.exists(_directory):
        if not os.path.isdir(_directory):
            logging.error("The output %s is a regular file",_directory)
            exit(errno.EIO)
        else:
            if test_file_exist(_directory+'_prev'):
                shutil.rmtree(_directory+'_prev')
            shutil.move(_directory, _directory+'_prev')

    try:
        logging.info("Making directory %s.",_directory)
        os.makedirs(_directory)
        return _directory
    except:
        logging.exception('').replace('//', '/')
        exit(errno.EIO)

def expand_path(path, default):
    if path == None:
        path = os.path.abspath(os.path.curdir+'/'+default+'/').replace('//','/')

    if '~' in path:
        path = os.path.expanduser(path)

    if '$' in path:
        path = os.path.expandvars(path)
        
    return os.path.abspath(path)


def inject(grid, field1, field2, level):
    
    u_l = grid.get_multigrid_field(field1,level)
    v_lp1 = grid.get_multigrid_field(field2,level+1)

    v_lp1[::2,::2,::2] = u_l

    


def display_welcome():
    module_path = os.path.dirname(os.path.dirname(__file__))
    r = svn.local.LocalClient(module_path)
    info = r.info()
    
    logging.info("======================================")
    logging.info(" ,-----.                              ")
    logging.info("'  .--./ ,---.  ,---.  ,---.,--. ,--. ")
    logging.info("|  |    | .-. |(  .-' | .-. |\  '  /  ")
    logging.info("'  '--'\\' '-' '.-'  `)| '-' ' \   '   ")
    logging.info(" `-----' `---' `----' |  |-'.-'  /    ")
    logging.info("                      `--'  `---'     ")
    logging.info("============ Cospy  15.01 ============")
    logging.info("")
    logging.info("")
    logging.info("  Authors: J. Fuentes and P. Galaviz  ")
    logging.info("")
    logging.info("")
    logging.info("======================================")
    logging.info("")
    logging.info("svn revision: "+str(info['entry_revision']))
    logging.info("")

    

    
    
    
    
def overwrite_settings(config_data,arg_data):
    
        
    for k,v in arg_data.__dict__.items():
        kk = k.replace('_',' ')
        if v != None and kk in config_data: 
            config_data.update({kk:str(v)})
            
    return config_data
    

