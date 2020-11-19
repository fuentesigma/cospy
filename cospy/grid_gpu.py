'''
Created on 18 May 2015

@author: Pablo Galaviz
'''
import logging
import pyopencl 

class Amr_Grid(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
        self._nx = params.getint('_nx')
        self._ny = params.getint('_ny',self._nx)
        self._nz = params.getint('_nz',self._ny)
                    
        logging.info('Amr_Grid size:   _nx=%d, _ny=%d, _nz=%d',self._nx,self._ny, self._nz)

        self._lx = params.getfloat('_lx')
        self._ly = params.getfloat('_ly',self._lx)
        self._lz = params.getfloat('_lz',self._ly)
         
        logging.info('Domain size: _lx=%d, _ly=%d, _lz=%d',self._lx,self._ly, self._lz)


        self.amr_fields=dict()
        self.multigrid_fields=dict()

    def add_field(self,name='', grid_type='amr'):

        if grid_type == 'amr':
            self.add_amr_field(name)
        elif grid_type == 'multigrid':
            self.add_multigrid_field(name)
            
        else:
            logging.error("Wrong type $s for field %s", grid_type, name)
            
            
    def add_amr_field(self,name):
        
        raise Exception('Not implemented')
        

    def add_multigrid_field(self,name):
        
        raise Exception('Not implemented')


    def get_type(self):
        return 'gpu grid'