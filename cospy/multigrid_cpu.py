'''
Created on 22 Jun 2015

@author: pablo
'''

import logging
import numpy as np 
from numpy import exp, sin, cos, tan 
import cospy.utils as cp_utils

class Multigrid(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        nx = params.getint('nx')
        ny = params.getint('ny',nx)
        nz = params.getint('nz',ny)

        self._mgl, self._n  = self.validate_multigrid(nx,ny,nz)
        
        self._mg_levels = len(self._mgl[0])

        logging.info('Multigrid size:   nx=%d, ny=%d, nz=%d',self._n[0],self._n[1], self._n[2])

        logging.info('Multigrid levels: %d',self._mg_levels)
        self._multigrid_fields = list()

        lx = params.getfloat('lx')
        ly = params.getfloat('ly',lx)
        lz = params.getfloat('lz',ly)
         
        self._l = np.array([lx,ly,lz])
         
        logging.info('Domain size: lx=%d, ly=%d, lz=%d',self._l[0],self._l[1], self._l[2])

        self._field_index=dict()
        self._fields = list()

        self._x = list()
        self._y = list()
        self._z = list()

        self._dxyz = list()

        for level in np.arange(self._mg_levels):
            x = np.linspace(-self._l[0]/2, self._l[0]/2, self._mgl[0,level], endpoint=True)
            y = np.linspace(-self._l[1]/2, self._l[1]/2, self._mgl[1,level], endpoint=True)
            z = np.linspace(-self._l[2]/2, self._l[2]/2, self._mgl[2,level], endpoint=True)

            _x, _y, _z = np.meshgrid(x,y,z,indexing='ij')
            self._x.append(_x)
            self._y.append(_y)
            self._z.append(_z)

            self._dxyz.append(np.array([l/(n-1.0) for l,n in zip(self._l, self._mgl[:,level]) ] ) )

         

    def make_grid(self, fields):
        
        
        if len(fields) == 0 :
            return 
        
        logging.info("Making multigrid")
        logging.info("Fields in grid: %s",','.join(fields))
        
        num_of_fields=0
        
        for k in fields:
            num_of_fields = self._add_field(num_of_fields,name=k)                

        for level in range(self._mg_levels):
            self._fields.append(np.zeros([self._mgl[0][level],self._mgl[1][level],self._mgl[2][level],num_of_fields]))


    def get_grid_parameters(self):
        return self._n, self._l


    def validate_multigrid(self, nx,ny,nz):

        nx = self.clip_grid_size(nx)
        ny = self.clip_grid_size(ny)
        nz = self.clip_grid_size(nz)
        
        mglx = self.get_multigrid_levels(nx)
        mgly = self.get_multigrid_levels(ny)
        mglz = self.get_multigrid_levels(nz)
        
        levels = np.min([len(mglx),len(mgly),len(mglz)])
        
        return np.array([mglx[:levels], mgly[:levels], mglz[:levels]]), np.array([nx, ny, nz])

    def clip_grid_size(self,n):
        
        valid_grid=np.array([3,     4,     5,     6,     7,     8,     9,     10, 
                            11,    12,    13,    14,    15,    17,    19,    21,
                            23,    25,    27,    29,    33,    37,    41,    45, 
                            49,    53,    57,    65,    73,    81,    89,    97, 
                            105,   113,   129,   145,   161,   177,   193,   209, 
                            225,   257,   289,   321,   353,   385,   417,   449, 
                            513,   577,   641,   705,   769,   833,   897,   1153, 
                            1281,  1409,  1665,  1793,  2305,  2817,  3329])
        indx = np.abs(valid_grid-n).argmin()
        return valid_grid[indx]
            

    def get_multigrid_levels(self,n):
        
        mgl=list()
        mgl.append(n)
        while True:
            n_new = (n+1)/2
            if n_new > 10:
                mgl.append(n_new)
                n=n_new
            else:
                break
        return mgl
        


    def _add_field(self,field_index, name='' ):

        if name in self._field_index.keys():
            logging.warning('Field %s already exists in amr grid', name)
            return field_index
        self._field_index[name]=field_index
        return field_index + 1


    def get_type(self):
        return 'cpu multigrid'

    def get_field_list(self):
        return list(self._field_index.keys())


    def get_field(self,name, level=0):

        if self.has_field(name):        
            if level < self._mg_levels:
                return self._fields[level][:,:,:,self._field_index[name]]
            logging.error('The grid has %d levels, error trying to access level %d', self._mg_levels, level)
                
        logging.error('Field %s does not exists', name)

    def get_coordinates(self,level=0):
        return self._x[level], self._y[level], self._z[level]

    def get_dxyz(self,level=0):
        return self._dxyz[level]

    def has_field(self,fieldname):
        return  fieldname in self._field_index.keys()



    def set_field(self,name, level, value):
        if level < self._mg_levels:
            self._fields[level][:,:,:,self._field_index[name]] = value
        else:
            logging.error('The grid has %d levels, error trying to access level %d', self._mg_levels, level)



    def parse_arg(self, arg, level):

        fields = self._field_index.keys()

        if arg == 'x':
            return self._x[level]
        if arg == 'y':
            return self._y[level]
        if arg == 'z':
            return self._z[level]

        for f in fields:

            if arg == f:
                return self.get_field(f,level)

            if arg == 'd'+f+'dx':
                return cp_utils.diff(self.get_field(f,level),self._dxyz[level],axis=0)
            if arg == 'd'+f+'dy':
                return cp_utils.diff(self.get_field(f,level),self._dxyz[level],axis=1)
            if arg == 'd'+f+'dz':
                return cp_utils.diff(self.get_field(f,level),self._dxyz[level],axis=2)

    
            if arg == 'd'+f+'ddx':
                return cp_utils.diff2(self.get_field(f,level),self._dxyz[level],axis=0)
            if arg == 'd'+f+'ddy':
                return cp_utils.diff2(self.get_field(f,level),self._dxyz[level],axis=1)
            if arg == 'd'+f+'ddz':
                return cp_utils.diff2(self.get_field(f,level),self._dxyz[level],axis=2)
        
            if arg == 'Lap_'+f:
                return cp_utils.lap(self.get_field(f,level),self._dxyz[level])

        logging.error('argument not implemented in grid_cpu parse_arg: %s',arg)
        exit(1)






