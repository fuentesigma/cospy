'''
Created on 18 May 2015

@author: Jesus Fuentes and Pablo Galaviz
'''
import logging
import numpy as np 
from numpy import exp, sin, cos, tan 
import cospy.utils as cp_utils
from mpi4py import MPI

class Grid(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''

        mpi_comm = MPI.COMM_WORLD
        mpi_size = mpi_comm.Get_size() # total number of nodes

        if mpi_size > 1:
            raise Exception('cospy with cpu grid running in %d cores. Change to mpi grid'%mpi_size)
            exit()

            
        self.diff_mg = cp_utils.diff_O2
        self.diff2_mg = cp_utils.diff2_O2
        self.lapC2_mg = cp_utils.lap_O2
        self.dlap_duC2_mg = cp_utils.dlap_du_O2

        
        diff_order = params.getint('finite difference order',2)
        
        if diff_order == 2:
            self.diff = cp_utils.diff_O2
            self.diff2 = cp_utils.diff2_O2
            self.lapC2 = cp_utils.lap_O2
            self.dlap_duC2 = cp_utils.dlap_du_O2
            
            logging.info('Using second order finite difference')
            
        elif diff_order == 4:
            self.diff = cp_utils.diff_O4
            self.diff2 = cp_utils.diff2_O4
            self.lapC2 = cp_utils.lap_O4
            self.dlap_duC2 = cp_utils.dlap_du_O4

            logging.info('Using fourth order finite difference')
        else:
            raise Exception('Finite difference order %d not  implemented'%self.diff_order)
            exit()
      
        
        self.boundary_gz = self.boundary_ghost_zones(diff_order)
        
        nx = params.getint('grid size x')
        ny = params.getint('grid size y',nx)
        nz = params.getint('grid size z',ny)

        self._mgl, self._n  = self.validate_multigrid(nx,ny,nz)
        
        self._mg_levels = len(self._mgl[0])

        logging.info('Multigrid size:   nx=%d, ny=%d, nz=%d',self._n[0],self._n[1], self._n[2])

        logging.info('Multigrid levels: %d',self._mg_levels)
        self._multigrid_fields = list()

        lx = params.getfloat('domain size x')
        ly = params.getfloat('domain size y',lx)
        lz = params.getfloat('domain size z',ly)
         
        self._l = np.array([lx,ly,lz])
         
        logging.info('Domain size: lx=%.3f, ly=%.3f, lz=%.3f',self._l[0],self._l[1], self._l[2])

        self._field_index=dict()
        self._fields = list()

        self._scalars = dict()

        self._x = list()
        self._y = list()
        self._z = list()

        self._dxyz = list()
        self._boundary_indx = list()
        self._left_boundary_indx = list()
        self._right_boundary_indx = list()

        for level in np.arange(self._mg_levels):
            x = np.linspace(-self._l[0]/2, self._l[0]/2, self._mgl[0,level], endpoint=True)
            y = np.linspace(-self._l[1]/2, self._l[1]/2, self._mgl[1,level], endpoint=True)
            z = np.linspace(-self._l[2]/2, self._l[2]/2, self._mgl[2,level], endpoint=True)

            _x, _y, _z = np.meshgrid(x,y,z,indexing='ij')
            self._x.append(_x)
            self._y.append(_y)
            self._z.append(_z)

            dx = np.ones_like(_x)*lx/(self._mgl[0,level]-1)
            dy = np.ones_like(_y)*ly/(self._mgl[1,level]-1)
            dz = np.ones_like(_z)*lz/(self._mgl[2,level]-1)

            self._dxyz.append(np.array([dx, dy, dz] ) )
 
            if level ==0:
                b_indx=np.zeros([3,2*self.boundary_gz], dtype=int)
                b_indx[:,:self.boundary_gz] = np.arange(self.boundary_gz)
                b_indx[:,self.boundary_gz:] = np.array(self._mgl[:,level]).reshape([3,1]) - np.arange(self.boundary_gz)-1
            else:
                b_indx=np.zeros([3,2], dtype=int)
                b_indx[:,:1] = np.arange(1)
                b_indx[:,1:] = np.array(self._mgl[:,level]).reshape([3,1]) - np.arange(1)-1
                    
            self._boundary_indx.append(b_indx)
 
            lb_indx=np.zeros([3,3*self.boundary_gz], dtype=int)            
            lb_indx[:,:] = np.arange(3*self.boundary_gz)
            self._left_boundary_indx.append(lb_indx)

            rb_indx=np.zeros([3,3*self.boundary_gz], dtype=int)            
            rb_indx[:,:] = np.array(self._mgl[:,level]).reshape([3,1]) - np.arange(3*self.boundary_gz)
            self._right_boundary_indx.append(rb_indx)

            
#        print(self._boundary_indx)
#        exit()
#            self._dxyz.append(np.array([l/(n-1.0) for l,n in zip(self._l, self._mgl[:,level]) ] ) )
        self.number_of_mol_variables = 0
         

    def boundary_ghost_zones(self,order):
        return  int(order/2)
        

    def set_scalar(self, name,value):
        self._scalars[name] = value

    def get_scalar(self, name):
        return self._scalars[name] 
    
    def getall_scalar(self):
        return self._scalars
    

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
        


    def make_grid(self, fields):
        
        
        if len(fields) == 0 :
            return 
    
        logging.info("Making grid")
        logging.info("Fields in grid: %s",','.join( fields.keys()))
        
        num_of_fields=0
        self.hyperbolic_fields=list()
        self.elliptic_fields=list()
        self.parameter_fields=list()
        
        for k, v in fields.items():
            if v == 'hyperbolic':
                self.hyperbolic_fields.append(k)
            elif v == 'elliptic':
                self.elliptic_fields.append(k)
            elif v == 'parameter':
                self.parameter_fields.append(k)
        
        for f in self.hyperbolic_fields:
            num_of_fields = self._add_field(num_of_fields,name=f)

        for f in self.elliptic_fields:
            num_of_fields = self._add_field(num_of_fields,name=f)
            num_of_fields = self._add_field(num_of_fields,name=f+'_omega')
            num_of_fields = self._add_field(num_of_fields,name=f+'_source')
            num_of_fields = self._add_field(num_of_fields,name=f+'_V')

        for f in self.parameter_fields:
            num_of_fields = self._add_field(num_of_fields,name=f)


        for level in range(self._mg_levels):
            self._fields.append(np.zeros([num_of_fields,self._mgl[0][level],self._mgl[1][level],self._mgl[2][level]]))


        self.number_of_mol_variables = self._n[0]*self._n[1]*self._n[2]*len(self.hyperbolic_fields)
        self.shape = np.array([len(self.hyperbolic_fields),self._n[0],self._n[1],self._n[2]] )


    def has_field(self,fieldname):
        return  fieldname in self._field_index.keys()

        
    def get_mol_variables(self):
        return self._fields[0][:self.shape[0],:,:,:].ravel()
#        return self._fields[0].ravel()[:self.number_of_mol_variables]

    def _add_field(self,field_index, name='' ):

        if name in self._field_index.keys():
            logging.warning('Field %s already exists in amr grid', name)
            return field_index

        self._field_index[name]=field_index
        return field_index + 1
    
            
    def get_type(self):
        return 'cpu grid'
        
    def __getitem__(self, args):
        
        if isinstance(args, str):
            narg=1
        else:
            narg=len(args)
            
        if narg > 5:
            raise Exception('Error trying to access grid. Wrong number of arguments: %d. Number of arguments should be less than 5 '%narg)

        if narg == 1:
            return self.getitem_1(args)

        if narg == 2:
            return self.getitem_2(*args)

        if narg == 4:
            return self.getitem_4(*args)

        if narg == 5:
            return self.getitem_5(*args)

        else :
            raise Exception('3 arguments not supported')

    def getitem_1(self,name):
        return self._fields[0][self._field_index[name]][:,:,:]
        
    def getitem_2(self,name,l):
        return self._fields[l][self._field_index[name]][:,:,:]
        
    def getitem_4(self,i,j,k,name):
        return self.getitem_5(i,j,k,name,0)

    def getitem_5(self,i,j,k,name,l):
        i = self.parse_index(i,l,0)
        j = self.parse_index(j,l,1)
        k = self.parse_index(k,l,2)  
        return self._fields[l][self._field_index[name]][i,j,k]
        
    def parse_index(self,i,l=0,axis=0):
        if i == 'b':
            return self._boundary_indx[l][axis]
            
        if i == 'lb':
            indx=np.zeros(2*self.boundary_gz, dtype=int)
            for i in np.arange(self.boundary_gz):
                indx[i]=i
                indx[self.boundary_gz-i-1]= self._mgl[axis,l]-i-1 
            return indx
       
        if i == 'i':
            if l == 0:
                return slice(self.boundary_gz,self._mgl[axis,l]-self.boundary_gz,1)
            return slice(1,self._mgl[axis,l]-1,1)


        return i
      
       
       
    def __setitem__(self, args, val):

        if isinstance(args, str):
            narg=1
        else:
            narg=len(args)
    
        
        if narg > 5:
            raise Exception('Error trying to access grid. Wrong number of arguments: %d. Number of arguments should be less than 5 '%narg)

        if narg == 1:
            return self.setitem_1(args,val=val)

        if narg == 2:
            return self.setitem_2(*args,val=val)

        if narg == 4:
            return self.setitem_4(*args,val=val)

        if narg == 5:
            return self.setitem_5(*args,val=val)

        else :
            raise Exception('3 arguments not supported')

    def setitem_1(self,name,val):
        self._fields[0][self._field_index[name]][:,:,:] = val
        
        
        
    def setitem_2(self,name,l,val):
        self._fields[l][self._field_index[name]][:,:,:] = val
        
    def setitem_4(self,i,j,k,name,val):
        self.setitem_5(i,j,k,name,0,val)

    def setitem_5(self,i,j,k,name,l,val):
        i = self.parse_index(i,l,0)
        j = self.parse_index(j,l,1)
        k = self.parse_index(k,l,2)     
        self._fields[l][self._field_index[name]][i,j,k] = val


        
    def get_field(self,name, level=0):

        if self.has_field(name):        
            if level < self._mg_levels:
                return self._fields[level][self._field_index[name]][:,:,:]
            logging.error('The grid has %d levels, error trying to access level %d', self._mg_levels, level)
                
        logging.error('Field %s does not exists', name)

    def get_coordinates(self,level=0):
        return self._x[level], self._y[level], self._z[level]

    def get_dxyz(self,level=0):
        return self._dxyz[level]

    
    def set_field(self,name, funct, *args):
        
        kwargs = dict()
        
        for arg in args:
            kwargs[arg] = self.parse_arg(arg)
                                                
        self._fields[0][self._field_index[name]][:,:,:] = funct(**kwargs)
        
    def set_mol_fields(self,y):
        self._fields[0][:self.shape[0],:,:,:] = y.reshape(self.shape)
        
        

    def get_rhs(self,rhs):

                                                                
        f = np.zeros(self.shape)
        
        for field in  self.hyperbolic_fields:
                                                                
            f[self._field_index[field],:,:,:] = rhs[field]

        return f 
    
    def parse_arg(self, arg,level=0):

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
                return self.diff(self.get_field(f,level),self._dxyz[level],axis=0)
            if arg == 'd'+f+'dy':
                return self.diff(self.get_field(f,level),self._dxyz[level],axis=1)
            if arg == 'd'+f+'dz':
                return self.diff(self.get_field(f,level),self._dxyz[level],axis=2)

    
            if arg == 'd'+f+'ddx':
                return self.diff2(self.get_field(f,level),self._dxyz[level],axis=0)
            if arg == 'd'+f+'ddy':
                return self.diff2(self.get_field(f,level),self._dxyz[level],axis=1)
            if arg == 'd'+f+'ddz':
                return self.diff2(self.get_field(f,level),self._dxyz[level],axis=2)
        
            if arg == 'Lap_'+f:
                return self.lap(self.get_field(f,level),self._dxyz[level])

        logging.error('argument not implemented in grid_cpu parse_arg: %s',arg)
        exit(1)



    def boundary(self, _f,field, kind, value = None):
        
        if kind == 'Periodic':
            _f[0,:,:,self._field_index[field]]  =  _f[-2,:,:,self._field_index[field]]
            _f[-1,:,:,self._field_index[field]] =  _f[1, :,:,self._field_index[field]]

            _f[:,0,:,self._field_index[field]]  =  _f[:,-2,:,self._field_index[field]]
            _f[:,-1,:,self._field_index[field]] =  _f[:, 1,:,self._field_index[field]]

            _f[:,:,0,self._field_index[field]]  =  _f[:,-2,:,self._field_index[field]]
            _f[:,:,-1,self._field_index[field]] =  _f[:,: ,1,self._field_index[field]]
            
            return _f
        
        if kind == 'Dirichlet':
                        
            _f[0,:,:,self._field_index[field]] =  value
            _f[-1,:,:,self._field_index[field]] =  value

            _f[:,0,:,self._field_index[field]] =  value
            _f[:,-1,:,self._field_index[field]] =  value

            _f[:,:,0,self._field_index[field]] =  value
            _f[:,:,-1,self._field_index[field]] =  value

            return _f
   
        logging.error('boundary not implemented in grid_cpu boundary: %s',kind)
        exit(1)

    
    
    def lap(self, *args):
        if isinstance(args, str):
            narg=1
        else:
            narg=len(args)
            
        if narg > 5:
            raise Exception('Error trying to access grid. Wrong number of arguments: %d. Number of arguments should be less than 5 '%narg)

        if narg == 1:
            return self.lap_1(*args)

        if narg == 2:
            return self.lap_2(*args)

        if narg == 4:
            return self.lap_4(*args)

        if narg == 5:
            return self.lap_5(*args)

        else :
            raise Exception('3 arguments not supported')

    def lap_1(self,name):
        return self.lapC2(self._fields[0][self._field_index[name]][:,:,:], self._dxyz[0])
        
    def lap_2(self,name,l):
        if l == 0:
            return self.lapC2(self._fields[l][self._field_index[name]][:,:,:], self._dxyz[l])
        return self.lapC2_mg(self._fields[l][self._field_index[name]][:,:,:], self._dxyz[l])
        
    def lap_4(self,i,j,k,name):
        i = self.parse_index(i)
        j = self.parse_index(j)
        k = self.parse_index(k)
        return self.lap_5(i,j,k,name,0)

    def lap_5(self,i,j,k,name,l):
        i = self.parse_index(i,l)
        j = self.parse_index(j,l)
        k = self.parse_index(k,l)  
        if l == 0:  
            return self.lapC2(self._fields[l][self._field_index[name]][i,j,k], self._dxyz[l], i,j,k)
        
        return self.lapC2_mg(self._fields[l][self._field_index[name]][i,j,k], self._dxyz[l], i,j,k)

    def dlap_du(self, *args):

        if isinstance(args, str):
            narg=1
        else:
            narg=len(args)
    

        if narg > 5:
            raise Exception('Error trying to access grid. Wrong number of arguments: %d. Number of arguments should be less than 5 '%narg)

        if narg == 1:
            return self.dlap_du_1(args)

        if narg == 2:
            return self.dlap_du_2(*args)

        if narg == 4:
            return self.dlap_du_4(*args)

        if narg == 5:
            return self.dlap_du_5(*args)

        else :
            raise Exception('3 arguments not supported')

    def dlap_du_1(self,name):
        return self.dlap_duC2(self._fields[0][self._field_index[name]][:,:,:], self._dxyz[0])
        
    def dlap_du_2(self,name,l):
        if l ==0:
            return self.dlap_duC2(self._fields[l][self._field_index[name]], self._dxyz[l])
        return self.dlap_duC2_mg(self._fields[l][self._field_index[name]], self._dxyz[l])
        
    def dlap_du_4(self,i,j,k,name):
        i = self.parse_index(i)
        j = self.parse_index(j)
        k = self.parse_index(k)
        return self.dlap_du_5(i,j,k,name,0)

    def dlap_du_5(self,i,j,k,name,l):
        i = self.parse_index(i,l)
        j = self.parse_index(j,l)
        k = self.parse_index(k,l)
        if l==0:        
            return self.dlap_duC2(self._fields[l][self._field_index[name]][i,j,k], self._dxyz[l])

        return self.dlap_duC2_mg(self._fields[l][self._field_index[name]][i,j,k], self._dxyz[l])

    def injection(self,field_h, level, field_2h):

        self._fields[level+1][self._field_index[field_2h]][:,:,:] = self._fields[level][self._field_index[field_h]][::2,::2,::2]
        
        
    def prolong(self,field_2h, level, field_h):

        u = self._fields[level][self._field_index[field_2h]][:,:,:]

        self._fields[level-1][self._field_index[field_h]][::2,::2,::2] =  u 
        u_ip1=np.roll(u,1,0)
        u_jp1=np.roll(u,1,1)
        u_kp1=np.roll(u,1,2)

        u_ijp1 = np.roll(u_ip1,1,1)
        u_ikp1 = np.roll(u_ip1,1,2)
        u_jkp1 = np.roll(u_jp1,1,2)

        u_ijkp1 = np.roll(u_ijp1,1,2)


 
        self._fields[level-1][self._field_index[field_h]][1::2,::2,::2] = 0.5*(u+u_ip1)[1:,:,:]
        self._fields[level-1][self._field_index[field_h]][::2,1::2,::2] = 0.5*(u+u_jp1)[:,1:,:] 
        self._fields[level-1][self._field_index[field_h]][::2,::2,1::2] = 0.5*(u+u_kp1)[:,:,1:]

 
    
        self._fields[level-1][self._field_index[field_h]][1::2,1::2,::2] =  0.25*(u+u_ip1+u_jp1+u_ijp1)[1:,1:,:] 
        self._fields[level-1][self._field_index[field_h]][1::2,::2,1::2] = 0.25*(u+u_ip1+u_kp1+u_ikp1)[1:,:,1:]
        self._fields[level-1][self._field_index[field_h]][::2,1::2,1::2] = 0.25*(u+u_jp1+u_kp1+u_jkp1)[:,1:,1:]

 
        self._fields[level-1][self._field_index[field_h]][1::2,1::2,1::2] = 0.125*(u+u_ip1+u_jp1+u_kp1+u_ijp1+u_ikp1+u_jkp1+u_ijkp1)[1:,1:,1:]

 
 
 
        
