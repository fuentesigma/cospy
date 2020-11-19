'''
Created on 18 May 2015

@author: Pablo Galaviz
'''
import logging
from mpi4py import MPI
import numpy as np 
import cospy.utils as cp_utils 

class Grid(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_size = self.mpi_comm.Get_size() # total number of nodes
        self.mpi_rank = self.mpi_comm.Get_rank() # current node 

        logging.info('MPI grid - cores:%d',self.mpi_size)

        self.diff_mg = cp_utils.diff_O2
        self.diff2_mg = cp_utils.diff2_O2
        self.lapC2_mg = cp_utils.lap_O2
        self.dlap_duC2_mg = cp_utils.dlap_du_O2


        self.cpu_part = self.get_partition()
        self.partition_index = self.get_partition_index()
        

        logging.info('CPU partition:')
        logging.info('======================================')
        logging.info('axis |\tx\ty\tz')
        logging.info('     |--------------------------------')
        logging.info('cores|\t%d\t%d\t%d',self.cpu_part[0],self.cpu_part[1],self.cpu_part[2])
        logging.info('======================================')


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

        logging.info('Local multigrid size:   nx=%d, ny=%d, nz=%d',self._n[0],self._n[1], self._n[2])
        logging.info('Global multigrid size:   nx=%d, ny=%d, nz=%d',
                    self._n[0]*self.cpu_part[0],
                    self._n[1]*self.cpu_part[1],
                    self._n[2]*self.cpu_part[2])

        logging.info('Multigrid levels: %d',self._mg_levels)
#        self._multigrid_fields = list()


        lx = params.getfloat('domain size x')
        ly = params.getfloat('domain size y',lx)
        lz = params.getfloat('domain size z',ly)

        self._l = np.array([lx,ly,lz])
         
        logging.info('Domain size: lx=%.3f, ly=%.3f, lz=%.3f',self._l[0],self._l[1], self._l[2])


        self._local_bound = self.set_grid_partition()
        
        logging.info('Local boundary:')
        logging.info('======================================')
        logging.info('x-->[%.3e, %.3e]',
                     self._local_bound[0,0],
                     self._local_bound[0,1])
        logging.info('y-->[%.3e, %.3e]',
                     self._local_bound[1,0],
                     self._local_bound[1,1])
        logging.info('z-->[%.3e, %.3e]',
                     self._local_bound[2,0],
                     self._local_bound[2,1])
 
        self._boundary_type = self.get_boundary_type()


        self._field_index=dict()
        self._fields = list()

        self._scalars = dict()

        self._x = list()
        self._y = list()
        self._z = list()

        self._gx = list()
        self._gy = list()
        self._gz = list()

        self._dxyz = list()
        self._boundary_indx = list()
        self._cpu_boundary_indx = list()

        self._global_index=list()

        for level in np.arange(self._mg_levels):

            if level > 0: 
                bzone=2
            else:
                bzone=2*self.boundary_gz
                
            gN=self._mgl[:,level]+(self.cpu_part-1)*(self._mgl[:,level]-bzone)
    
            
            gx = np.linspace(-0.5*self._l[0],0.5*self._l[0], gN[0], endpoint=True)
            gy = np.linspace(-0.5*self._l[1],0.5*self._l[1], gN[1], endpoint=True)
            gz = np.linspace(-0.5*self._l[2],0.5*self._l[2], gN[2], endpoint=True)

            _gx, _gy, _gz = np.meshgrid(gx,gy,gz,indexing='ij')
            
            self._gx.append(_gx)
            self._gy.append(_gy)
            self._gz.append(_gz)

            ii = self.partition_index[self.mpi_rank]*(self._mgl[:,level]-bzone)     

            self._global_index.append([ii,ii+self._mgl[:,level]])

            x = gx[ii[0]:ii[0]+self._mgl[0,level]]
            y = gy[ii[1]:ii[1]+self._mgl[1,level]]
            z = gz[ii[2]:ii[2]+self._mgl[2,level]]

            _x, _y, _z = np.meshgrid(x,y,z,indexing='ij')
            self._x.append(_x)
            self._y.append(_y)
            self._z.append(_z)

            dr=self._l/(gN-1)
            
            dx = np.ones_like(_x)*dr[0]
            dy = np.ones_like(_y)*dr[1]
            dz = np.ones_like(_z)*dr[2]

            logging.debug('dx:%f dy:%f dz: %f', dr[0],dr[1],dr[2])

            self._dxyz.append(np.array([dx, dy, dz] ) )

            b_indx, c_indx = self.get_boundary_index(level)
                     
            self._boundary_indx.append(b_indx)
            self._cpu_boundary_indx.append(c_indx)
 
#            logging.debug(z)

            
        logging.debug(self._boundary_indx)
        logging.debug(self._cpu_boundary_indx)
        logging.debug(self._global_index)
        self.number_of_mol_variables = 0


        
    
    
    def boundary_ghost_zones(self,order):
        return  int(order/2)


    def get_partition(self):
        
        P=[ 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67]

        len_P=len(P)-1

        for l in np.arange(start=0,stop=3):
            for m in np.arange(start=0,stop=3):
                for n in np.arange(start=0,stop=3):
                    for i in np.arange(start=0,stop=len_P):
                        for j in np.arange(start=0,stop=len_P):
                            for k in np.arange(start=0,stop=len_P):
                                a=P[i]**n 
                                b=P[j]**(n+m) 
                                c=P[k]**(n+m+l)
                                if self.mpi_size == int(a*b*c):
                                    return np.sort([a,b,c])
        raise Exception('get_partition failed')
        exit(1)

    
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
        

    def get_partition_index(self):
        
        pi = list()
        
        n,m,l = self.cpu_part
        
        for i in np.arange(n):
            for j in np.arange(m):
                for k in np.arange(l):
                    pi.append(np.array([i,j,k]))
                        
        return pi 
        

    def set_grid_partition(self):

        ll = self._l/self.cpu_part
        
        ml = -0.5*self._l+self.partition_index[self.mpi_rank]*ll
        pl = ml+ll
        
        
        return np.array([ml,pl]).T


    def get_boundary_type(self):
        
        bt = -np.ones([3,2])
        
        pi = self.partition_index[self.mpi_rank]
        
        axis_vector = [np.array([1,0,0]),
                       np.array([0,1,0]),
                       np.array([0,0,1])]
        
        for axis in np.arange(3):
            if self.cpu_part[axis] == 1:
                continue
            for cpu in np.arange(self.mpi_size):
                other_pi = self.partition_index[cpu]

                indx_left = other_pi == pi-axis_vector[axis]
                indx_right = other_pi == pi+axis_vector[axis]
                if np.all(indx_left):                    
                    bt[axis,0] = cpu
                elif np.all(indx_right):                    
                    bt[axis,1] = cpu
            
        logging.debug('Boundary with cpu:' )
        logging.debug('x-->[%d, %d]',bt[0,0],bt[0,1])
        logging.debug('y-->[%d, %d]',bt[1,0],bt[1,1])
        logging.debug('z-->[%d, %d]',bt[2,0],bt[2,1])
            
        return bt
        
    def get_boundary_index(self,level):

        b_indx=list()
        c_indx=list()
        pi = self.partition_index[self.mpi_rank]

        if level == 0:
            buffer = self.boundary_gz
        else:
            buffer = 1
        
        for axis in np.arange(3):   
                                
            if pi[axis] == 0:
                bi = [slice(0,buffer,1)]
                ci = list()
            else:
                ci = [slice(0,buffer,1)]
                bi = list()

            if pi[axis] == self.cpu_part[axis]-1:                        
                bi.append( slice(self._mgl[axis,level]-buffer,self._mgl[axis,level],1))
            else:
                ci.append( slice(self._mgl[axis,level]-buffer,self._mgl[axis,level],1))
  
            b_indx.append(np.array(bi))
            c_indx.append(np.array(ci))

        

        return b_indx, c_indx
   
   
    def get_mol_variables(self):
        
        for field in self.hyperbolic_fields:
            self.sync_field(field, 0)

        
        return self._fields[0][:self.shape[0],:,:,:].ravel()

    def get_rhs(self,rhs):

                                                                
        f = np.zeros(self.shape)
        
        for field in  self.hyperbolic_fields:
                                                                
            f[self._field_index[field],:,:,:] = rhs[field]
            self.sync_field(field, 0)

        return f 


    def set_mol_fields(self,y):
        self._fields[0][:self.shape[0],:,:,:] = y.reshape(self.shape)
        for field in self.hyperbolic_fields:
            self.sync_field(field, 0)
   
   
   
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

    def get_field(self,name, level=0):

        if self.has_field(name):        
            if level < self._mg_levels:
                return self._fields[level][self._field_index[name]][:,:,:]
            logging.error('The grid has %d levels, error trying to access level %d', self._mg_levels, level)
                
        logging.error('Field %s does not exists', name)
  
   
    def _add_field(self,field_index, name='' ):

        if name in self._field_index.keys():
            logging.warning('Field %s already exists in amr grid', name)
            return field_index

        self._field_index[name]=field_index
        return field_index + 1
   
    def get_coordinates(self,level=0, cglobal=False):
        if cglobal:
            return self._gx[level], self._gy[level], self._gz[level]
            
        return self._x[level], self._y[level], self._z[level]

    def get_dxyz(self,level=0):
        return self._dxyz[level]


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
        self.sync_field(name, 0)
        
        
    def setitem_2(self,name,l,val):
        self._fields[l][self._field_index[name]][:,:,:] = val
        self.sync_field(name, l)
        
    def setitem_4(self,i,j,k,name,val):
        self.setitem_5(i,j,k,name,0,val)
        self.sync_field(name, 0)

    def setitem_5(self,i,j,k,name,l,val):
        i = self.parse_index(i,l,0)
        j = self.parse_index(j,l,1)
        k = self.parse_index(k,l,2)     
        self._fields[l][self._field_index[name]][i,j,k] = val
        self.sync_field(name, l)


    def gather_field(self,name,l=0):
        
        
        data = self._fields[l][self._field_index[name]]                               
        gather_data = self.mpi_comm.gather(data, root=0)

        gindex=self._global_index[l]
        gather_gindex = self.mpi_comm.gather(gindex, root=0)

                
        if gather_data == None:
            return 
                        
                        
        full_data = np.empty(self._gx[l].shape)


        for cpu in np.arange(self.mpi_size):

            index=gather_gindex[cpu]           
            ii=index[0]
            fi=index[1]
            
            
            full_data[ii[0]:fi[1],ii[1]:fi[1],ii[2]:fi[2]] = gather_data[cpu]
        
        #logging.debug(full_data)
                
        return full_data


    def sync_field(self,name,l=0):


        if l ==0:
            bgz = self.boundary_gz
        else:
            bgz = 1


        for axis in np.arange(3):
            bt=self._boundary_type[axis]

            bslice=self._cpu_boundary_indx[l][axis]
            if len(bslice) > 0 :
                bindex=np.arange(start=bslice[0].start,stop=bslice[0].stop,step=bslice[0].step)
            else:
                bindex=bslice
            
            if(len(bindex) > self.boundary_gz):                    
                lboundary_index=bindex[:bgz]
                rboundary_index=bindex[bgz:]
            else:
                lboundary_index=bindex
                rboundary_index=bindex
                
      
                
            if bt[0] >= 0:
                buf_l = self.get_field_face(name, axis, lboundary_index+bgz, l)
                buf_l_shape=buf_l.shape
                buf_l=buf_l.flatten()
                stag=2*axis
                rtag=2*axis+1
                #logging.debug('send data stag:%d to cpu: %d, rtag: %d',stag,bt[0],rtag)
                #logging.debug(lboundary_index+bgz)
                #logging.debug(lboundary_index)
                self.mpi_comm.Sendrecv_replace(buf_l, dest=bt[0], sendtag=stag, source=bt[0], recvtag=rtag)
                buf_l=buf_l.reshape(buf_l_shape)
                self.set_field_face(buf_l, name, axis, lboundary_index, l)
                    
            if bt[1] >= 0:
                buf_r = self.get_field_face(name, axis, rboundary_index-bgz, l)
                buf_r_shape=buf_r.shape
                buf_r=buf_r.flatten()
                stag=2*axis+1
                rtag=2*axis
                #logging.debug('send data stag:%d to cpu: %d, rtag: %d',stag,bt[1],rtag)
                #logging.debug(rboundary_index-bgz)
                #logging.debug(rboundary_index)

                self.mpi_comm.Sendrecv_replace(buf_r, dest=bt[1], sendtag=stag, source=bt[1], recvtag=rtag)
                buf_r=buf_r.reshape(buf_r_shape)
                self.set_field_face(buf_r,name, axis, rboundary_index, l)

            self.mpi_comm.Barrier()
            


    def get_field_face(self,name,axis,sl,l=0 ):
        if axis==0:
            return self._fields[l][self._field_index[name]][sl,:,:]                               
        elif axis==1:
            return self._fields[l][self._field_index[name]][:,sl,:]                               
        elif axis==2:
            return self._fields[l][self._field_index[name]][:,:,sl]                               
        
    def set_field_face(self,value,name,axis,sl ,l=0):
        if axis==0:
            self._fields[l][self._field_index[name]][sl,:,:] = value                              
        elif axis==1:
            self._fields[l][self._field_index[name]][:,sl,:] = value                              
        elif axis==2:
            self._fields[l][self._field_index[name]][:,:,sl] = value 
        
        
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
 #       self.sync_field(field_2h, level+1)
        
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

#        self.sync_field(field_h, level-1)
 




        