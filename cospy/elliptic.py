'''
Created on 20 Jun 2015

@author: pablo
'''

import numpy as np 
import cospy.utils as cp_utils
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib.backends.backend_qt5 import SHIFT
from mpi4py import MPI


class Elliptic(object):
    '''
    General multigrid elliptic equation solver
    '''


    def __init__(self, grid, elliptic_system, params):
        '''
        Constructor
        '''

        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_size = self.mpi_comm.Get_size() # total number of nodes
        self.mpi_rank = self.mpi_comm.Get_rank() # current node 


        
        self.elliptic_fields = grid.elliptic_fields
        
        self.grid = grid
             
        self.elliptic_system = elliptic_system()   

        self.verbose = params.getboolean('verbose multigrid', False)

        self.pre_cycles = params.getint('pre cycles',10)
        self.post_cycles = params.getint('post cycles',10)

        self.fine_tolerance = params.getfloat('fine grid tolerance',1e-6)
        self.coarse_tolerance = params.getfloat('coarse grid tolerance',1e-8)



    def solve(self):
        
        if len(self.elliptic_fields) == 0:
            return 

        for field, data in self.elliptic_system.items():

            x, y , z = self.grid.get_coordinates(0)

            self.grid[field+'_source',0] = data['source'](self.grid,0)
#            self.grid[field,0] = 1+0.01*np.exp(-20*(x**2+y**2+z**2))


 
 #==============================================================================
 #           self.grid[field] = np.sin(x+y+z)
 #           self.grid[field+'_source'] = -3*np.sin(x+y+z)
 #           self.grid[field+'_omega'] = self.grid.lap(field)
 # 
 #==============================================================================
 #==============================================================================
 #            level = 0
 #            bx = self.grid.parse_index('b',level,0)
 #            by = self.grid.parse_index('b',level,1)
 #            bz = self.grid.parse_index('b',level,2)
 #             
 #            self.grid[bx,:,:,field+'_omega'] = self.grid[bx,:,:,field+'_source'] 
 #            self.grid[:,by,:,field+'_omega'] = self.grid[:,by,:,field+'_source'] 
 #            self.grid[:,:,bz,field+'_omega'] = self.grid[:,:,bz,field+'_source'] 
 #     
 #            self.grid[bx,by,:,field+'_omega'] = self.grid[bx,by,:,field+'_source'] 
 #            self.grid[:,by,bz,field+'_omega'] = self.grid[:,by,bz,field+'_source'] 
 #            self.grid[bx,:,bz,field+'_omega'] = self.grid[bx,:,bz,field+'_source'] 
 #     
 #            self.grid[bx,by,bz,field+'_omega'] = self.grid[bx,by,bz,field+'_source'] 
 # 
 #            print(np.max(self.grid[field+'_omega']-self.grid[field+'_source']))
 #==============================================================================
 
 #           self.plotC(field+'_omega',0,field+'_source',0 ,axis=2)
 #           self.mpi_comm.barrier()
 #           exit()

            if self.verbose:       
                logging.info('Solving elliptic equation for field: %s',field)


            Loo=1
            i=0
            while Loo > self.fine_tolerance:        
            #    Loo, it = self.smooth(field, data, 0,1)

            #    if np.mod(i,1000) == 0:
            #        self.plot(field, 0, 2)
                Loo, it = self.VCycle(field,data,0,i)
                i+=1
#                print(Loo)
                #break

            #self.plot2D(field, 0, 1)


            if self.verbose:               
                logging.info('Elliptic solver L_inf error = %e, in iteration %d',Loo,i)
            
        
    def plot(self,field,level, axis=0, shift_1=0,shift_2=0, block=True):
        
        x, y , z = self.grid.get_coordinates(level)
            
        U = self.grid[field,level]  
        
        nx=U.shape[0]/2
        ny=U.shape[1]/2
        nz=U.shape[2]/2
        
        if axis == 2:
            plt.plot(z[nx+shift_1,ny+shift_2,:], U[nx+shift_1,ny+shift_2,:],'b-o')
        elif axis == 1:
            plt.plot(y[nx+shift_1,:,nz+shift_2], U[nx+shift_1,:,nz+shift_2],'g-o')
        elif axis == 0:
            plt.plot(x[:,ny+shift_1,nz+shift_2], U[:,ny+shift_1,nz+shift_2],'r-o')
        
        plt.show(block=block)
        time.sleep(0.1)
        plt.close()

    def plotC(self,field1,level1,field2,level2, axis=0, shift_1=0,shift_2=0,block=True):
        
        x1, y1 , z1 = self.grid.get_coordinates(level1)
            
        U1 = self.grid[field1,level1]  
        
        nx1=U1.shape[0]/2
        ny1=U1.shape[1]/2
        nz1=U1.shape[2]/2

        x2, y2 , z2 = self.grid.get_coordinates(level2)
            
        U2 = self.grid[field2,level2]  
        
        nx2=U2.shape[0]/2
        ny2=U2.shape[1]/2
        nz2=U2.shape[2]/2
        
        if axis == 2:
            plt.plot(z1[nx1+shift_1,ny1+shift_2,:], U1[nx1+shift_1,ny1+shift_2,:],'r-^',z2[nx2+shift_1,ny2+shift_2,:], U2[nx2+shift_1,ny2+shift_2,:],'b-o')
        elif axis == 1:
            plt.plot(y1[nx1+shift_1,:,nz1+shift_2], U1[nx1+shift_1,:,nz1+shift_2],'r-^',y2[nx2+shift_1,:,nz2+shift_2], U2[nx2+shift_1,:,nz2+shift_2],'b-o')
        elif axis == 0:
            plt.plot(x1[:,ny1+shift_1,nz1+shift_2], U1[:,ny1+shift_1,nz1+shift_2],'r-^',x2[:,ny2+shift_1,nz2+shift_2], U2[:,ny2+shift_1,nz2+shift_2],'b-o')
        
        plt.show(block=block)
        time.sleep(0.1)
        plt.close()
        


    def plot2D(self,field,level,axis=0, shift=0,block=True):
        
        fig = plt.figure()
        fig.clf()
        ax = Axes3D(fig)

        x, y , z = self.grid.get_coordinates(level)
            
        U = self.grid[field,level]  
        
        nx=U.shape[0]/2+shift
        ny=U.shape[1]/2+shift
        nz=U.shape[2]/2+shift
        
        if axis == 2:
            ax.plot_surface(x[:,:,nz], y[:,:,nz], U[:,:,nz])
        elif axis == 1:
            ax.plot_surface(x[:,ny,:], z[:,ny,:], U[:,ny,:])
        elif axis == 0:
            ax.plot_surface(y[nx,:,:], z[nx,:,:], U[nx,:,:])

        plt.show(block=block)
        time.sleep(0.1)
        plt.close()
        

    def FAS(self,field, data, level, cycle):


   #     data['boundary'](self.grid,level) 
        Loo = self.smooth(field, data, level,200)

            
        self.grid.injection(field,level,field)

        self.plot( self.grid[field,level+1])

        #I^0_1 u^1


        Lu = data['operator'](self.grid,level) 
        Lup1 = data['operator'](self.grid,level+1) 
        
        
        self.grid[field+'_omega',level] = self.grid[field+'_source',level] - Lu  

        self.grid.injection(field+'_omega',level,field+'_omega')

        self.grid[field+'_source',level+1] = self.grid[field+'_omega',level+1] + Lup1


            
        Up1 = self.grid[field,level+1]

        if level+2 == self.grid._mg_levels:
            i=0
            while Loo > 0.001:
                Loo = self.smooth(field, data, level+1, 10)
                if i > 1000:
                    print(Loo)
                    exit()
                i+=1
#            self.plot( self.grid[field,level+1])
#            exit()
        else:
            Loo = self.FAS(field,data,level+1, cycle)

#        self.plot( self.grid[field,level+1])

        self.grid[field+'_V',level+1] = self.grid[field,level+1] - Up1
        
        self.grid.prolong(field+'_V',level+1,field+'_V')

 
        self.grid[field,level] += self.grid[field+'_V',level]


        Loo = self.smooth(field, data, level,20)

        return Loo



    def VCycle(self,field, data, level, cycle):


     #   self.plot(field+'_source', level)
     #   exit()

        Loo, i = self.smooth(field, data, level,self.pre_cycles)
        if self.verbose:
            logging.info('%s Level %d Loo = %e it=%d','.'.join(np.zeros(level+1,dtype=str)) ,level,Loo,i)


        
        self.grid[field+'_omega',level] = self.grid[field+'_source',level] - data['operator'](self.grid,level) 


        self.grid.injection(field+'_omega',level,field+'_source')



  
        self.grid[field,level+1] = 0
            

        if level+2 == self.grid._mg_levels:
            i=0
            Loo=1
            while Loo > self.coarse_tolerance:
                Loo, j = self.smooth(field, data, level+1, 10)
                if i > 10000:
                    print(Loo)
                    exit()
                i+=1
            if self.verbose:
                logging.info('%s Level %d Loo = %e it=%d','.'.join(np.zeros(level+2,dtype=str)) ,level+1,Loo,i*j)
        else:
            Loo = self.VCycle(field,data,level+1, cycle)


#        self.plot(field, level+1)


        
        self.grid.prolong(field,level+1,field+'_V')

        U=self.grid[field,level] 

        self.grid[field,level] = U+ self.grid[field+'_V',level]


        Loo, i = self.smooth(field, data, level,self.post_cycles)

        if self.verbose:       
            logging.info('%s Level %d Loo = %e it=%d','.'.join(np.zeros(level+1,dtype=str)) ,level,Loo,i)

        return Loo, i

    def smooth(self,field, data, level, max_it = 1):

        i=0                        

        Loo_prev = 1e6

        while True:

    
            r = data['operator'](self.grid,level) - self.grid[field+'_source',level]

            self.grid[field+'_omega',level] = r

#            self.plot(field+'_omega', level, axis=2)

            dr = data['diff_operator'](self.grid,level)
               

            #self.grid[field+'_omega',level] =data['operator'](self.grid,level) 

            
        
            self.grid[field,level] = self.grid[field,level]  - r/dr

            Loo = np.zeros(self.mpi_size, dtype=np.float64)

            ix = self.grid.parse_index('i',level,0)
            iy = self.grid.parse_index('i',level,1)
            iz = self.grid.parse_index('i',level,2)


            Loo_max= np.ones(self.mpi_size)*np.max(np.abs(r[ix,iy,iz]))
            
            self.mpi_comm.Allreduce(Loo_max,Loo,op=MPI.MAX)


            Loo = np.max(Loo)


  #          print(np.abs(self.grid[field+'_omega',level]).max())
#            logging.debug("Send data "+str(Loo_max))     
            logging.debug("Recv data "+str(Loo))     
        
        
           # self.plot(field+'_omega', level)
          #      exit()
           # self.plot(field+'_omega', level)
            
                
            Loo_prev = Loo
        #

            if  i > max_it:
                break
            i+=1

        return Loo, i 
        
        