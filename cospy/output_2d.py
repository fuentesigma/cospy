'''
Created on 16 May 2015

@author: Jesus Fuentes and Pablo Galaviz
'''
import logging
import h5py

class Output_2D(object):
    '''
    classdocs
    '''

    def __init__(self, params, project_directory):
        '''
        Constructor
        '''

        self.output_axis = list()

        output_axis = params.get('2D output')
        if output_axis != None:
            output_axis = output_axis.replace(' ','').split(',')
        
            for ax in output_axis:
                if ax == 'xy':
                    self.output_axis.append(ax)
                elif ax == 'yz':
                    self.output_axis.append(ax)
                elif ax == 'xz':
                    self.output_axis.append(ax)
                else:
                    logging.warning('Not valid 2D keyword found in parameter file %s', ax)
     
            
            logging.info("Output one dimensional axis: %s", ''.join([ax+',' for ax in self.output_axis])[:-1])
        
            self.delta_t = abs(params.getfloat('2D delta output',0))

            if self.delta_t > 0:
                logging.info("2D output every  %s", self.delta_t)
            else:
                logging.info("2D output every  time-step")

        self.next_output = 0

        self.xy_origin = params.getfloat('2D xy origin',0)
        self.yz_origin = params.getfloat('2D yz origin',0)
        self.xz_origin = params.getfloat('2D xz origin',0)

        self.verbose = params.getboolean('2D verbose', False)

        self._fields = params.get('2D output _fields').replace(' ','').split(',')

        self.h5file = h5py.File(project_directory+'/cospy_output_2d.h5')

        self.interpolate = params.getboolean('2D interpolate')

        self.iteration = 0



    def init(self, grid):
        
        for fname in self._fields:
            self.h5file.create_group(fname)
            for ax in self.output_axis:
                self.init_axis(grid, fname, ax)
            


    def init_axis(self, grid, fname, ax):
        
        axis={'yz':0, 'xz':1,'xy':2}

        x1, x2 = grid.init_slice_field(fname, axis=axis[ax],origin=self.xy_origin, interpolate = self.interpolate)
        coordinates = self.h5file.create_group(fname+'/coordinates')
    
        coordinates[ax[0]] = x1
        coordinates[ax[1]] = x2
            

            
        
    def update(self,grid,t):
        
        if len(self.output_axis) > 0:
            if t >= self.next_output + self.delta_t or t == 0:
                if self.verbose:
                    logging.info('2D Output at time: %f',t)
                
                for ax in self.output_axis:
                    getattr(self, 'output_axis_'+ax)(grid, t)
                
                self.next_output += self.delta_t

                self.iteration += 1
        
        
    def output_axis(self, grid, t, ax):

        axis={'yz':0, 'xz':1,'xy':2}
        
        for fname in self._fields:
            u = grid.get_slice_field(fname, axis=axis[ax], interpolate = self.interpolate)
            
            u_name = fname+'/'+str(self.iteration)
                
            self.h5file[u_name] = u
            self.h5file[u_name].attrs['time'] = t
                                        
        

        
