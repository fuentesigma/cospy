'''
Created on 16 May 2015

@author: Pablo Galaviz
'''
import logging
import h5py


class Output_1D(object):
    '''
    classdocs
    '''

    def __init__(self, params, project_directory):
        '''
        Constructor
        '''

        self.output_axis = list()

        output_axis = params.get('1D output')
        if output_axis != None:
            output_axis = output_axis.replace(' ','').split(',')
        
            for ax in output_axis:
                if ax == 'x':
                    self.output_axis.append(ax)
                elif ax == 'y':
                    self.output_axis.append(ax)
                elif ax == 'z':
                    self.output_axis.append(ax)
                else:
                    logging.warning('Not valid 1D keyword found in parameter file %s', ax)
     
            
            logging.info("Output one dimensional axis: %s", ''.join([ax+',' for ax in self.output_axis])[:-1])
        
            self.delta_t = abs(params.getfloat('1D delta output',0))

            if self.delta_t > 0:
                logging.info("1D output every  %s", self.delta_t)
            else:
                logging.info("1D output every  time-step")

        self.next_output = 0

        self.x_origin = self.parse_origin(params.get('1D x origin',None))
        self.y_origin = self.parse_origin(params.get('1D y origin',None))
        self.z_origin = self.parse_origin(params.get('1D z origin',None))

        self.verbose = params.getboolean('1D verbose', False)

        self._fields = params.get('1D output _fields').replace(' ','').split(',')

        self.h5file = h5py.File(project_directory+'/cospy_output_1d.h5')

        self.interpolate = params.getboolean('1D interpolate')

        self.iteration = 0


        
    def parse_origin(self,vals):
        if vals == None:
            return 0,0
        
        try:
            return [float(x) for x in vals.replace(' ','').split(',')[:2]]
        except:
            logging.error("Not valid origin in par file %s", vals)
        
        
    def init(self, grid):
        
        for fname in self._fields:
            self.h5file.create_group(fname)
            for ax in self.output_axis:
                self.init_axis(grid, fname, ax)
            
    def init_axis(self, grid, fname, ax):
        
        axis = {'x':0, 'y':1, 'z':2}    
        xi = grid.init_ray_field(fname, axis=axis[ax],origin=self.x_origin, interpolate = self.interpolate)
        coordinates = self.h5file.create_group(fname+'/coordinates')
        coordinates[ax] = xi
            


    def update(self,grid,t):
        
        if len(self.output_axis) > 0:
            if t >= self.next_output + self.delta_t or t == 0:
                if self.verbose:
                    logging.info('1D Output at time: %f',t)
                
                for ax in self.output_axis:
                    self.output_axis(grid, t, ax)
                
                self.next_output += self.delta_t

                self.iteration += 1
        
        
    def output_axis(self, grid, t, ax):

        axis={'x':0, 'y':1,'z':2}
        
        for fname in self._fields:
            u = grid.get_ray_field(fname, axis=axis[ax], interpolate = self.interpolate)
            
            u_name = fname+'/'+str(self.iteration)
                
            self.h5file[u_name] = u
            self.h5file[u_name].attrs['time'] = t
                                        
        


        
        

        