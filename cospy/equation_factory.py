'''
Created on 18 May 2015

@author: Jesus Fuentes and Pablo Galaviz
'''


from sympy import *
import numpy as np 
from IPython.display import display

from sympy.interactive import printing
from asyncio.locks import Condition
printing.init_printing(use_latex=True)
import shutil
import os 
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader
import cospy.utils as cp_utils

    
    
def parse_expr(expr):
    
    arg_list=list()
    for arg in expr.args:
        arg_list.append(parse_expr(arg))
        
    if expr.func.is_Add:       
        return '+'.join(arg_list)
    if expr.func.is_Mul:       
        return '*'.join(arg_list)
    if expr.func.is_Pow:       
        return '**'.join(arg_list)
    if expr.func.is_Derivative:   

        var = arg_list[0]
        x_der=0
        y_der=0
        z_der=0 
        for c in arg_list[1:]:
            if 'x' in c:
                x_der+=1
            elif 'y' in c:
                y_der+=1
            elif 'z' in c:
                z_der+=1 
        if x_der !=0:    
            return 'cp_diff.der("'+var+'",axis=0, order='+str(x_der)+')'
        elif y_der !=0:    
            return 'cp_diff.der("'+var+'",axis=1, order='+str(y_der)+')'
        elif z_der !=0:    
            return 'cp_diff.der("'+var+'",axis=2, order='+str(z_der)+')'


    if expr.func.is_Function:
        if len(expr.args) == 1:
            return sstr(expr.func)+'('+','.join(arg_list)+')'
        else:
            return sstr(expr.func)
    
    return sstr(expr)
    



class Cospy_Fields(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.amr_variables=dict()
        self.multigrid_variables=dict()


    def add_amr_variable(self,name):
        
        self.amr_variables[name]=Function(name)

        return self.amr_variables[name]

    def add_multigrid_variable(self,name):
        print('Not implemented')
        
        return None

    def add_auxiliary_variable(self,name):
        print('Not implemented')
        
        return None



class Cospy_Equations(object):
    '''
    classdocs
    '''


    def __init__(self, variables):
        '''
        Constructor
        '''
        self.variables = variables
        self.amr_equations = list()        
        self.amr_boundary = list()
        self.amr_initial_data = list()

    def add_amr_rhs(self, variable_name, rhs):

        if not variable_name in self.variables.amr_variables.keys():
            raise Exception('Variable %s is not defined in amr variables', variable_name)
            return

        new_equation = dict()
       
        new_equation['lhs']=self.variables.amr_variables[variable_name]
        new_equation['rhs']=S(rhs)
        
        self.amr_equations.append(new_equation)

    def display(self):
        

        for e in self.amr_equations:
            display(Eq(e['lhs'](t).diff(t),e['rhs']))
        
        
    def add_amr_boundary(self, variable_name, condition,value = 0):

        if not variable_name in self.variables.amr_variables.keys():
            raise Exception('Variable %s is not defined in amr variables', variable_name)
            return

        if not condition in ['Dirichlet', 'Periodic']:
            raise Exception('Condition %s is not defined in amr boundary', condition)
            return

        new_boundary = dict()
    
        new_boundary['variable']=variable_name
        new_boundary['condition'] = condition
        if condition == 'Dirichlet':
            new_boundary['value'] = value

                
        self.amr_boundary.append(new_boundary)
        
    def boundary(self):
        
        for b in self.amr_boundary:
            print('boundary condition for variable '+b['variable']+' is '+b['condition'])

        
    def add_amr_initial_data(self, variable_name, initial_data, args):

        if not variable_name in self.variables.amr_variables.keys():
            raise Exception('Variable %s is not defined in amr variables', variable_name)
            return

        new_initial_data = dict()
    
        new_initial_data['variable']=variable_name
        new_initial_data['function'] = S(initial_data)
        new_initial_data['arguments'] = args
                
        self.amr_initial_data.append(new_initial_data)
        
    def initial_data(self):
        
        for i_d in self.amr_initial_data:
            print('initial data for variable '+i_d['variable']+' is:')
            display( i_d['function'])
    
    
def make_problem(problem_name, equations):

    dir_name = os.path.abspath(os.path.curdir+'/../')
    template_file_name = 'equation_template.py' 
    output_file = dir_name+'/'+problem_name.replace(' ','_')+'.py'    
  
    
    template_enviroment = Environment(
        autoescape=False,
        loader=FileSystemLoader(os.path.join(dir_name, 'templates')),
        trim_blocks=False)
    
    
    template = template_enviroment.get_template(template_file_name)
    
    templete_fields = dict()

    field_map = list()
    field_dict=dict()
    for k in equations.variables.amr_variables.keys():
        field_dict[k] = 'amr'
        field_map.append(sstr(k))

    templete_fields['field_map'] = field_map
        
    templete_fields['field_dict'] = field_dict

    templete_fields['field_list'] = list(equations.variables.amr_variables.keys())
    
    
    id_dict = dict()
    
    for id_f in equations.amr_initial_data:
        id_dict[id_f['variable']] = parse_expr(id_f['function'])

    
    templete_fields['id_dict'] = id_dict

    rhs_dict = dict()
    
    for eq in equations.amr_equations:
        rhs_dict[sstr(eq['lhs'])] = parse_expr(eq['rhs'])
    
    templete_fields['rhs_dict']=rhs_dict
    
    boundary_dict = dict()
    for boundary in equations.amr_boundary:
        if boundary['condition'] == 'Dirichlet':
            boundary_dict[boundary['variable']] = boundary['condition']+':'+str(boundary['value'])
        elif boundary['condition'] == 'Periodic':
            boundary_dict[boundary['variable']] = boundary['condition']
 
           
            
    templete_fields['boundary_dict']=boundary_dict
    
    with open(output_file, "w") as pfile:
        pfile.write(template.render(templete_fields))
    #print(template.render(templete_fields))
        
    file_import=dir_name+'/equations.py'
    import_exist=False
    
    module_import = 'import cospy.'+problem_name.replace(' ','_')
    with open(file_import, "r") as ifile:
        for line in ifile:
            if 'import' in line:
                if module_import in line:
                    import_exist = True
                    break
    if not import_exist:
        with open(file_import, "a") as ifile:
            ifile.write(module_import+'\n')    
    
        
        

def review_make_equation(problem_name, equations ,make_equation = False):
    
    print('SUMARY:')
    print('Problem name: '+problem_name)

    dir_name = os.path.abspath(os.path.curdir+'/../')
    output_file = dir_name+'/'+problem_name.replace(' ','_')+'.py'    

    if cp_utils.test_file_exist(output_file):
        print('Warining: Problem exists, OVERWRITE?')

    
    if make_equation:
        make_problem(problem_name, equations)
        print('Source file created: ', output_file)

    



    
    
    
x, y, z, t = symbols('x y z t')
x0, y0, z0, t0 = symbols('x0 y0 z0 t0')
xN, yN, zN = symbols('xN yN zN ')
k, m, n = symbols('k m n', integer=True)

