#!/usr/bin/env python2.7
'''
Created on 23 May 2015

@author: pablo
'''

import wx
import visvis as vv
import argparse
import h5py
import numpy as np 


app = vv.use('wx')


class MainWindow(wx.Frame):
    def __init__(self, size):
        wx.Frame.__init__(self, None, -1, "Cospy Plot 1D", size=size)
        
        self.h5file = None
        self.fields = None

        self.nx=0
        self.ny=0
        self.nz=0

        self.max_axis=0
        self.min_axis=0
        
        self.SetMinSize([560, 420])
        
        self.MakeMenu()
        
        # Make a panel with a button
        panel = self.MakeTools()         
        # Make figure using "self" as a parent
        Figure = app.GetFigureClass()
        self.fig = Figure(self)
        
        # Make sizer and embed stuff
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(panel, 0, wx.EXPAND)
        self.sizer.Add(self.fig._widget, 2, wx.EXPAND)
        
        
        # Apply sizers        
        self.SetSizer(self.sizer)
        self.SetAutoLayout(True)
        self.Layout()   
        
        # Finish
        self.Show()


    def MakeMenu(self):

        menubar = wx.MenuBar()

        fileMenu = wx.Menu()
        
        open_menu_item = wx.MenuItem(fileMenu, wx.ID_OPEN, '&Open\tCtrl+O')
        fileMenu.AppendItem(open_menu_item)
        self.Bind(wx.EVT_MENU, self.OnOpen, open_menu_item)

        fileMenu.AppendSeparator()

        quit_menu_item = wx.MenuItem(fileMenu, wx.ID_EXIT, '&Quit\tCtrl+W')
        fileMenu.AppendItem(quit_menu_item)

        self.Bind(wx.EVT_MENU, self.OnQuit, quit_menu_item)

        menubar.Append(fileMenu, '&File')
        self.SetMenuBar(menubar)
        
        
    def MakeTools(self):

        panel = wx.Panel(self)

        label = wx.StaticText(panel, label='Field', style=wx.ALIGN_CENTRE)
        self.fields_combo_box = wx.ComboBox(panel, pos=(50, 30), choices=[''], style=wx.CB_READONLY)
        
        self.axis_x = wx.RadioButton(panel, label='X', pos=(10, 10), style=wx.RB_GROUP)
        self.axis_y = wx.RadioButton(panel, label='Y', pos=(30, 10))
        self.axis_z = wx.RadioButton(panel, label='Z', pos=(50, 10))

        self.axis_x.Bind(wx.EVT_RADIOBUTTON, self.SetAxis)
        self.axis_y.Bind(wx.EVT_RADIOBUTTON, self.SetAxis)
        self.axis_z.Bind(wx.EVT_RADIOBUTTON, self.SetAxis)

        
        self.label_c1 = wx.StaticText(panel, label='Y')
        self.index1 = wx.SpinCtrl(panel, value='0',  size=(60, -1))
        self.index1.SetRange(0, 0)       
        
        self.label_c2 = wx.StaticText(panel, label='Z')
        self.index2 = wx.SpinCtrl(panel, value='0', size=(60, -1))
        self.index2.SetRange(0, 0)       


        self.label_iter = wx.StaticText(panel, label='time index')
        self.index_time = wx.Slider(panel, value=0, minValue=0, maxValue=1, pos=(20, 20), size=(250, -1), style=wx.SL_HORIZONTAL)
        self.index_time.SetRange(0, 0)       
        self.index_time.Bind(wx.EVT_SCROLL, self.Plot)

        button = wx.Button(panel, -1, 'Plot')
        button.Bind(wx.EVT_BUTTON, self.Plot)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)        
        sizer.Add(label, proportion=0, flag=wx.ALL, border=5)
        sizer.Add(self.fields_combo_box, proportion=0, flag=wx.ALL, border=5)
        
        sizer.Add(self.axis_x, proportion=0, flag=wx.ALL, border=5)
        sizer.Add(self.axis_y, proportion=0, flag=wx.ALL, border=5)
        sizer.Add(self.axis_z, proportion=0, flag=wx.ALL, border=5)

        sizer.Add(self.label_c1, proportion=0, flag=wx.ALL|wx.ALIGN_CENTRE, border=5)
        sizer.Add(self.index1, proportion=0, flag=wx.ALL, border=5)
        
        sizer.Add(self.label_c2, proportion=0, flag=wx.ALL|wx.ALIGN_CENTRE, border=5)
        sizer.Add(self.index2, proportion=0, flag=wx.ALL, border=5)

        sizer.Add(self.label_iter, proportion=0, flag=wx.ALL|wx.ALIGN_CENTRE, border=5)
        sizer.Add(self.index_time, proportion=0, flag=wx.ALL, border=5)

        sizer.Add(button, proportion=0, flag=wx.ALL, border=5)
        
        panel.SetSizer(sizer)

        return panel 



        
    def OnQuit(self, e):
        self.Close()


    def SetAxis(self, e):
        
        if self.axis_x.GetValue():
            self.index1.SetRange(0, self.ny) 
            self.index2.SetRange(0, self.nz) 

            self.index1.SetValue(self.ny/2) 
            self.index2.SetValue(self.nz/2) 
                
            self.label_c1.SetLabel('Y')
            self.label_c2.SetLabel('Z')
            
        elif self.axis_y.GetValue():
            self.index1.SetRange(0, self.nx) 
            self.index2.SetRange(0, self.nz) 

            self.index1.SetValue(self.nx/2) 
            self.index2.SetValue(self.nz/2) 

            self.label_c1.SetLabel('X')
            self.label_c2.SetLabel('Z')
            
        elif self.axis_z.GetValue():
            self.index1.SetRange(0, self.nx) 
            self.index2.SetRange(0, self.ny) 

            self.index1.SetValue(self.nx/2) 
            self.index2.SetValue(self.ny/2) 

            self.label_c1.SetLabel('X')
            self.label_c2.SetLabel('Y')
        

  
    def OnOpen(self, event):
    
        openFileDialog = wx.FileDialog(self, "Open HDF5 file", "", "",
                                       "HDF5 files (*.h5)|*.h5", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return     # the user changed idea...

        # proceed loading the file chosen by the user
        # this can be done with e.g. wxPython input streams:
        file_path = openFileDialog.GetPath()

        self.h5file = h5py.File(file_path)

        self.fields_combo_box.Clear()
        
        self.fields = list()
        for k in self.h5file.keys():
            if k != 'coordinates':
                self.fields_combo_box.Append(k)
                self.fields.append(k)
        
        self.fields_combo_box.SetValue(self.fields[0])
        
        self.x=self.h5file['coordinates/x'][:,0,0]
        self.y=self.h5file['coordinates/y'][0,:,0]
        self.z=self.h5file['coordinates/z'][0,0,:]

        self.nx=self.x.shape[0]
        self.ny=self.y.shape[0]
        self.nz=self.z.shape[0]
        
        
        self.index1.SetRange(0, self.ny) 
        self.index2.SetRange(0, self.nz) 

        self.index1.SetValue(self.ny/2) 
        self.index2.SetValue(self.nz/2) 

        self.frames = len(self.h5file[self.fields[0]].keys())-1
        self.index_time.SetRange(0, self.frames) 


    def get_data_at_iteration(self,iter_time):
        v1=self.index1.GetValue()
        v2=self.index2.GetValue()

        if self.axis_x.GetValue():

            x=self.h5file['coordinates/x'][:,v1,v2]
            y=self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time][:,v1,v2]

            c0=self.h5file['coordinates/y'][0,v1,v2]
            c1=self.h5file['coordinates/z'][0,v1,v2]

            axis_name='x'
            c0_name='y'
            c1_name='z'

            time = self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time].attrs['time']

        elif self.axis_y.GetValue():
            x=self.h5file['coordinates/y'][v1,:,v2]
            y=self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time][v1,:,v2]

            c0=self.h5file['coordinates/x'][v1,0,v2]
            c1=self.h5file['coordinates/z'][v1,0,v2]
            
            axis_name='y'
            c0_name='x'
            c1_name='z'

            time = self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time].attrs['time']

        elif self.axis_z.GetValue():
            x=self.h5file['coordinates/z'][v1,v2,:]
            y=self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time][v1,v2,:]
                
            c0=self.h5file['coordinates/x'][v1,v2,0]
            c1=self.h5file['coordinates/y'][v1,v2,0]

            axis_name='z'
            c0_name='x'
            c1_name='y'

            time = self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time].attrs['time']

        return x, y, c0, c1, axis_name, c0_name, c1_name, time
    
    
    def make_plot(self,x, y, time, c0_name, c1_name, axis_name, c0, c1 ):
        
        self.max_axis=max(self.max_axis, np.max(y))
        self.min_axis=min(self.min_axis, np.min(y))
        
        p=vv.plot(x,y, lw=2)
        vv.legend([c0_name+'='+str(c0)+' '+c1_name+'='+str(c1)])        

        vv.title('\b{Time = }'+str(time))

        ax = vv.gca()                
        ax.axis.xlabel = axis_name
        ax.axis.ylabel = self.fields_combo_box.GetValue()        
        ax.SetLimits(rangeX=(min(x),max(x)),rangeY=(self.min_axis,self.max_axis))
        
        return p
    
    def Plot(self, event):
        
        # Make sure our figure is the active one
        # If only one figure, this is not necessary.
        #vv.figure(self.fig.nr)
        
        if self.h5file == None:
            return
        
        # Clear it
        vv.clf()
        
        # Plot
        
        iter_time=str(self.index_time.GetValue())
        
        x, y, c0, c1, axis_name, c0_name, c1_name, time = self.get_data_at_iteration(str(iter_time))
        
        self.make_plot(x, y, time, c0_name, c1_name, axis_name, c0, c1)
        
        
        #self.fig.DrawNow()
    


if __name__ == '__main__':
    
    epilog_text=" =============================\n     Author: Pablo Galaviz    \n     pablo.galaviz@me.com  \n =============================\n   License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html> \n   This is free software: you are free to change and redistribute it.\n   There is NO WARRANTY, to the extent permitted by law."

    parser = argparse.ArgumentParser(description='Cospy, cosmology amr solver - Plot 1d', epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('-s','--size', nargs=2, type=int, default=(1024, 720) ,help='Set the default windows size')

    args = parser.parse_args()
    
    
    app.Create()
    m = MainWindow(args.size)
    app.Run()
