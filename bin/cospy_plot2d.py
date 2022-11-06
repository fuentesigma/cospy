#!/usr/bin/env python2.7
'''
Created on 23 May 2015

@author: Jesus Fuentes and Pablo Galaviz
'''

import wx
import visvis as vv
import argparse
import h5py
import numpy as np 


app = vv.use('wx')


class MainWindow(wx.Frame):
    def __init__(self, size):
        wx.Frame.__init__(self, None, -1, "Cospy Plot 2D", size=size)
        
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
        
        self.plane_xy = wx.RadioButton(panel, label='XY', pos=(10, 10), style=wx.RB_GROUP)
        self.plane_yz = wx.RadioButton(panel, label='YZ', pos=(30, 10))
        self.plane_xz = wx.RadioButton(panel, label='XZ', pos=(50, 10))

        self.plane_xy.Bind(wx.EVT_RADIOBUTTON, self.SetAxis)
        self.plane_yz.Bind(wx.EVT_RADIOBUTTON, self.SetAxis)
        self.plane_xz.Bind(wx.EVT_RADIOBUTTON, self.SetAxis)

        
        self.label_c = wx.StaticText(panel, label='Z')
        self.index = wx.SpinCtrl(panel, value='0',  size=(60, -1))
        self.index.SetRange(0, 0)       
        

        self.label_iter = wx.StaticText(panel, label='time index')
        self.index_time = wx.Slider(panel, value=0, minValue=0, maxValue=1, pos=(20, 20), size=(250, -1), style=wx.SL_HORIZONTAL)
        self.index_time.SetRange(0, 0)       
        self.index_time.Bind(wx.EVT_SCROLL, self.Plot)

        button = wx.Button(panel, -1, 'Plot')
        button.Bind(wx.EVT_BUTTON, self.Plot)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)        
        sizer.Add(label, proportion=0, flag=wx.ALL, border=5)
        sizer.Add(self.fields_combo_box, proportion=0, flag=wx.ALL, border=5)
        
        sizer.Add(self.plane_xy, proportion=0, flag=wx.ALL, border=5)
        sizer.Add(self.plane_yz, proportion=0, flag=wx.ALL, border=5)
        sizer.Add(self.plane_xz, proportion=0, flag=wx.ALL, border=5)

        sizer.Add(self.label_c, proportion=0, flag=wx.ALL|wx.ALIGN_CENTRE, border=5)
        sizer.Add(self.index, proportion=0, flag=wx.ALL, border=5)
        

        sizer.Add(self.label_iter, proportion=0, flag=wx.ALL|wx.ALIGN_CENTRE, border=5)
        sizer.Add(self.index_time, proportion=0, flag=wx.ALL, border=5)

        sizer.Add(button, proportion=0, flag=wx.ALL, border=5)
        
        panel.SetSizer(sizer)

        return panel 



        
    def OnQuit(self, e):
        self.Close()


    def SetAxis(self, e):
        
        if self.plane_xy.GetValue():
            self.index.SetRange(0, self.nz) 

            self.index.SetValue(self.nz/2) 
                
            self.label_c.SetLabel('Z')
            
        elif self.plane_yz.GetValue():
            self.index.SetRange(0, self.nx) 

            self.index.SetValue(self.nx/2) 

            self.label_c.SetLabel('X')
            
        elif self.plane_xz.GetValue():
            self.index.SetRange(0, self.ny) 

            self.index.SetValue(self.ny/2) 

            self.label_c.SetLabel('Y')
        

  
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
        
        
        self.index.SetRange(0, self.nz) 

        self.index.SetValue(self.nz/2) 

        self.frames = len(self.h5file[self.fields[0]].keys())-1
        self.index_time.SetRange(0, self.frames) 


    def get_data_at_iteration(self,iter_time):
        
        v=self.index.GetValue()

        if self.plane_xy.GetValue():

            x=self.h5file['coordinates/x'][:,:,v]
            y=self.h5file['coordinates/y'][:,:,v]
            z=self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time][:,:,v]

            c=self.h5file['coordinates/z'][0,0,v]

            axis_name='z'
            c0_name='x'
            c1_name='y'

            time = self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time].attrs['time']

        elif self.plane_yz.GetValue():
            
            x=self.h5file['coordinates/y'][v,:,:]
            y=self.h5file['coordinates/z'][v,:,:]
            z=self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time][v,:,:]

            c=self.h5file['coordinates/x'][v,0,0]

            axis_name='x'
            c0_name='y'
            c1_name='z'

            time = self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time].attrs['time']

        elif self.plane_xz.GetValue():
            
            x=self.h5file['coordinates/x'][:,v,:]
            y=self.h5file['coordinates/z'][:,v,:]
            z=self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time][:,v,:]

            c=self.h5file['coordinates/y'][0,v,0]

            axis_name='y'
            c0_name='x'
            c1_name='z'

            time = self.h5file[self.fields_combo_box.GetValue()+'/'+iter_time].attrs['time']

        return x, y, z, c, axis_name, c0_name, c1_name, time
    
    
    def make_plot(self,x, y, z,time, c0_name, c1_name, axis_name, c ):
        
        self.max_axis=max(self.max_axis, np.max(z))
        self.min_axis=min(self.min_axis, np.min(z))

        
        p=vv.plot(x,y,z, lw=2)
        vv.legend([axis_name+'='+str(c)])        

        vv.title('\b{Time = }'+str(time))

        ax = vv.gca()                
        ax.axis.xlabel = c0_name
        ax.axis.ylabel = c1_name
        
        ax.SetLimits(rangeX=(np.min(x),np.max(x)),rangeY=(np.min(y),np.max(y)),rangeZ=(self.min_axis,self.max_axis))

        
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
        
        x, y, z, c, axis_name, c0_name, c1_name, time = self.get_data_at_iteration(str(iter_time))
        
        self.make_plot(x, y, z, time, c0_name, c1_name, axis_name, c)
        
        
        #self.fig.DrawNow()
    


if __name__ == '__main__':
    
    epilog_text=" =============================\n     Author: Pablo Galaviz    \n     pablo.galaviz@me.com  \n =============================\n   License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html> \n   This is free software: you are free to change and redistribute it.\n   There is NO WARRANTY, to the extent permitted by law."

    parser = argparse.ArgumentParser(description='Cospy, cosmology amr solver - Plot 1d', epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('-s','--size', nargs=2, type=int, default=(1024, 720) ,help='Set the default windows size')

    args = parser.parse_args()
    
    
    app.Create()
    m = MainWindow(args.size)
    app.Run()
