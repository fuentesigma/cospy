#!/usr/bin/env python2.7
'''
Created on 23 May 2015

@author: Jesus Fuentes and Pablo Galaviz
'''

import logging
import wx  
import visvis as vv
import h5py
import numpy as np 

def init_slice_field(self, name,axis,origin, interpolate=False):
        
    f = self.get_amr_field(name)
    if axis == 0:
        return self.init_yz_slice(f, origin, interpolate) 
    elif axis == 1:
        return self.init_xz_slice(f, origin, interpolate) 
    elif axis == 2:
        return self.init_xy_slice(f, origin, interpolate) 
    else:
        logging.error("Axis out of range in grid_cpu")
        raise Exception

def init_yz_slice(self,field, origin, interpolate):
    
    self.yz_slice_ready = True
    
    self.idyz = np.argmin(np.abs(origin - self._x))
    return  self.y[self.idyz,:,:], self.z[self.idyz, :, :]

def init_xz_slice(self,field, origin, interpolate):
    
    self.xz_slice_ready = True
    
    self.idxz = np.argmin(np.abs(origin - self.y))
    return  self._x[:,self.idyz,:], self.z[:,self.idyz, :]

def init_xy_slice(self,field, origin, interpolate):
    
    self.xy_slice_ready = True
    
    self.idxy = np.argmin(np.abs(origin - self.z))
    return self._x[:,:,self.idxy], self.y[:,:,self.idxy]

    

def get_slice_field(self, name,axis, interpolate=False):
        
    f = self.get_amr_field(name)
    if axis == 0:
        return self.get_yz_slice(f, interpolate) 
    elif axis == 1:
        return self.get_xz_slice(f, interpolate) 
    elif axis == 2:
        return self.get_xy_slice(f, interpolate) 
    else:
        logging.error("Axis out of range in grid_cpu")
        raise Exception



def get_yz_slice(self,field, interpolate):       
    return  field[self.idyz,:,:]

def get_xz_slice(self,field, interpolate):       
    return  field[:,self.idxz,:]

def get_xy_slice(self,field, interpolate):       
    return  field[:,:,self.idxy]



class TabPanel1D(wx.Panel):
    def __init__(self, parent):

        self.h5file = None

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        label = wx.StaticText(self, label='Field', style=wx.ALIGN_CENTRE)
        self.fields_combo_box = wx.ComboBox(self, pos=(50, 30), choices=[''], style=wx.CB_READONLY)
        button = wx.Button(self, -1, 'Plot')
        button.Bind(wx.EVT_BUTTON, self.Plot)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)        
        sizer.Add(label, proportion=0, flag=wx.ALL, border=5)
        sizer.Add(self.fields_combo_box, proportion=0, flag=wx.ALL, border=5)
        sizer.Add(button, proportion=0, flag=wx.ALL, border=5)
        
        self.SetSizer(sizer)

        

    def SetH5File(self,h5file):
        
        self.h5file = h5file
                
        self.fields_combo_box.Clear()
        
        for k in h5file.keys():
            if k != 'coordinates':
                self.fields_combo_box.Append(k)

    def Plot(self, event):
        
        logging.info('Plot')
        
        # Clear it
        vv.clf()
        
        # Plot
        vv.plot([1,2,3,1,6])
        vv.legend(['this is a line'])        
        self.fig.DrawNow()

class TabPanel2D(wx.Panel):
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        txtOne = wx.TextCtrl(self, wx.ID_ANY, "Text")
        txtTwo = wx.TextCtrl(self, wx.ID_ANY, "")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(txtOne, 0, wx.ALL, 5)
        sizer.Add(txtTwo, 0, wx.ALL, 5)

        self.SetSizer(sizer)

    def SetH5File(self,h5file):
        print(h5file)


class NotebookPlot(wx.Notebook):

    def __init__(self, parent):
        wx.Notebook.__init__(self, parent, id=wx.ID_ANY, style=wx.BK_DEFAULT )

        self.parent = parent

        self.tabOne = TabPanel1D(self)
        self.AddPage(self.tabOne, "1D Ray")
   
        self.tabTwo = TabPanel2D(self)
        self.AddPage(self.tabTwo, "2D Slice")

    

    def SetH5File(self,h5file):
        self.tabOne.SetH5File(h5file)
        self.tabTwo.SetH5File(h5file)
        

class MainWindow(wx.Frame):
    def __init__(self):

        self.h5file = None
        
        wx.Frame.__init__(self, None, -1, "Cospy plot")

        self.MakeMenu()
        
        self.panel = wx.Panel(self)        
        self.notebook = NotebookPlot(panel)   
            
        Figure = app.GetFigureClass()
        self.fig = Figure(panel)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.ALL|wx.EXPAND, 5)
        sizer.Add(self.fig._widget, 2, wx.EXPAND)
        panel.SetSizer(sizer)
        self.Layout()
   
        # Finish
        self.Show()
        self.Maximize(True)
    
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
        
        
            
    def OnOpen(self, event):
    
    
        openFileDialog = wx.FileDialog(self, "Open HDF5 file", "", "",
                                       "HDF5 files (*.h5)|*.h5", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return     # the user changed idea...

        # proceed loading the file chosen by the user
        # this can be done with e.g. wxPython input streams:
        file_path = openFileDialog.GetPath()

        self.h5file = h5py.File(file_path)

        self.notebook.SetH5File(self.h5file)
            
    def OnQuit(self, e):
        self.Close()
    



if __name__ == '__main__':

    # Create a visvis app instance, which wraps a wx application object.
    # This needs to be done *before* instantiating the main window. 
    global app
    app = vv.use('wx')        
    
    wxApp = wx.App()    
    m = MainWindow()
    wxApp.MainLoop()



