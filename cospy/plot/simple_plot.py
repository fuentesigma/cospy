#!/usr/bin/env python
'''
Created on 20 May 2015

@author: Jesus Fuentes and Pablo Galaviz
'''

import visvis as vv

import argparse
import logging
import sys 
import numpy as np 

app = vv.use()
fig = vv.clf()
ax = vv.cla()


def plot_frame(data, t,ymin,ymax):
    
    dd = np.array(data)
    x=dd[:,0]
    y=dd[:,1]
    if np.max(y) > ymax:
        ymax=np.max(y)
    if np.min(y) < ymin:
        ymin=np.min(y)
    
    ax.Clear()
    vv.plot(x,y, lw=2)
    ax.Draw()
    ax.SetLimits(rangeY=(ymin,ymax))
    fig.DrawNow()

    return ymin, ymax

if __name__ == '__main__':

    ymax=0
    ymin=0

    logFormatter = logging.Formatter('Cospy %(levelname)s [%(asctime)s] | %(message)s')

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    
    
    epilog_text=" =============================\n     Author: Pablo Galaviz    \n     pablo.galaviz@me.com  \n =============================\n   License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html> \n   This is free software: you are free to change and redistribute it.\n   There is NO WARRANTY, to the extent permitted by law."

    parser = argparse.ArgumentParser(description='Cospy, cosmology amr solver', epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('input_file', help='Input parameter file.',metavar='intput.par')
    parser.add_argument('-s','--silent',action='store_false' ,help='Starts in silent mode, no message will be output.')

    args = parser.parse_args()

    if  args.silent :
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

    data=list()
    with open(args.input_file) as f:
        for line in f:
            if not line.strip() == '':
                if 'Time' in line:
                    t=float(line.split('=')[1])
                    if len(data) > 0:
                        ymin, ymax = plot_frame(data,t,ymin,ymax)
                        data=list()
                else :
                    data.append([float(x) for x in line.split('\t')])
                

