# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:40:06 2019

@author: akulow
"""

import numpy as np
import h5py
import tensorflow as tf
import matplotlib as plt
import os
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from scipy.ndimage.filters import gaussian_filter

saveloc = 'C:\\Users\\akulow\\Documents\\Python Scripts\\Modules\\CodedApertures\\Bilder'
#pfad = 'M://Meins//Doktorarbeit//Coded Apertures//Experimental//20KW06//Rohdaten//'
#pfad = 'M://Meins//Doktorarbeit//Coded Apertures//Experimental//19KW48//Rohdaten//'
pfad = 'M://Meins//Doktorarbeit//Coded Apertures//Experimental//19KW40//Rohdaten//'
#pfad = 'M://Meins//Doktorarbeit//Coded Apertures//Experimental//20KW26//Rohdaten//20kw26_Oezlem//'
#pfad = 'M://Meins//Doktorarbeit//Coded Apertures//Experimental//20KW26//Rohdaten//20kw26_Wouter//'
pfad = './'
#------------------------------------------------------------------------------
def showdata(datei):
    datei_mit_endung = datei+'.h5'
    opendat = pfad+datei+'//'+datei_mit_endung
    
    hf = h5py.File(opendat,'r')
    key1 = 'Energy'
    key2 = 'Event'
    key3 = 'Header'
    key4 = 'Hit'
    key5 = 'Raw'
    key6 = 'Skip'
    key7 = 'Spectrum'
    
    n1 = hf.get('Energy')
    n1a = np.array(n1)
    n2 = hf.get('Event')
    n2a = np.array(n2)
    n3 = hf.get('Header')
    n4 = hf.get('Hit')
    n4a = np.array(n4)
    n5 = hf.get('Raw')
    n5a = np.array(n5)
    n6 = hf.get('Skip')
    n6a = np.array(n6)
    n7 = hf.get('Spectrum')
    
    # plt.pyplot.plot(n7)
    # plt.pyplot.show()
    
    what_to_see = input('What do you want to see?\n (Energy, Event, Hit, Raw, Skip)\n')
    
    if what_to_see == 'Raw':
        start = input('Choose start: ')
        start1 = int(start)
        end = input('Choose end: ')
        end1 = int(end)
        aoi = n5a
        roiarray = np.zeros([264,264])
        for i in range(264):
            for j in range(264):
                roiarray[i,j] = np.sum(aoi[i,j,start1:end1])
        plt.pyplot.matshow(roiarray)
    if what_to_see == 'Skip':
        plt.pyplot.matshow(n6a)
    if what_to_see == 'Energy':
        plt.pyplot.matshow(n1a)
    if what_to_see == 'Event':
        plt.pyplot.matshow(n2a)
    if what_to_see == 'Hit':
        plt.pyplot.matshow(n4a)
    
    hf.close()

#------------------------------------------------------------------------------
        
def getimages(datei,energy=1,event=1,hit=1,raw=1,skip=1):
    datei_mit_endung = datei+'.h5'
    opendat = pfad+datei+'//'+datei_mit_endung
    
    hf = h5py.File(opendat,'r')
    
    if energy==1:
        n1 = hf.get('Energy')
        n1a = np.array(n1)
    if event==1:
        n2 = hf.get('Event')
        n2a = np.array(n2)
    if hit==1:
        n4 = hf.get('Hit')
        n4a = np.array(n4)
    if raw==1:
        n5 = hf.get('Raw')
        n5a = np.array(n5)
        start = input('Choose start: ')
        start1 = int(start)
        end = input('Choose end: ')
        end1 = int(end)
        aoi = n5a
        roiarray = np.zeros([264,264])
        for i in range(264):
            for j in range(264):
                roiarray[i,j] = np.sum(aoi[i,j,start1:end1])
    if skip==1:
        n6 = hf.get('Skip')
        n6a = np.array(n6)
        
    hf.close()

    #nur 1 Rückgabewert
    if (energy==1 and event==0 and hit==0 and raw==0 and skip==0):
        return(n1a)
    if (energy==0 and event==1 and hit==0 and raw==0 and skip==0):
        return(n2a)
    if (energy==0 and event==0 and hit==1 and raw==0 and skip==0):
        return(n4a)
    if (energy==0 and event==0 and hit==0 and raw==1 and skip==0):
        return(roiarray)
    if (energy==0 and event==0 and hit==0 and raw==0 and skip==1):
        return(n6a)
    #2 Rückgabewerte
    if (energy==1 and event==1 and hit==0 and raw==0 and skip==0):
        return(n1a,n2a)
    if (energy==1 and event==0 and hit==1 and raw==0 and skip==0):
        return(n1a,n4a)
    if (energy==1 and event==0 and hit==0 and raw==1 and skip==0):
        return(n1a,roiarray)
    if (energy==1 and event==0 and hit==0 and raw==0 and skip==1):
        return(n1a,n6a)
    if (energy==0 and event==1 and hit==1 and raw==0 and skip==0):
        return(n2a,n4a)
    if (energy==0 and event==1 and hit==0 and raw==1 and skip==0):
        return(n2a,roiarray)
    if (energy==0 and event==1 and hit==0 and raw==0 and skip==1):
        return(n2a,n6a)
    if (energy==0 and event==0 and hit==1 and raw==1 and skip==0):
        return(n4a,roiarray)
    if (energy==0 and event==0 and hit==1 and raw==0 and skip==1):
        return(n4a,n6a)
    if (energy==0 and event==0 and hit==0 and raw==1 and skip==1):
        return(roiarray,n6a)
    #3 Rückgabewerte
    if (energy==1 and event==1 and hit==1 and raw==0 and skip==0):
        return(n1a,n2a,n4a)
    if (energy==1 and event==1 and hit==0 and raw==1 and skip==0):
        return(n1a,n2a,roiarray)
    if (energy==1 and event==1 and hit==0 and raw==0 and skip==1):
        return(n1a,n2a,n6a)
    if (energy==1 and event==0 and hit==1 and raw==1 and skip==0):
        return(n1a,n4a,roiarray)
    if (energy==1 and event==0 and hit==1 and raw==0 and skip==1):
        return(n1a,n4a,n6a)
    if (energy==1 and event==0 and hit==0 and raw==1 and skip==1):
        return(n1a,roiarray,n6a)
    if (energy==0 and event==1 and hit==1 and raw==1 and skip==0):
        return(n2a,n4a,roiarray)
    if (energy==0 and event==1 and hit==1 and raw==0 and skip==1):
        return(n2a,n4a,n6a)
    if (energy==0 and event==1 and hit==0 and raw==1 and skip==1):
        return(n2a,roiarray,n6a)    
    if (energy==0 and event==0 and hit==1 and raw==1 and skip==1):
        return(n4a,roiarray,n6a)
    #4 Rückgabewerte
    if (energy==1 and event==1 and hit==1 and raw==1 and skip==0):
        return(n1a,n2a,n4a,roiarray)
    if (energy==1 and event==1 and hit==1 and raw==0 and skip==1):
        return(n1a,n2a,n4a,n6a)
    if (energy==1 and event==1 and hit==0 and raw==1 and skip==1):
        return(n1a,n2a,roiarray,n6a)
    if (energy==1 and event==0 and hit==1 and raw==1 and skip==1):
        return(n1a,n4a,roiarray,n6a)
    if (energy==0 and event==1 and hit==1 and raw==1 and skip==1):
        return(n2a,n4a,roiarray,n6a)
    #5 Rückgabewerte
    if (energy==1 and event==1 and hit==1 and raw==1 and skip==1):
        return(n1a,n2a,n4a,roiarray,n6a)
           
#------------------------------------------------------------------------------   

def getnet(datei,roistart,roistop):
    datei_mit_endung = datei+'.h5'
    opendat = pfad+datei+'//'+datei_mit_endung
    
    hf = h5py.File(opendat,'r')
        
    n5 = hf.get('Raw')
    n5a = np.array(n5)
    
    hf.close()
    
    start1 = roistart
    end1 = roistop
    
    roiarray = np.zeros([264,264])
    for i in range(264):
        for j in range(264):
            roiarray[i,j] = np.sum(n5a[i,j,start1:end1])
            
    return(roiarray)
    
#------------------------------------------------------------------------------
    
def getspec(datei):
    datei_mit_endung = datei+'.h5'
    opendat = pfad+datei+'//'+datei_mit_endung
    
    hf = h5py.File(opendat,'r')
        
    n7 = hf.get('Spectrum')
    n7a = np.array(n7)
    
    hf.close()
    
    return(n7a)
    
#------------------------------------------------------------------------------
    
def normedspec(datei):
    """
    return a spectrum normalized by the nof frames (nn7a) or normalized by 
    seconds (nn7b)
    """
    datei_mit_endung = datei+'.h5'
    opendat = pfad+datei+'//'+datei_mit_endung
    
    hf = h5py.File(opendat,'r')

    hdat = hf['Header']
    noframes = hdat[0][9]
    mtime = hdat[0][2]
    stringtime = mtime.decode('utf-8')
    hours = int(stringtime[0:2])
    minutes = int(stringtime[3:5])
    seconds = int(stringtime[6:8])
    
    allseconds = hours*3600+minutes*60+seconds        
    
    n7 = hf.get('Spectrum')
    n7a = np.array(n7)
    
    nn7a = n7a/noframes
    nn7b = n7a/allseconds
    
    hf.close()
    
    return(nn7b)
    
#------------------------------------------------------------------------------

def showraw(datei,roistart,roistop):
    """
    Show the distribution of all energies in the range roistart:roistop of 
    datei. 
    Path of the file is global!
    
    Parameters
    ----------
    datei : file
        .h5-file with data from the CXC
    roistart : int
        start energy channel (0:1024) to be displayed
    roistop : int
        stop energy channel (0:1024) to be displayed

    Returns
    -------
    None.

    """
    
    datei_mit_endung = datei+'.h5'
    opendat = pfad+datei+'//'+datei_mit_endung
    
    hf = h5py.File(opendat,'r')
        
    n5 = hf.get('Raw')
    n5a = np.array(n5)
    
    hf.close()
    
    start1 = roistart
    end1 = roistop
    
    roiarray = np.zeros([264,264])
    for i in range(264):
        for j in range(264):
            roiarray[i,j] = np.sum(n5a[i,j,start1:end1])
            
    plt.pyplot.matshow(roiarray)
    
#------------------------------------------------------------------------------
    
def get_raw(datei):
    datei_mit_endung = datei+'.h5'
    opendat = pfad+datei+'//'+datei_mit_endung
    
    hf = h5py.File(opendat,'r')
        
    n5 = hf.get('Raw')
    n5a = np.array(n5)
    
    hf.close()
    
    return(n5a)
#------------------------------------------------------------------------------
    
def makenmf(datei):
    """
    perform a non negative matrix factorization to get the single components 
    of datei.
    Number of components has to be chosen.    
    """
    datei_mit_endung = datei+'.h5'
    opendat = pfad+datei+'//'+datei_mit_endung
    hf = h5py.File(opendat,'r')
        
    n5 = hf.get('Raw')
    n5a = np.array(n5)
    
    hf.close()
    
    #without scatter peak
    n5ac = n5a[:,:,50:500]
    n5arcs = np.reshape(n5ac,(264*264,450))
    
    n_comps = 10
    
    model = NMF(n_components=n_comps, random_state=0)
    W = model.fit_transform(n5arcs)
    H = model.components_
    
    for i in range(n_comps):
        w = np.reshape(W[:,i],(264,264))
        plt.pyplot.matshow(w)
    
    #extrahiere die interessanten Bilder    
    w1 = np.reshape(W[:,0],(264,264))
    w2 = np.reshape(W[:,1],(264,264))
    #w3 = np.reshape(W[:,2],(264,264))
    w3 = np.reshape(W[:,4],(264,264))
    w4 = np.reshape(W[:,5],(264,264))
    w5 = np.reshape(W[:,6],(264,264))
    
    
    return(w1,w2,w3,w4,w5,H)
    
    
#------------------------------------------------------------------------------
    
def params():
    #look for peak positions and according energies in the "spectrum" file
    peak1at = 30
    peak2at = 225
    energy1at = 0.989
    energy2at = 5.9225
    
    #peak1at = 36
    #peak2at = 63
    #energy1at = 5.471
    #energy2at = 5.496
    
    slope = (energy2at-energy1at)/(peak2at-peak1at)
    
    intercept = energy1at - slope*peak1at
    
    return(slope,intercept)
    
#------------------------------------------------------------------------------

def equation():
    slope,intercept = params()
    print('y = ', slope, '* x + ', intercept)

#------------------------------------------------------------------------------
    
def Energy2Channel(y):
    slope,intercept = params()
    x = (y-intercept)/slope
    
    return(x)
    
#------------------------------------------------------------------------------
    
def Channel2Energy(x):
    slope,intercept = params()
    y = slope*x + intercept
    
    return(y)
    
#------------------------------------------------------------------------------

def Espec(inspec):
    nofchannels = len(inspec)
    outarray = np.empty([nofchannels,2])
    inx = np.arange(nofchannels)
    slope,intercept = params()
    outx = inx*slope+intercept
    for i in range(nofchannels):
        outarray[i,0] = outx[i]
        outarray[i,1] = inspec[i]
    
    plt.pyplot.plot(outarray[:,0],outarray[:,1])
    
    return(outarray)

#------------------------------------------------------------------------------

def change_scale(inspec):
    nofchannels = len(inspec)
    outarray = np.empty([nofchannels,2])
    inx = np.arange(nofchannels)
    slope,intercept = params()
    outx = inx*slope+intercept
    for i in range(nofchannels):
        outarray[i,0] = outx[i]
        outarray[i,1] = inspec[i]
    
    return(outarray)
    
#------------------------------------------------------------------------------
    
def plot_Espec(inspec,Emin,Emax):
    """
    Parameters
    ----------
    inspec : numpy array 
        The spectrum of a measurement with the CXC, intensity per channel
    Emin : float
        The minimum energy [keV] for the plot.
    Emax : float
        The maximum energy [keV] for the plot.
   
    Returns
    -------
    None.

    """
    outarray = change_scale(inspec)
    fromhere = int(Energy2Channel(Emin))
    tohere = int(Energy2Channel(Emax))
    fig,ax = plt.pyplot.subplots()
    ax.plot(outarray[fromhere:tohere,0],outarray[fromhere:tohere,1])
    ax.set(xlabel='energy (keV)', ylabel = 'counts/second')
  
#------------------------------------------------------------------------------

def save_Espec(inspec,Emin,Emax,filename):
    """
    Parameters
    ----------
    inspec : numpy array 
        The spectrum of a measurement with the CXC, intensity per channel
    Emin : float
        The minimum energy for the plot.
    Emax : float
        The maximum energy for the plot.
    filename : string
        The filename to save the spectrum

    Returns
    -------
    None.

    """
    outarray = change_scale(inspec)
    fromhere = int(Energy2Channel(Emin))
    tohere = int(Energy2Channel(Emax))
    fig,ax = plt.pyplot.subplots()
    ax.plot(outarray[fromhere:tohere,0],outarray[fromhere:tohere,1])
    ax.set(xlabel='energy (keV)', ylabel = 'counts/second')
    os.chdir(saveloc)
    fig.savefig(filename)
    
#------------------------------------------------------------------------------

def get_ca(orgCApro):
    """
    
    Parameters
    ----------
    orgCApro : this is the 2d picture of the CA extracted from the recorded
    image. It should be as small as possible and noise should already be 
    removed.

    Returns
    -------
    A smeared mask with the same overall intensity in all the holes.

    """
    
    mask = orgCApro
    newmask = np.zeros(np.shape(mask))
    pos = []
    # list all the positions, where mask is not equal to zero
    for i in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[1]):
            if mask[i,j] > 0:
                pos.append((i,j))
                
    pos = np.asarray(pos)
    
    kmeans = KMeans(n_clusters=12, random_state=0).fit(pos)
    #print(kmeans.cluster_centers_)
        
    for i in range(12):
        holepoints = pos[kmeans.labels_==i]
       # print(holepoints)
        holesum = 0
        maskpos = []
        for j in range(len(holepoints)):
            holepos = holepoints[j]
            holesum = holesum+mask[holepos[0],holepos[1]]
            maskpos.append((holepos[0],holepos[1]))
        #print(holesum)
        for k in range(len(maskpos)):
            aktpos = maskpos[k]
            newmask[aktpos[0],aktpos[1]] = (mask[aktpos[0],aktpos[1]])/holesum*100
        
        newmask = gaussian_filter(newmask,0.4)
    
    return(newmask)