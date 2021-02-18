# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:23:37 2019
last changes: Sat Oct 26 2019
                deleted all the functions that are not necessary
                normalization of the transferfunction
              Thu Oct 31 2019
                -changed realtf_matrix_turn: as input a matrix with the hole
                positions is taken
                -inserted maskformatf
              Tue Dec 10 2019
                -added the timelapse, timelapse_rep and timelapse2 
              Wed Jan 22 2020
                -added agalistrecob (from all_in_one, adapted: no more 
                 dictionnaries, but arrays)
              Thu Mar 12 2020
                -added sparse matf
                -updated realtf_matrix_turn to avoid negative 
                 uniquedetpixelnumbers
              Fri Mar 13 2020
                -added c_sparse for the reconstruction with the sparse
                 matf
              Mon Mar 16 2020
                -added learning rate as variable to c and c_sparse 
              Fri Apr 24 2020
                -in matf_sparse changed nobpix from 
                nobpix = fovlength//sizeobpix
                to nobpix = int(np.ceil(fovlength/sizeobpix)) 
              Wed Apr 29 2020
                -removed everything that is not sparse
                -changed the buildmatf to buildmatf_for_reconstruction, this
                 returns now the tf mstf, so that it is no longer necessary
                 to do all this transformation everytime when calculating the
                 reconstruction (c_sparse) or the projection (sparse_projection)
	      Fri Jun 12 2020
	        -changed the genetic algorithm to sparse and matrix representation 
@author: akulow

The reconstruction of a coded picture is done with the function "c" which takes
as input the image, the transferfunction, a first guess, a difference 
tolerance, and a max_iter, and the learning rate. 
The image is the screen. This can be experimental data or the result of 
realmatrixRTT. It has to be in 1d. If it is not in 1d, use from221 to convert
it into 1d. The transferfunction contains all the geometrical arrangement:
detlength, sizedetpix, mask, masklength, sizemaskpix, maskthick, sizeobpix,
d1_mm, d2_mm. If all parameters of the experiment are known, you can calculate
the sizeobpix (for a given side length of the object (in pixel)) or the 
side length of the object (in pixel) for a given object pixel size.


- mura(x):
    Create a MURA of length x - original building rules.
    
- invmura(x):
    Create a MURA of length x - compared to the original building rules, the
    zeros and ones are exchanged.
    
- pmmask(x,hpix,mpix):
    Create a mask with non touching holes based on a MURA with length x
    x - rank of the MURA
    hpix - holepixel
    mpix - materialpixel
    (for different ratios of hole/materialpixelsize)
    
- pinvmask(x,hpix,mpix):
    Create a mask with non touching holes based on a MURA with length x
    x - rank of the MURA
    hpix - holepixel
    mpix - materialpixel
    (for different ratios of hole/materialpixelsize)
    
- randommask(length,n_holes):
    Create mask with length "Length" with "n_holes" randomly distributed holes
    
- maskformatf(mask,physlen,holesize):
    Create the parameters of the mask from a pixelmask.
    Take the mask (as array) with the physical length of physlen and the 
    holesize and return all important parameters of the mask: physical length, 
    number of holes, holesize and hole positions coordinates.
    
- reconstruct(image,antimask):
    reconstruct the image with a given antimask via convolution
    
- buildob(h=60, w=60, starth=300, startw=300, picture='Hydrangeas', dim = 0):
    build an object of length h x w from a picture
    h and w are the height and width of the built image
    picture is the picture that is used
    dim is the dimension of the picture
    starth and startw are the starting points of the image extract

- verticalstripes(h=100, w=100, strpix=2):
    Build an object of size h*w with horizontal stripes of width strpix pixel

- horizontalstripes(h=100, w=100, strpix=2):
    Build an object of size h*w with horizontal stripes of width strpix pixel

- chesspattern(h=100, w=100, pix=2):
    Build an object of size h*w with chess pattern

- from122(ob,high,wide):
    transform a 1d image into a 2d image

- from221(ob):
    transform a 2d image into a 1d image

- matf_sparse(detlength,sizedetpix,masksize,holesize,holepos,maskthick,
                       angle,sizeobpix,d1_mm,d2_mm):
    Calculate a matrix with the transferfunction as a sparse matrix.
    detlength: length of the detector in pixel (detector is assumed to be 
    square)
    sizedetpix: length of one detector pixel in microns (square)
    masksize: physical length of the mask in micrometer
    holesize: physical diameter of one (circular) hole (in micrometer)
    holepos: physical position of holes in the mask (in micrometer)
    maskthick: thickness of the mask in microns
    angle: angle of rotation of the mask in degree
    d1_mm: distance between object and mask in mm
    d2_mm: distance between mask and detector in mm
    one systemtransferfunction matrix is returned with dimensions
    (number of detector pixel)x(number of objectpixel)
    
- buildmatf_for_reconstruction():
    
    put in the values that you know and get the mstf in as tf tensor out
    
  
- c_sparse(image,matf,x0,tol=1e-6,max_iter=500,lr=0.5):
    Reconstruction of the object from the image with tensorflow optimization
    with the sparse matf.
    Image is the screen - has to be in 1d.
    here, the firstguess, x0, is the transpose of the x0 in c

- sparse_projection(ob,matf):
    project the object with the sparse matf to the screen

- testsetup():

- timelapse(screen,matf,firstguess,filename,number_of_series=10,
              steps_per_series=100,start_step=0):
    calculate the optimization severeal times (number of series), with starting
    every new series with the result of the previous one

- timelapse_rep(screen1,screen2,matf,firstguess,
                  filename1,filename2, number_of_series=10,
                  steps_per_series=100,start_step=0):
    like a timelapse for more than one screen
  
    """
    
#------------------------------------------------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import quadres
import matplotlib as plt
import sys
import os
import copy
import math
import time
import tensorflow as tf
#import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import coo_matrix
from collections import Counter
saveloc = 'C:\\Users\\akulow\\Documents\\Python Scripts\\Modules\\CodedApertures\\Bilder'

#------------------------------------------------------------------------------

def mura(x):
    """
    Create a MURA of length x - original building rules. This is not the 
    self supporting NTHT version.
    """
    #create an array for the MURA
    mura = np.zeros([x,x])
    res = quadres.quadres(x)

    # fill the mura with 1 and 0 
    for i in range(mura.shape[0]):
        for j in range(mura.shape[1]):
            if i in res:
                Ci = 1
            else:
                Ci = -1
                
            if j in res:
                Cj = 1
            else:
                Cj = -1
                
            if i == 0:
                mura[i,j] = 0
            elif (j == 0 and i != 0):
                mura[i,j] = 1
            elif (Ci*Cj == 1):
                mura[i,j] = 1
            else:
                mura[i,j] = 0
                
    return(mura)

#------------------------------------------------------------------------------

def invmura(x):
    """
    Create a MURA of length x - compared to the original building rules, the
    zeros and ones are exchanged.
    """
    # create an array filled with zeros for the MURA
    mura = np.zeros([x,x])
    res = quadres.quadres(x)

    # fill the mura with 1 and 0 
    for i in range(mura.shape[0]):
        for j in range(mura.shape[1]):
            if i in res:
                Ci = 1
            else:
                Ci = -1
                
            if j in res:
                Cj = 1
            else:
                Cj = -1
                
            if i == 0:
                mura[i,j] = 1
            elif (j == 0 and i != 0):
                mura[i,j] = 0
            elif (Ci*Cj == 1):
                mura[i,j] = 0
            else:
                mura[i,j] = 1
                
    return(mura)        

#------------------------------------------------------------------------------

def pmmask(x,hpix,mpix):
    """
    Create a mask with non touching holes based on a MURA with length x
    x - rank of the MURA
    hpix - holepixel
    mpix - materialpixel
    the absolute length (in pixel) of the mask is x*(hpix+mpix)
    (for different ratios of hole/material pixels)
    """
    m1 = mura(x)
    
    # Create an array for a self supporting mask
    dist = hpix+mpix
    nofpix = x*dist
    b1 = np.zeros([nofpix,nofpix])
    
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            if m1[i,j] == 1:
                newi = dist*i
                newj = dist*j
                b1[newi:newi+hpix,newj:newj+hpix] = m1[i,j]
                
    return(b1)
    
#------------------------------------------------------------------------------

def pinvmask(x,hpix,mpix):
    """
    Create a mask with non touching holes based on a MURA with length x
    x - rank of the MURA
    hpix - holepixel
    mpix - materialpixel
    the absolute length (in pixel) of the mask is x*(hpix+mpix)
    (for different ratios of hole/material pixels)
    """
    m1 = invmura(x)
    
    # Create an array for a self supporting mask
    dist = hpix+mpix
    nofpix = x*dist
    b1 = np.zeros([nofpix,nofpix])
    
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            if m1[i,j] == 1:
                newi = dist*i
                newj = dist*j
                b1[newi:newi+hpix,newj:newj+hpix] = m1[i,j]
                
    return(b1)
    
#------------------------------------------------------------------------------

def randommask(length,n_holes):
    """
    create mask with length "Length" with "n_holes" randomly distributed holes
    length is the length in pixels
    """
    r1 = np.zeros([length,length])
    holecount = n_holes
    
    while holecount > 0:
        rnx = np.floor(length*np.random.rand())
        ix = int(rnx)
        rny = np.floor(length*np.random.rand())
        iy = int(rny)
        if r1[ix,iy] == 0:
            r1[ix,iy] = 1
            holecount = holecount - 1
            
    return(r1)
    
#------------------------------------------------------------------------------

def maskformatf(mask,physlen,holesize):
    """
    Create the parameters of the mask from a pixelmask.
    Take the mask (as array) with the physical length of physlen and the 
    holesize and return all important parameters of the mask: physical length, 
    number of holes, holesize and hole position coordinates. All coordinates 
    are positive, i.e. the origin is in one corner of the mask.
    """
    nholes = int(np.sum(mask))
    pix = len(mask)
    pixlen = physlen/pix
    holepos = np.empty([nholes,2])
    count = 0
    for i in range(pix):
        for j in range(pix):
            if mask[i,j] != 0:
                holepos[count,0] = pixlen*i+0.5*pixlen
                holepos[count,1] = pixlen*j+0.5*pixlen
                count+=1

    return(physlen,nholes,holesize,holepos)

#------------------------------------------------------------------------------

def deltamask_H(detlength,sizedetpix,mask,masklength,sizemaskpix,
               angle,sizeobpix,d1_mm,d2_mm):
    
    """
    Calculate the deltamask after the building rule of Haboub et al. 
    (but without tiling): project the mask to the detector (with appropriate
    scaling, without consideration of the turning angle), and build the 
    antimask as follows:
        antimask[x,y] = 2*(mask[x,y]-0.5), and antimask[0,0] = 0
    one maskpixel is assigned to only one detectorpixel. 
        
    detlength: length of the detector in pixel (detector is assumed to be 
    square)
    sizedetpix: length of one detector pixel in microns (square)
    mask: pattern of the mask as array with 0 and 1 entries
    masklength: length of the mask in pixel (assumed to be square)
    sizemaskpix: length of one mask pixel in microns, assumed to be square
    maskthick: thickness of the mask in microns
    angle: angle of rotation of the mask in degree
    d1_mm: distance between object and mask in mm
    d2_mm: distance between mask and detector in mm
    one systemtransferfunction matrix is returned with dimensions
    (number of detector pixel)x(number of objectpixel)
    
    """
   
    #real sizes of system components in micrometer units
    detsize = detlength*sizedetpix
    masksize = masklength*sizemaskpix
    d1 = d1_mm*1000
    d2 = d2_mm*1000
    
    #field of view in mm
    fovlength = (detsize*d1-masksize*(d1+d2))/(d2)
        
    #field of view as multiple of detpixsize
    nobpix = fovlength//sizeobpix
    calcfov = nobpix*sizeobpix
    
    #offsets in microns always refered to the detector 
    offob = detsize/2 - calcfov/2
    offmask = detsize/2 - masksize/2
    
    #mask, detector and objectpixel in 1d
    nofpix_mask = round(masklength*masklength)
    
    #empty detector 2D
    det2d = np.zeros([detlength,detlength])
    
    #point in the middle of the object plane
    xpix_ob = nobpix/2
    ypix_ob = nobpix/2
    xpos_ob = xpix_ob*sizeobpix+0.5*sizeobpix
    ypos_ob = ypix_ob*sizeobpix+0.5*sizeobpix
    
    #raytracing for every mask pixel
    for j in range(nofpix_mask):
    #pixel and real position of the mask pixel
    
        xpix_mask = j//masklength
        ypix_mask = j%masklength
        xmid_mask = xpix_mask*sizemaskpix+0.5*sizemaskpix
        ymid_mask = ypix_mask*sizemaskpix+0.5*sizemaskpix
         
        #position difference between object and mask pixel
        dxmid = offmask+xmid_mask - (offob+xpos_ob)
        dymid = offmask+ymid_mask - (offob+ypos_ob)
            
        #detectorpositions hitted by the transmitted ray
        Dxmid = dxmid*((d2)/(d1))
        Dymid = dymid*((d2)/(d1)) 
        PosXmid = xmid_mask + offmask + Dxmid
        PosYmid = ymid_mask + offmask + Dymid
            
        #detector pixel in the area of the transmitted ray
        Detxmid = int(round(PosXmid//sizedetpix))
        Detymid = int(round(PosYmid//sizedetpix))
            
        if mask[xpix_mask,ypix_mask] == 1:
            Hvalue = 0
        if mask[xpix_mask,ypix_mask] == 0:
            Hvalue = -1
        if xpix_mask+ypix_mask == 0:
            Hvalue = 0
            
        #detector illumination
        if (Detxmid < detlength and Detymid < detlength):
            det2d[Detxmid,Detymid] = Hvalue
                
    #det2d = gaussian_filter(det2d,0.75)
    #plt.pyplot.matshow(det2d[75:175,75:175])        
        
    antimask = det2d
    return(antimask)

#------------------------------------------------------------------------------

def deltamask2(detlength,sizedetpix,mask,masklength,sizemaskpix,
               angle,sizeobpix,d1_mm,d2_mm):
    
    """
    Calculates an antimask from the given mask. The turning angle of the
    mask is considered. Project the mask to the detector and build then the 
    antimask as ifft(1/(fft(scaled mask))
    
    
    detlength: length of the detector in pixel (detector is assumed to be 
    square)
    sizedetpix: length of one detector pixel in microns (square)
    mask: pattern of the mask as array with 0 and 1 entries
    masklength: length of the mask in pixel (assumed to be square)
    sizemaskpix: length of one mask pixel in microns, assumed to be square
    maskthick: thickness of the mask in microns
    angle: angle of rotation of the mask in degree
    d1_mm: distance between object and mask in mm
    d2_mm: distance between mask and detector in mm
    one systemtransferfunction matrix is returned with dimensions
    (number of detector pixel)x(number of objectpixel)
    
    """
   
    #real sizes of system components in micrometer units
    detsize = detlength*sizedetpix
    masksize = masklength*sizemaskpix
    maskmiddle = masksize/2
    radan = angle*np.pi/180
    d1 = d1_mm*1000
    d2 = d2_mm*1000
    
    #field of view in mm
    fovlength = (detsize*d1-masksize*(d1+d2))/(d2)
        
    #field of view as multiple of detpixsize
    nobpix = fovlength//sizeobpix
    calcfov = nobpix*sizeobpix
    
    #offsets in microns always refered to the detector 
    offob = detsize/2 - calcfov/2
    offmask = detsize/2 - masksize/2
    
    #mask, detector and objectpixel in 1d
    nofpix_mask = round(masklength*masklength)
    
    det2d = np.zeros([detlength,detlength])
    
    #calculation of the transferfunction
    for j in range(nofpix_mask):
    #pixel and real position of the object and mask pixel
        xpix_ob = nobpix/2
        ypix_ob = nobpix/2
        xpos_ob = xpix_ob*sizeobpix+0.5*sizeobpix
        ypos_ob = ypix_ob*sizeobpix+0.5*sizeobpix
        xpix_mask = j//masklength
        ypix_mask = j%masklength
        xmid_mask = xpix_mask*sizemaskpix+0.5*sizemaskpix
        ymid_mask = ypix_mask*sizemaskpix+0.5*sizemaskpix
        xdist = xmid_mask-maskmiddle
        ydist = ymid_mask-maskmiddle
        Rmid_mask = np.sqrt(xdist**2+ydist**2)
        if Rmid_mask != 0:
            if ydist >= 0:
                alpharad = np.arccos(xdist/Rmid_mask)
            else:
                alpharad = -np.arccos(xdist/Rmid_mask)
        else:
            alpharad = 0
        newalpha = alpharad+radan
        if Rmid_mask != 0:
            xdist = Rmid_mask*np.cos(newalpha)
            ydist = Rmid_mask*np.sin(newalpha)
            xmid_mask = xdist+maskmiddle
            ymid_mask = ydist+maskmiddle
        
        #position difference between object and mask pixel
        if (mask[xpix_mask,ypix_mask] != 0):
            dxmid = offmask+xmid_mask - (offob+xpos_ob)
            dymid = offmask+ymid_mask - (offob+ypos_ob)
            
            #detectorpositions hitted by the transmitted ray
            Dxmid = dxmid*((d2)/(d1))
            Dymid = dymid*((d2)/(d1)) 
            PosXmid = xmid_mask + offmask + Dxmid
            PosYmid = ymid_mask + offmask + Dymid
            
            #detector pixel in the area of the transmitted ray
            Detxmid = int(round(PosXmid//sizedetpix))
            Detymid = int(round(PosYmid//sizedetpix))
            
            #print(Detxmid,Detymid)
            
            #detector illumination
            if (Detxmid < detlength and Detymid < detlength and Detxmid >= 0 and Detymid >= 0):
                det2d[Detxmid,Detymid] = 1
                
    #det2d = gaussian_filter(det2d,0.75)
    #plt.pyplot.matshow(det2d[75:175,75:175])        
    maskfft = np.fft.fft2(det2d)
    a = np.where(maskfft == 0)
    if (np.size(a) != 0):
        maskfft = maskfft+0.0001
        print('maskfft plus')

    antimask = np.fft.ifft2(1/maskfft)
    return(antimask)

#------------------------------------------------------------------------------
    
def reconstruct(image,antimask):
    """
    reconstruct the image with a given antimask via convolution
    """
    recob = np.fft.ifft2(np.fft.fft2(image)*np.fft.fft2(antimask))
    recob = np.fft.fftshift(recob)
    
    plt.pyplot.matshow(abs(recob))
    
    return(recob)     
    
#------------------------------------------------------------------------------
    
def buildob(h=60, w=60, starth=300, startw=300, picture='Hydrangeas', dim = 0):
    """
    build an object of length length from a picture
    
    h and width are the height and width of the built image
    picture is the picture that is used
    dim is the dimension of the picture
    starth and startw are the starting points of the image extract
    """
    if dim > 2 or dim < 0:
        print('dim has to be between 0 and 2')
        sys.exit()
    
    #os.chdir('M:\\Meins\\Doktorarbeit\\Coded Apertures\\Experimental\\19KW48\\Images')
    #os.chdir('C:\\Users\\akulow\\Documents\\Python Scripts\\Modules\\CodedApertures') #this is where the 
                                                       #pictures are located
    
    picname = picture+'.png'
    
    image = plt.image.imread(picname)
    im2d =  image[:,:,dim]
    shape = np.shape(im2d)
    width = shape[1]
    height = shape[0]
    maxh = h+starth
    maxw = w+startw
    if maxh > height or maxw> width:
        print('max height is', height, 'max width is', width)
        sys.exit()
    ob = im2d[starth:starth+h,startw:startw+w]
    
    control = copy.copy(im2d)
    
    control[starth-10:starth+h,startw-10:startw]=0
    control[starth-10:starth+h+10,startw+w:startw+w+10]=0
    control[starth-10:starth,startw-10:startw+w]=0
    control[starth+h:starth+h+10,startw-10:startw+w+10]=0
        
    plt.pyplot.matshow(control)
    
    plt.pyplot.matshow(ob)
    
    return(ob)
    
#------------------------------------------------------------------------------
    
def verticalstripes(h=100, w=100, strpix=2):
    """
    Build an object of size h*w with horizontal stripes of width strpix pixel
    """
    ob = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            if j%(2*strpix) == 0:
                ob[i,j:j+strpix] = 1
    
    return(ob)

#------------------------------------------------------------------------------
    
def horizontalstripes(h=100, w=100, strpix=2):
    """
    Build an object of size h*w with horizontal stripes of width strpix pixel
    """
    ob = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            if i%(2*strpix) == 0:
                ob[i:i+strpix,j] = 1
    
    return(ob)
    
#------------------------------------------------------------------------------
    
def chesspattern(h=100, w=100, pix=2):
    """
    Build an object of size h*w with chess pattern
    """
    ob = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            if ((i%(2*pix) == 0 and j%(2*pix)== 0) or i%(2*pix) == pix and j%(2*pix) == pix):
                #print(i,j)
                ob[i:i+pix,j:j+pix] = 1
                
    return(ob)
    
#------------------------------------------------------------------------------    

def from122(ob,high,wide):
    """
    transform a 1d image into a 2d image
    """
    shop = np.shape(ob)[0]
    test = high*wide
    if (shop != test):
        print('dimensions do not match!')
        sys.exit()
        
    ob2d = np.zeros([high,wide])
    for i in range(shop):
        x = i//wide
        y = i%wide
        ob2d[x,y] = ob[i]
        
    return(ob2d)   

#------------------------------------------------------------------------------    

def from221(ob):
    """
    transform a 2d image into a 1d image
    """
    high = np.shape(ob)[0]
    wide = np.shape(ob)[1]
       
    ob1d = np.zeros([high*wide])
    for i in range(high*wide):
        ob1d[i] = ob[i//wide,i%wide]
        
    return(ob1d)   
    
#------------------------------------------------------------------------------

def matf_sparse(detlength,sizedetpix,masksize,holesize,holepos,maskthick,
                       angle,sizeobpix,d1_mm,d2_mm,pcfov=1,norm=1):
    
    """
    Calculates a matrix with the transferfunction.
    The projection of every object pixel through every mask hole is calculated.
    
    detlength: length of the detector in pixel (detector is assumed to be 
    square)
    sizedetpix: length of one detector pixel in microns (square)
    masksize: physical length of the mask in micrometer
    nholes: number of holes in the mask
    holesize: physical diameter of one (circular) hole
    holepos: physical position of holes in the mask
    maskthick: thickness of the mask in microns
    angle: angle of rotation of the mask in degree
    d1_mm: distance between object and mask in mm
    d2_mm: distance between mask and detector in mm
    one systemtransferfunction matrix is returned with dimensions
    (number of detector pixel)x(number of objectpixel)
    pcfov: the whole field of view is considered (all that can fall through 
            the mask). if pcfov = 0 only the fully covered field of view is
            considered
    norm: the matf is normed, sum of the matf == 1
    
    """
   
    #begin
    print ('start at: ',time.asctime(time.localtime(time.time())))
    t0 = time.time()
    
    #real sizes of system components in micrometer units
    detsize = detlength*sizedetpix
    nholes = len(holepos)
    dm = maskthick
    maskmiddle = masksize/2
    radan = angle*np.pi/180
    d1 = d1_mm*1000-0.5*dm
    d2 = d2_mm*1000-0.5*dm
    masksizeforfov = np.sqrt(2)*masksize #maximum possible masksize (when a hole is in the corner)
    
    pixYs = np.zeros([detlength])
    
    #field of view in mm (all that can fall trhough the mask) (partially coded
    #field of view)
    if pcfov == 1:
        #fovlength = ((detsize*(d1+0.5*dm))+masksizeforfov*(d1+d2+dm))/(d2+0.5*dm) #maximal möglich, wenn Maske gedreht
        fovlength = ((detsize*(d1+0.5*dm))+masksize*(d1+d2+dm))/(d2+0.5*dm)
        
    #field of view in mm (fully coded field of view)
    else:
        #fovlength = ((detsize*(d1+0.5*dm))-masksizeforfov*(d1+d2+dm))/(d2+0.5*dm) #maximal möglich, wenn Maske gedreht
        fovlength = ((detsize*(d1+0.5*dm))-masksize*(d1+d2+dm))/(d2+0.5*dm)
        
    #field of view as multiple of detpixsize
    nobpix = int(np.ceil(fovlength/sizeobpix))
    calcfov = nobpix*sizeobpix
    
    #offsets in microns always refered to the detector 
    offob = detsize/2 - calcfov/2
    offmask = detsize/2 - masksize/2
    
    #maximal slope of a ray that can pass through a hole in the mask
    if dm != 0:
        slope = holesize/dm
        print('slope = ',slope)
    else:
        slope = (offmask+masksize)/d1
    
    #mask, detector and objectpixel in 1d
    nofpix_det = round(detlength*detlength)
    nofpix_ob = round(nobpix*nobpix)
    
    det2d = np.zeros([detlength,detlength])
    
    #lists for the indices and values
    indob = []
    inddet = []
    valu = []
    detsum = 0
    
    #calculation of the transferfunction
    for i in range(nofpix_ob):
        for j in range(nholes):
            #pixel and real position of the object and mask pixel
            xpix_ob = i//nobpix
            ypix_ob = i%nobpix
            xpos_ob = xpix_ob*sizeobpix+0.5*sizeobpix
            ypos_ob = ypix_ob*sizeobpix+0.5*sizeobpix
            xmid_mask = holepos[j,0]
            ymid_mask = holepos[j,1]
            xdist = xmid_mask-maskmiddle    #nur innerhalb der Maske
            ydist = ymid_mask-maskmiddle    #nur innerhalb der Maske
            Rmid_mask = np.sqrt(xdist**2+ydist**2)
            if Rmid_mask != 0:
                if ydist >= 0:
                    alpharad = np.arccos(xdist/Rmid_mask)
                else:
                    alpharad = -np.arccos(xdist/Rmid_mask)
            else:
                alpharad = 0
            newalpha = alpharad+radan
            if Rmid_mask != 0:
                xdist = Rmid_mask*np.cos(newalpha)
                ydist = Rmid_mask*np.sin(newalpha)
                xmid_mask = xdist+maskmiddle
                ymid_mask = ydist+maskmiddle
            
            Rneu = 0.5*holesize*(d1+d2+dm)/(d1+0.5*dm)
                        
            xmin_mask = xmid_mask-0.5*holesize
            xmax_mask = xmid_mask+0.5*holesize
            ymin_mask = ymid_mask-0.5*holesize
            ymax_mask = ymid_mask+0.5*holesize
            
            #position difference between object and mask pixel
            dxmid = offmask+xmid_mask - (offob+xpos_ob)
            dymid = offmask+ymid_mask - (offob+ypos_ob)
            dxmin = offmask+xmin_mask - (offob+xpos_ob)
            dxmax = offmask+xmax_mask - (offob+xpos_ob)
            dymin = offmask+ymin_mask - (offob+ypos_ob)
            dymax = offmask+ymax_mask - (offob+ypos_ob)
            if (xpos_ob + offob > xmax_mask + offmask):
                if (dxmax/d1 < -slope):
                    Dxmin = 0
                    Dxmax = 0
                else:
                    Dxmin = dxmin*(d2/(d1+dm))
                    Dxmax = dxmax*((d2+dm)/d1)
            if (xpos_ob + offob < xmin_mask + offmask):
                if (dxmin/d1 > slope):
                    Dxmin = 0
                    Dxmax = 0
                else:
                    Dxmin = dxmin*((d2+dm)/d1)
                    Dxmax = dxmax*(d2/(d1+dm))
            if (xpos_ob + offob <= xmax_mask + offmask and xpos_ob + offob >= xmin_mask + offmask):
                Dxmin = dxmin*(d2/(d1+dm))
                Dxmax = dxmax*(d2/(d1+dm))
                
            if (ypos_ob + offob > ymax_mask + offmask):
                if (dymax/d1 < -slope):
                    Dymin = 0
                    Dymax = 0
                else:
                    Dymin = dymin*(d2/(d1+dm))
                    Dymax = dymax*((d2+dm)/d1)
            if (ypos_ob + offob < ymin_mask + offmask):
                if (dymin/d1 > slope):
                    Dymin = 0
                    Dymax = 0
                else:
                    Dymin = dymin*((d2+dm)/d1)
                    Dymax = dymax*(d2/(d1+dm))
            if (ypos_ob + offob <= ymax_mask + offmask and ypos_ob + offob >= ymin_mask + offmask):
                Dymin = dymin*(d2/(d1+dm))
                Dymax = dymax*(d2/(d1+dm))
            
            if ((Dxmin == 0 and Dxmax == 0) or (Dymin == 0 and Dymax == 0)):
                continue
       
            #detectorpositions hitted by the transmitted ray
            Dxmid = dxmid*((d2+(dm/2))/(d1+(dm/2)))
            Dymid = dymid*((d2+(dm/2))/(d1+(dm/2)))
            PosXmid = xmid_mask + offmask + Dxmid
            PosYmid = ymid_mask + offmask + Dymid
            PosXmin = xmin_mask + offmask + Dxmin
            PosXmax = xmax_mask + offmask + Dxmax
            PosYmin = ymin_mask + offmask + Dymin
            PosYmax = ymax_mask + offmask + Dymax
            
#                print('Max:',PosXmax,PosYmax)
#                print('Min:',PosXmin,PosYmin)
#                print('Mitte:',PosXmid,PosYmid)
#                print('min:',PosXmin,PosYmin)
#                print('max:',PosXmax,PosYmax)

            #detector pixel in the area of the transmitted ray
            Detxmid = PosXmid//sizedetpix
            Detymid = PosYmid//sizedetpix
            Detxmin = PosXmin//sizedetpix
            Detymin = PosYmin//sizedetpix
            Detxmax = PosXmax//sizedetpix
            Detymax = PosYmax//sizedetpix
            
            #in this range the detpixel must be considered
            urange = int(Detxmax - Detxmin + 1)
            vrange = int(Detymax - Detymin + 1)
            
            #centers of the circles in the case of thick mask
            #circles that touch Pos(X/Y)(min/max)
            M1x = PosXmin + Rneu 
            M1y = PosYmid
            M2x = PosXmid
            M2y = PosYmin + Rneu
            M3x = PosXmax - Rneu 
            M3y = PosYmid
            M4x = PosXmid
            M4y = PosYmax - Rneu
            
            #all pixel in the area
            for u in range(urange):
                for v in range(vrange):
                    pixX = int(Detxmin + u)
                    pixY = int(Detymin + v)
                    if pixX < 0 or pixX > detlength or pixY < 0 or pixY > detlength:
                        continue
                    
#                        print('Pixel:',pixX,pixY)
                    #1.Quadrant
                    prop = 0
                    if (pixX-Detxmid>=0 and pixY-Detymid>=0):
#                            print(1)
                        farx = pixX*sizedetpix+sizedetpix
                        fary = pixY*sizedetpix+sizedetpix
                        nearx = pixX*sizedetpix
                        neary = pixY*sizedetpix
                        dist1far = np.sqrt((farx-M1x)**2+(fary-M1y)**2)
                        dist1near = np.sqrt((nearx-M1x)**2+(neary-M1y)**2)
                        dist2far = np.sqrt((farx-M2x)**2+(fary-M2y)**2)
                        dist2near = np.sqrt((nearx-M2x)**2+(neary-M2y)**2)
                        dist3far = np.sqrt((farx-M3x)**2+(fary-M3y)**2)
                        dist3near = np.sqrt((nearx-M3x)**2+(neary-M3y)**2)
                        dist4far = np.sqrt((farx-M4x)**2+(fary-M4y)**2)
                        dist4near = np.sqrt((nearx-M4x)**2+(neary-M4y)**2)
                        mindist = min(dist1near,dist2near,dist3near,dist4near)
                        maxdist = max(dist1far,dist2far,dist3far,dist4far)
#                            print(mindist,maxdist)
                        if (maxdist<=Rneu):
                            prop = 1
                        if (mindist>Rneu):
                            prop = 0
                        if (mindist<=Rneu and maxdist > Rneu and maxdist < Rneu+sizedetpix):
                            prop = 1-((maxdist-Rneu)/(np.sqrt(2)*sizedetpix))
#                                print(prop)
                    
                    #2.Quadrant
                    if (pixX-Detxmid<0 and pixY-Detymid>=0):
#                            print(2)
                        farx = pixX*sizedetpix
                        fary = pixY*sizedetpix+sizedetpix
                        nearx = pixX*sizedetpix+sizedetpix
                        neary = pixY*sizedetpix
                        dist1far = np.sqrt((farx-M1x)**2+(fary-M1y)**2)
                        dist1near = np.sqrt((nearx-M1x)**2+(neary-M1y)**2)
                        dist2far = np.sqrt((farx-M2x)**2+(fary-M2y)**2)
                        dist2near = np.sqrt((nearx-M2x)**2+(neary-M2y)**2)
                        dist3far = np.sqrt((farx-M3x)**2+(fary-M3y)**2)
                        dist3near = np.sqrt((nearx-M3x)**2+(neary-M3y)**2)
                        dist4far = np.sqrt((farx-M4x)**2+(fary-M4y)**2)
                        dist4near = np.sqrt((nearx-M4x)**2+(neary-M4y)**2)
                        mindist = min(dist1near,dist2near,dist3near,dist4near)
                        maxdist = max(dist1far,dist2far,dist3far,dist4far)
                        if (maxdist<=Rneu):
                            prop = 1
                        if (mindist>Rneu):
                            prop = 0
                        if (mindist<=Rneu and maxdist > Rneu and maxdist < Rneu+sizedetpix):
                            prop = 1-((maxdist-Rneu)/(np.sqrt(2)*sizedetpix))
                    
                    #3.Quadrant
                    if (pixX-Detxmid<0 and pixY-Detymid<0):   
#                            print(3)
                        farx = pixX*sizedetpix
                        fary = pixY*sizedetpix
                        nearx = pixX*sizedetpix+sizedetpix
                        neary = pixY*sizedetpix+sizedetpix
                        dist1far = np.sqrt((farx-M1x)**2+(fary-M1y)**2)
                        dist1near = np.sqrt((nearx-M1x)**2+(neary-M1y)**2)
                        dist2far = np.sqrt((farx-M2x)**2+(fary-M2y)**2)
                        dist2near = np.sqrt((nearx-M2x)**2+(neary-M2y)**2)
                        dist3far = np.sqrt((farx-M3x)**2+(fary-M3y)**2)
                        dist3near = np.sqrt((nearx-M3x)**2+(neary-M3y)**2)
                        dist4far = np.sqrt((farx-M4x)**2+(fary-M4y)**2)
                        dist4near = np.sqrt((nearx-M4x)**2+(neary-M4y)**2)
                        mindist = min(dist1near,dist2near,dist3near,dist4near)
                        maxdist = max(dist1far,dist2far,dist3far,dist4far)
                        if (maxdist<=Rneu):
                            prop = 1
                        if (mindist>Rneu):
                            prop = 0
                        if (mindist<=Rneu and maxdist > Rneu and maxdist < Rneu+sizedetpix):
                            prop = 1-((maxdist-Rneu)/(np.sqrt(2)*sizedetpix))
                            
                    #4.Quadrant
                    if (pixX-Detxmid>=0 and pixY-Detymid<0):
#                            print(4)
                        farx = pixX*sizedetpix+sizedetpix
                        fary = pixY*sizedetpix
                        nearx = pixX*sizedetpix
                        neary = pixY*sizedetpix+sizedetpix
                        dist1far = np.sqrt((farx-M1x)**2+(fary-M1y)**2)
                        dist1near = np.sqrt((nearx-M1x)**2+(neary-M1y)**2)
                        dist2far = np.sqrt((farx-M2x)**2+(fary-M2y)**2)
                        dist2near = np.sqrt((nearx-M2x)**2+(neary-M2y)**2)
                        dist3far = np.sqrt((farx-M3x)**2+(fary-M3y)**2)
                        dist3near = np.sqrt((nearx-M3x)**2+(neary-M3y)**2)
                        dist4far = np.sqrt((farx-M4x)**2+(fary-M4y)**2)
                        dist4near = np.sqrt((nearx-M4x)**2+(neary-M4y)**2)
                        mindist = min(dist1near,dist2near,dist3near,dist4near)
                        maxdist = max(dist1far,dist2far,dist3far,dist4far)
                        if (maxdist<=Rneu):
                            prop = 1
                        if (mindist>Rneu):
                            prop = 0
                        if (mindist<=Rneu and maxdist > Rneu and maxdist < Rneu+sizedetpix):
                            prop = 1-((maxdist-Rneu)/(np.sqrt(2)*sizedetpix))
                    
                    pixelint = prop
                    uniquedetpixelnumber = int(pixX*detlength + pixY)
                    
                    if (pixX < detlength and pixY < detlength and uniquedetpixelnumber > 0):
                         indob.append(i)
                         inddet.append(uniquedetpixelnumber)
                         valu.append(pixelint)
                         detsum += pixelint
                         det2d[pixX,pixY] += pixelint
                         pixYs[pixY]+=1
                         
                    #else:
                    #    print(pixX, pixY)
                                          
    #counter=Counter(indob)
    #result = [list(counter.keys()),list(counter.values())]
    #print(result[0])
    #print(nofpix_ob)
    stf = coo_matrix((valu,(indob,inddet)),shape=[nofpix_ob,nofpix_det])
    if norm == 1:
        stf = stf*nholes/np.sum(stf)
        #stf = nofpix_ob*stf*nholes/np.sum(stf)
    t1 = time.time()
    print('computation time: ',t1-t0) 
    return(det2d,stf,pixYs)

#------------------------------------------------------------------------------

def matf_to_mstf(matf):
    """

    Parameters
    ----------
    matf : cac.matf_sparse, tuple containing the detector image (projection of 
            uniformly illuminated object), the transferfunction (coo_matrix
            containing the contribution of each object pixel in the field of 
            view to each detector pixel, dimension: number of object pixels x 
            number of detector pixels) and the pixYs (number of hits at the 
            detector)
        
    DESCRIPTION.
    Calculate a tensorflow sparse tensor from the matf[1]
    Use this if you have calculated the matf_sparse instead of the 
    buildmatf_for_reconstruction
    The output of matf_to_mstf(matf_sparse) is the same as the output of 
    buildmatf_for_reconstruction

    Returns
    -------
    mstf, a sparse tensor containing the information of the transfer function.

    """
    mstf = matf[1]
    mstf = mstf.astype('float32')
    mstf = mstf.transpose()
    indices = np.mat([mstf.row,mstf.col]).transpose()
    datavalues = mstf.data.astype('float32')
    mstf = tf.SparseTensor(indices,datavalues,mstf.shape)
        
    return(mstf)
    

#------------------------------------------------------------------------------
    
def buildmatf_for_reconstruction(mask,detlength=264,detpixsize=48,masklength=1000,
                                 maskholesize=90,maskthick=25,angle=281.31,
                                 obpixsize=100,d1=10,d2=6.05,pcfov=1,norm=1):
    """
    mask is the mask in pixel (for example: pmmask(5,1,1))
    calculate the mstf as tensorflow tensor
    masklength is physical masklenth in microns
    maskholesize is pyhsical maskholesize in microns
    maskthick is physical maskthickness in microns
    angle is rotation angle of the mask in degree
    obpixsize is object pixel size in microns - the field of view is determined
        by the physical arrangement, the objectpixelsize defines the number
        of object pixels
    d1 is the distance between object and mask in mm
    d2 is the distance between mask and detector in mm
    """

    maskmatf = maskformatf(mask,masklength,maskholesize)
    holepos = maskmatf[3]

    matf = matf_sparse(detlength,detpixsize,masklength,maskholesize,
                              holepos,maskthick,angle,obpixsize,d1,d2,pcfov=pcfov,norm=norm)
    mstf = matf[1]
    mstf = mstf.astype('float32')
    mstf = mstf.transpose()
    indices = np.mat([mstf.row,mstf.col]).transpose()
    datavalues = mstf.data.astype('float32')
    mstf = tf.SparseTensor(indices,datavalues,mstf.shape)
        
    return(mstf)

#------------------------------------------------------------------------------

def c_sparse(image,mstf,x0, tol=1e-100, max_iter=500, lr=0.5):
    """
    Reconstruction of the object from the image with tensorflow optimization.
    Image is the screen - has to be in 1d.
    matf is the transferfunction
    x0 is the "first guess" - has to have the size of the object:
    tf.Variable(tf.random.normal([nofpixob,1]))
    lr is the learning rate
    ! The first guess is transposed compared to the first guess in cac.c
    """
    orgscreen = tf.constant(image, shape=[len(image),1],dtype='float32')
    screensum = tf.reduce_sum(orgscreen)
    obpix = mstf.shape[1]
    oblen = np.sqrt(obpix)
    oblint = int(oblen)
    detlen = mstf.shape[0]
   
    if detlen != len(image):
        print('Image has not the correct size!')
        sys.exit()

###############################################################################
#                                                                             #
#    VERY IMPORTANT!                                                          #
#                                                                             #
#    Choice of the optimizer and the parameters of the optimizer.             #
#                                                                             #
###############################################################################
###############################################################################
    opt = tf.optimizers.Adam(learning_rate=lr)
###############################################################################
###############################################################################
    
    #x = tf.Variable(tf.random.normal([obpix,1]))
    
    # with positivity constraint:
    x = tf.Variable(tf.random.normal([obpix,1]),constraint=lambda t: tf.nn.relu(t))
    
    fac = mstf
    
    x.assign(x0)
    xs = []
    diffs = []
    # To use minimize you have to define your loss computation as a funcction
    def cost():
        
#        x2 = tf.nn.relu(x)
#        y = tf.matmul(x2,fac)

        y = tf.sparse.sparse_dense_matmul(fac,x)
        retvar = tf.reduce_sum(tf.square(y-orgscreen))
        return retvar
    
    err = np.Inf # step error (banach), not actual error
    diff = np.Inf

    i = 0
    print ('start at: ',time.asctime(time.localtime(time.time())))
    t1 = time.time()
    while err > tol:
        x0 = x.numpy()
        #print(np.sum(x0))
        print(i,'/',max_iter)
        actscreen = tf.sparse.sparse_dense_matmul(fac,x)
        #noise = tf.Variable(tf.random.uniform([1,obpix]))
        diff = tf.reduce_mean(tf.sqrt(tf.square(actscreen-orgscreen)))
        print(diff)
        # store the actual "x" (object guess) to xs. This is
        # the values of all pixels
        xs.append(x.numpy())
        diffs.append(diff)

        opt.minimize(cost,var_list=[x])
        err = np.max(np.abs(x.numpy() - x0))
#        err = 1
#        print(err)
#        print(i)
        i += 1
        if i > max_iter:
            print(f'stopping at max_iter={max_iter}')
            # fill the developement of the pixel values (xs)
            # into an array to plot them
            xsarray = np.zeros([obpix,len(xs)])
            for i in range(len(xs)):
                npbla = xs[i]
                xsarray[:,i:i+1] = npbla
            plt.pyplot.plot(xsarray.transpose())
#            print('x:', x)
            t2 = time.time()
            dt = t2-t1
            print('time: ',dt,'seconds')
            solxs = xsarray[:,len(xs)-1]
            solstr = solxs
            sol2d = from122(solstr,oblint,oblint)
            return(sol2d,dt,diffs,xsarray.transpose(),i)
#            return(sol2d,dt,xsarray,i)
    print(f'stopping at err={err}<{tol}')
#    print('found: ',x.numpy)
    xsarray = np.zeros([obpix,len(xs)])
    for i in range(len(xs)):
        npbla = xs[i]
        xsarray[:,i:i+1] = npbla
    #plt.pyplot.plot(xsarray.transpose())
    t2 = time.time()
    dt = t2-t1
    print('time: ',dt,'seconds')
    solxs = xsarray[:,len(xs)-1]
    solstr = solxs
    sol2d = from122(solstr,oblint,oblint)        
    return(sol2d,dt,diffs,xsarray.transpose(),i)
#    return(sol2d,dt,xsarray,i)
    
#------------------------------------------------------------------------------
    
def sparse_projection(ob,mstf):
    """
    Compute the projection of the object with the sparse mstf.
    ob: 2d array of dimension mxn (m*n = nobpix)
    mstf: sparse mstf of dimension (nofdetpix x nofobpix)
    """
    t0 = time.time()
    
    inputoblen = len(ob)
    #inob = from221(ob)
    nobpix = np.shape(mstf)[1]
    oblen = int(np.sqrt(nobpix))
    obdiff = oblen-inputoblen
    dim1ob = from221(ob)
        
    if obdiff < 0:
        start = math.floor(abs(obdiff)/2)
        hob = ob[start:start+oblen,start:start+oblen]       
        dim1ob = from221(hob)
        
    elif obdiff > 0:
        hob = np.zeros([oblen,oblen])
        start = math.floor(obdiff/2)
        hob[start:start+inputoblen,start:start+inputoblen] = ob
        dim1ob = from221(hob)
        
    ob1dtf = tf.constant(dim1ob, shape=[nobpix,1],dtype='float32')
    proob = tf.sparse.sparse_dense_matmul(mstf,ob1dtf)
    
    t1 = time.time()

    print('Time to calculate the projection: ', t1-t0,' seconds')

    return(proob)

#------------------------------------------------------------------------------
    
def timelapse(screen,matf,firstguess,filename,number_of_series=20,
              steps_per_series=50,start_step=0):
    os.chdir(saveloc)
    fg_0 = firstguess
    obpix = np.shape(matf)[1]
    oblen = int(np.sqrt(obpix))
    for i in range(number_of_series):        
        rec = c_sparse(from221(screen),matf,fg_0,max_iter = steps_per_series,lr=75)
        fig1,ax1 = plt.pyplot.subplots()
        fig2,ax2 = plt.pyplot.subplots()
        fig3,ax3 = plt.pyplot.subplots()
        ax1.matshow(rec[0])
        ax2.plot(rec[3])
        ax3.plot(rec[2])
        steps = (i+1)*steps_per_series+start_step
        fileout1 = filename+str(steps)
        fileout2 = filename+str(steps)+'_dev'
        fileout3 = filename+str(steps)+'_error'
        fig1.savefig(fileout1)
        fig2.savefig(fileout2)
        fig3.savefig(fileout3)
        plt.pyplot.close('all')
        fg_0 = np.zeros([obpix,1],dtype = 'float32')
        for j in range(obpix):
            x = j//oblen
            y = j%oblen
            fg_0[j,0] = rec[0][x,y]
        fg_0 = tf.Variable(fg_0)
        
    rec[0].dump('latest_result.mat')
    return(rec)
        
#------------------------------------------------------------------------------
        
# def timelapse_rep(screen1,screen2,matf,firstguess,
#                   filename1,filename2, number_of_series=10,
#                   steps_per_series=100,start_step=0):
def timelapse_rep(screen,matf,firstguess,filename1,nofs1=2,nofs2=2,nofs3=2,
                  steps_per_series=200,start_step=0):

    os.chdir(saveloc)
    
    # sparse matf
    obpix = np.shape(matf[1])[0]
    oblen = int(np.sqrt(obpix))
    fg_0 = firstguess
    
    #rec1
    for i in range(nofs1):
        rec = c_sparse(from221(screen),matf,fg_0,max_iter=steps_per_series,lr=2)
        fig1,ax1 = plt.pyplot.subplots()
        fig2,ax2 = plt.pyplot.subplots()
        ax1.matshow(rec[0])
        ax2.plot(rec[2])
        steps = (i+1)*steps_per_series+start_step
        fileout1 = filename1+str(steps)
        fileout2 = filename1+str(steps)+'_error'
        fig1.savefig(fileout1)
        fig2.savefig(fileout2)
        plt.pyplot.close('all')
        fg_0 = np.zeros([obpix,1],dtype = 'float32')
        for m in range(obpix):
            x = m//oblen
            y = m%oblen
            fg_0[m,0] = rec[0][x,y]
        fg_0 = tf.Variable(fg_0)
        
    for j in range(nofs2):
        rec = c_sparse(from221(screen),matf,fg_0,max_iter=steps_per_series,lr=1)
        fig1,ax1 = plt.pyplot.subplots()
        fig2,ax2 = plt.pyplot.subplots()
        ax1.matshow(rec[0])
        ax2.plot(rec[2])
        steps = (j+1)*steps_per_series+start_step+steps_per_series*nofs1
        fileout1 = filename1+str(steps)
        fileout2 = filename1+str(steps)+'_error'
        fig1.savefig(fileout1)
        fig2.savefig(fileout2)
        plt.pyplot.close('all')
        fg_0 = np.zeros([obpix,1],dtype = 'float32')
        for o in range(obpix):
            x = o//oblen
            y = o%oblen
            fg_0[o,0] = rec[0][x,y]
        fg_0 = tf.Variable(fg_0)
        
    for k in range(nofs3):
        rec = c_sparse(from221(screen),matf,fg_0,max_iter=steps_per_series,lr=0.5)
        fig1,ax1 = plt.pyplot.subplots()
        fig2,ax2 = plt.pyplot.subplots()
        ax1.matshow(rec[0])
        ax2.plot(rec[2])
        steps = (k+1)*steps_per_series+start_step+steps_per_series*nofs1+steps_per_series*nofs2
        fileout1 = filename1+str(steps)
        fileout2 = filename1+str(steps)+'_error'
        fig1.savefig(fileout1)
        fig2.savefig(fileout2)
        plt.pyplot.close('all')
        fg_0 = np.zeros([obpix,1],dtype = 'float32')
        for p in range(obpix):
            x = p//oblen
            y = p%oblen
            fg_0[p,0] = rec[0][x,y]
        fg_0 = tf.Variable(fg_0)        
    
    #rec2
    # fg_0 = firstguess
    # for i in range(number_of_series):
    #     rec = c(from221(screen2),matf,fg_0,max_iter = steps_per_series)
    #     fig1,ax1 = plt.pyplot.subplots()
    #     fig2,ax2 = plt.pyplot.subplots()
    #     ax1.matshow(rec[0])
    #     ax2.plot(rec[3])
    #     steps = (i+1)*steps_per_series+start_step
    #     fileout1 = filename2+str(steps)
    #     fileout2 = filename2+str(steps)+'_dev'
    #     fig1.savefig(fileout1)
    #     fig2.savefig(fileout2)
    #     plt.pyplot.close('all')
    #     fg_0 = np.zeros([1,obpix],dtype = 'float32')
    #     for j in range(obpix):
    #         x = j//oblen
    #         y = j%oblen
    #         fg_0[0,j] = rec[0][x,y]
    #     fg_0 = tf.Variable(fg_0)
        
    #rec3
#    fg_0 = firstguess
#    for i in range(number_of_series):
#        rec = c(from221(screen3),matf,fg_0,max_iter = steps_per_series)
#        fig1,ax1 = plt.pyplot.subplots()
#        fig2,ax2 = plt.pyplot.subplots()
#        ax1.matshow(rec[0])
#        ax2.plot(rec[3])
#        steps = (i+1)*steps_per_series+start_step
#        fileout1 = filename3+str(steps)
#        fileout2 = filename3+str(steps)+'_dev'
#        fig1.savefig(fileout1)
#        fig2.savefig(fileout2)
#        plt.pyplot.close('all')
#        fg_0 = np.zeros([1,10816],dtype = 'float32')
#        for j in range(10816):
#            x = j//104
#            y = j%104
#            fg_0[0,j] = rec[0][x,y]
#        fg_0 = tf.Variable(fg_0)
#    
#    #rec4
#    fg_0 = firstguess
#    for i in range(number_of_series):
#        rec = c(from221(screen4),matf,fg_0,max_iter = steps_per_series)
#        fig1,ax1 = plt.pyplot.subplots()
#        fig2,ax2 = plt.pyplot.subplots()
#        ax1.matshow(rec[0])
#        ax2.plot(rec[3])
#        steps = (i+1)*steps_per_series+start_step
#        fileout1 = filename4+str(steps)
#        fileout2 = filename4+str(steps)+'_dev'
#        fig1.savefig(fileout1)
#        fig2.savefig(fileout2)
#        plt.pyplot.close('all')
#        fg_0 = np.zeros([1,10816],dtype = 'float32')
#        for j in range(10816):
#            x = j//104
#            y = j%104
#            fg_0[0,j] = rec[0][x,y]
#        fg_0 = tf.Variable(fg_0)
            
       
#------------------------------------------------------------------------------
        
def agalistrecob(screen,mstf,mutation=0.05,breakoff=0.05,maxstep=100,
              popsize=60,bestfrac=0.125,sigmap=100000):
    """
    The screen is 1d (sparse_projection or experimental data)
    
    The parents and offsprings
    the first population is completely arbitrary. 
    every individual is forward projected and compared to the screen.
    the difference between projected guess and screen is calculated as a 
    measure for the fitness
    the individuals are recombined and mutated to build offsprings.
    """
    
    ndetpix = np.shape(mstf)[0]
    original = tf.cast(tf.transpose(tf.reshape(tf.concat(popsize*[screen],0),shape=[popsize,ndetpix])),dtype='float32')
    
#    screenlen = np.shape(original)[0]
    
    steps=0
        
    nobpix = np.shape(mstf)[1]
       
#    parents = np.random.rand(nobpix,popsize)
    tfparents = tf.Variable(tf.random.uniform([nobpix,popsize]),dtype='float32')
 #   tfparents = tf.Variable(parents,shape=[nobpix,popsize],dtype='float32')
    enfants = np.zeros([nobpix,popsize])
    
#    evolution = 1
    best = 10
    bestlist = []
    nobest = round(bestfrac*popsize)
    
    #test the goodness 
    t0 = time.time()
    while (best >= breakoff and steps < maxstep):
        steps=steps+1
        
        proparents = tf.sparse.sparse_dense_matmul(mstf,tfparents)
        diffs = (proparents-original)**2
        meandiffs = tf.reduce_mean(diffs,0)
        tops = tf.nn.top_k(-meandiffs,k=nobest)
        #print(tops.indices)
        #posbest = tops.indices[0]
        valbest = -tops.values[0]
        bestlist.append(valbest)
        print(steps,valbest)
        
        #recombine and fill a new 'parents' list  
        enfants = np.empty([nobpix,popsize],dtype='float32')
        for i in range(popsize):
            rand1 = np.random.randint(0,nobest)
            rand2 = np.random.randint(0,nobest)
            cand1 = tfparents[:,tops.indices[rand1]]
            cand2 = tfparents[:,tops.indices[rand2]]
            goodmaman = meandiffs[tops.indices[rand1]]
            goodpapa = meandiffs[tops.indices[rand2]]
            partmaman = goodpapa/(goodmaman+goodpapa)
            proboffspring = tf.random.uniform([nobpix,],0,1)
            probmutation = tf.random.uniform([nobpix,],0,1)
            mask1 = tf.less(proboffspring,partmaman*tf.ones_like(cand1))
            heritage_maman = tf.multiply(cand1, tf.cast(mask1,cand1.dtype))
            mask2 = tf.greater_equal(proboffspring,partmaman*tf.ones_like(cand1))
            heritage_papa = tf.multiply(cand2, tf.cast(mask2,cand2.dtype))
            enfant = heritage_maman+heritage_papa
            mask3 = tf.less(probmutation,mutation*tf.ones_like(probmutation))
            sigma = sigmap*((goodmaman+goodpapa)/2)**2
            #print(sigma)
            #sigma = 0.1
            #print(sigma)
            mutationpix = tf.multiply(tf.random.normal([nobpix,],0,sigma,dtype='float32'), tf.cast(mask3,enfant.dtype))
            #genomutation = tf.multiply(enfant,tf.cast(mutationpix,enfant.dtype))
            enfant+=mutationpix
            #positivity constraint
            enfant = tf.nn.relu(enfant)
            enfants[:,i] = enfant
        
        tfparents = tf.Variable(enfants,shape=[nobpix,popsize],dtype='float32')
        
    t1 = time.time()    
    duration = t1-t0    
   
    return(bestlist,tfparents,duration,steps)
