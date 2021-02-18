#Quadratic residues module
"""
Return the quadratic residues of 'modul'.

Created 2018-09-19
Author akulow
"""

###############################################################################
import math

def quadres(x):
    """
    Make a list of the quadratic residues of x
    Here, all a = m² + tx are considered, not only the coprime ones. 
    """
    list1 = []
    tmax = math.floor(x/2)
    for i in range(1,tmax+1):
        i2 = i**2
        rest = i2%x
        if (rest not in list1):
            list1.append(rest)

    list1.sort()
    return(list1)       