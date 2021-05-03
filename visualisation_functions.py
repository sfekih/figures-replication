
import numpy as np
import pandas as pd

def less_points (y,cut_int=20,where_begin=200):
    """
    logarithm gives concentration of points in big distances
    we take less points to smooth the curve
    N:distance between each two points taken
    cut_int : cut interval : for example, if cut_int=20, we keep only 1 point out of 20
    where_begin : We do this process beginning from point 'where_begin' : where it begins to be croudy
    """
    y1=y.copy()
    nb_=y[where_begin::cut_int].shape[0]
    y1[where_begin:nb_+where_begin]=y[where_begin::cut_int]
    return y1[:nb_+where_begin]
    
def apply_median(y,N=50):
    """
    smoth curve using median
    y : to be plotted
    N : Number of items to use to smooth curve
    We choose not to do this process for small values because the curve is already smooth for small distances
    """
    y1=np.copy(y)
    
    for i in range (N,len(y)):
        if i<len(y)-N : y1[i]=np.median(y[i-N:i+N])
        else : y1[i]=np.median(y[i-N:len(y)])
    
    return y1

