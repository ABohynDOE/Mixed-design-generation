#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:17:47 2017

@author: eendebakpt, schoen, vazquez
"""


#%% Load packages
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import oapackage
from oapackage.scanf import sscanf


#%% Function to read designs in a hadamard file

def _get_array(f, N=32, k=32, verbose=1):
        """ Return array from file """
        A = np.zeros((N, k), dtype=int)
        row = 0
        al = None
        while True:
            l = f.readline()
            if verbose >= 2:
                print(l)
                print('len(l) %d' % len(l))
            if len(l) == 0:
                break
            if l=='\n':
                continue

            if 0:
                for c in l:
                    print(c)
            if len(l) == (k / 4) + 1:
                l = l.strip()
                if verbose:
                    print(f'row {row}: {l}')
                x = bin(int(l, 16))[2:]
                A[row, :] = [int(y) for y in list(x)]
                row = row + 1
                if row == N:
                    break

        al = oapackage.makearraylink(A[:row,:])
        return al
    
def _get_integer(f, verbose=0):
        ii = -1
        while True:
            l = f.readline()
            if l=='\n':
                continue
            if len(l) == 0:
                break
            if verbose >= 2:
                print(l)
            try:
                i = sscanf(l, '%d')
            except:
                i = ''
            if len(i) > 0:
                if verbose:
                    print(i)
                ii = i[0]
                break
        return ii

def read_hadamard(f, narrays=1e9, N=32, k=32):
    """ Read all designs in a file with hex encoding """

    if isinstance(f, str):
        fid = open(f, 'rt')
    else:
        fid = f
    ll = []
    for ii in range(int(narrays)):
        oapackage.tprint('reading array %d' % ii, dt=.5)
        ii = _get_integer(fid)
        if ii == -1:
            break
        al = _get_array(fid, N, k)
        if al is not None:
            ll += [al]
    if isinstance(f, str):
        fid.close()
    print('read_hadamard: %d designs' % len(ll))
    return ll

#%% Hadamard matrix to OA

def construct_oa(al):
    """ Transform one Hadamard matrix into several OAs """

    oalist = []
    nfactors = al.n_columns
    altwo = al * 2 - 1
    for ii in range(nfactors):
        Hadmat = altwo.getarray()
        norm_col = Hadmat[:, ii]
        OAtwo = np.multiply(norm_col.reshape((-1, 1)), Hadmat)
        OAtwo = np.delete(OAtwo, ii, 1)
        A = oapackage.makearraylink((OAtwo + 1) / 2)
        oalist.append(oapackage.makearraylink(A))

    return(oalist)

#%% Select isomorphism classes

def selectIsomorphismClasses(sols, verbose=1):
    """ Select isomorphism classes from a list of designs 
    
    Args:
        sols (list of arrays)
        verbose (int)
    Return:
        indices (list of integers): indices of the isomorphism classes
        mm (list of arrays): the arrays in normal form

        
    """

    # perform check on array data type
    mm = []
    for ii, al in enumerate(sols):
        if verbose:
            oapackage.tprint('selectIsomorphismClasses: process aray %d/%d'  % ( ii, len(sols)), dt=4 )
        al = oapackage.makearraylink(al)

        tt = oapackage.reduceOAnauty(al, verbose >= 2)

        #pp, tt = reduceBliss(al, arrayclass, verbose >= 2)
        #tt = graph2arrayTransformation(pp, arrayclass)
        alx = tt.apply(al)
        mm.append(np.array(alx))
        pass

    # perform uniqueness check
    nn = len(mm)
    qq = np.array([None] * nn, dtype=object)
    for ii in range(nn):
        qq[ii] = mm[ii].flatten()


    if 0:
        # Trick to make unique work...
        # old code
        nx = qq[0].size
        dt = qq[0].dtype.descr * nx
        qqq = [np.array(q, dtype=dt) for q in qq]
        qqq = np.array(qq, dtype=dt)
        a, indices = np.unique(qqq, return_inverse=True)

    # Trick to make unique work...
    a, indices = np.unique(np.vstack( qq), axis=0, return_inverse=True)


    if verbose >= 1:
        print('selectIsomorphismClasses: %d reduced to %d' %
              (len(sols), np.unique(indices).size))

    return indices, mm
