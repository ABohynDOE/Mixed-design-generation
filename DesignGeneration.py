# -*- coding: utf-8 -*-
#%% Packages
import functions as fn
import pyDOE2 as py
import numpy as np
import oapackage as oa
import pandas as pd
import itertools 
import os
from HadamardTools import selectIsomorphismClasses
from oapackage.oahelper import create_pareto_element
from sklearn.preprocessing import PolynomialFeatures

#%% Specific function
def regCheck(array,zerocoding=True): 
    ar = array.copy();
    if zerocoding:
        ar[:,1:] = ar[:,1:]*2-1;

    # Select a,b and the first column of D as basic factors
    D = ar[:,1:];
    mat = fn.make2lvl(ar[:,0])
    bf = np.c_[mat[:,[0,1]],D[:,1]];
    
    # Compute the generalized intercations
    P = PolynomialFeatures(3,interaction_only=True,include_bias=False);
    G = P.fit_transform(bf);
    
    # Test each column left in D against G
    # and add the first orthogonal column to the design.
    bfFound = False;
    colTest = list(D[:,1:].T);
    for i in colTest:
        if np.matmul(i.T,G).sum() == 0:
            bfFound = True;
            break
    
    if bfFound:
        bf = np.c_[bf,i];
    else:
        return False;
    
    # Generate full interactions with these 4 cols
    P = PolynomialFeatures(4,interaction_only=True,include_bias=False);
    B = P.fit_transform(bf);
    
    #Generate full two-level D
    D = np.c_[mat,D];
    
    # Compute inner product
    A = np.matmul(B.T,D);
    valTupl = (ar.shape[0],0,-(ar.shape[0]));
    val = True;
    for i in list(np.unique(A)):
        if int(i) not in valTupl:
            val =  False;
            break
    return val;

#%% General values 
nruns = 16;
n4 = 1

#%% Generate regular designs

### Saturated design 
genList = fn.n2fn(4);
gen = ' '.join(genList);
D2 = py.fracfact(gen);

### Four-level factor
D4 = fn.make4lvl((D2+1)//2,colindex=(0,1,2),zerocoding=True);
D4 = D4[D4[:,0].argsort()];

### Get base design
indColList = (0,1,5)
baseD = D4[:,indColList];
remainD = np.delete(D4,indColList,1);

### Iterate for each nbr of 2-level fac
Dsummary = pd.DataFrame();
allniD = [];
for i in range(1,remainD.shape[1]+1):
    c = list(itertools.combinations(range(remainD.shape[1]),i));
    Dlist = [];
    for j in range(len(c)):
        candiD = np.c_[baseD,remainD[:,c[j]]];
        Dlist.append(candiD);
    OADlist = [oa.array_link(x.astype(int)) for x in Dlist];
    
    ## Isomorphism selection
    ind,isoClass = selectIsomorphismClasses(OADlist, verbose=2);
    vals,zz = np.unique(ind, return_index=True);
    niD = [OADlist[x] for x in list(zz)];
    allniD.append(niD);
    
    ## Pareto optimality selection
    #Compute pareto values
    pareto=oa.ParetoMultiDoubleLong(); #Creates Pareto object
    vv=[];
    for ii, alx in enumerate(niD):
        g=alx.GWLP();
        v=[-g[3], -g[4]];
        vv+=[v]    
        mvec = create_pareto_element( v, pareto=pareto )
        r=pareto.addvalue(mvec, ii)
    #Select optimal arrays
    optD = [niD[x] for x in pareto.allindices()];

    ## Summary table
    sumdict = {'nbrArray' : len(c) , 'NonIsoArray' : len(niD), 'OptArray' : len(optD) , 'Fac' : i+3}
    Dsummary = Dsummary.append(sumdict, ignore_index = True);

    ## Write to folder
    if not os.path.exists('Designs'):
        os.mkdir("Designs");
    if not os.path.exists('Designs/regular'):
        os.mkdir("Designs/regular");
    filename = ("Designs/regular/"+str(i+3)+"factors.txt");
    oa.writearrayfile(filename, oa.arraylist_t(optD), oa.ATEXT);
    
# Cleaning 
del i,j,ii,alx,ind,isoClass,vals,zz,c,g,mvec,r,v,vv

#%% Generate OA
Dsummary2 = pd.DataFrame();
RegSummary = pd.DataFrame();
allOA =[];

for n2 in range(3,(nruns-1)-2*n4):
    OarrayList = fn.genOA(nruns,n4,n2,strength=2);
    OAlist = [];
    
    for i in range(len(OarrayList)):
        Oarray = np.array(OarrayList[i])  
        ## Check for replicated runs
        if np.unique(Oarray,axis=0).shape[0] == Oarray.shape[0]//2:
            Oarray = np.unique(Oarray,axis=0);
  
        ## Check for regularity
        if regCheck(Oarray):
            OAlist.append(oa.array_link(Oarray));
    
    allOA.append(OAlist)
    ## Pareto optimality selection
    #Compute pareto values
    pareto=oa.ParetoMultiDoubleLong(); #Creates Pareto object
    vv=[];
    for ii, alx in enumerate(OAlist):
        g=alx.GWLP();
        v=[-g[3], -g[4]];
        vv+=[v]    
        mvec = create_pareto_element( v, pareto=pareto )
        r=pareto.addvalue(mvec, ii)
    #Select optimal arrays
    optOA = [OAlist[x] for x in pareto.allindices()];
    
    ## Summary table
    sumdict = {'nbrOA' : len(OAlist) , 'OptOA' : len(optOA)}
    Dsummary2 = Dsummary2.append(sumdict, ignore_index = True);

    ## Write to folder
    if not os.path.exists('Designs/OA'):
        os.mkdir("Designs/OA");
    filename = ("Designs/OA/"+str(n2+1)+"factors.txt");
    oa.writearrayfile(filename, oa.arraylist_t(optOA), oa.ATEXT);

# Cleaning 
del i,n2,ii,alx,v,vv,mvec,r,g

#%% Outputs
Dsum = pd.concat([Dsummary,Dsummary2],axis=1);
del Dsummary,Dsummary2
Dsum = Dsum[["Fac",'nbrArray',"NonIsoArray", 'OptArray', 'nbrOA', 'OptOA']]
## Summary table to latex
print(Dsum.to_latex(index=False,float_format="%.0f"))

#%% Compare outputs
nbNI = [];
for i in range(len(allOA)):
    L = allniD[i] + allOA[i];
    niL = fn.isoSelection(L,zerocoding=True);
    nbNI.append(len(niL));
    