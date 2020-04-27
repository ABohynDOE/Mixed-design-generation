import functions as fn
import numpy as np
from itertools import chain 
from pyDOE2 import fracfact
import pandas as pd


def getIntType(char,gen=[['a','b','ab']]):
    val=0
    l = len(char);
    for g in gen:
        if char.find(g[0]) != -1 or char.find(g[1]) != -1:
            val = 1;
        if char.find(g[2]) != -1:
            val = 1;
            l -= 1;
    return (2*l+val)
    

def getstartD(nruns,n4lvl):
    # Starting two-level design
    nbf = np.log2(nruns).astype(int);
    fourlvlgens = np.array(fn.generators(nbf));
    fourlvlgen = list(chain(*fourlvlgens[0:n4lvl]));
    startDcols = [i for i in fn.n2fn(nbf) if len(i) == 1 or i in fourlvlgen];
    startD = (fracfact(' '.join(startDcols))+1)//2;
    
    # Make four-level variable
    varnamechr = 65
    D = startD.copy();
    Dcols = startDcols.copy();
    for gen in fourlvlgens[0:n4lvl]:
        D,Dcols = fn.make4lvl(D,Dcols,[Dcols.index(i) for i in gen],
                              chr(varnamechr),zerocoding=True);
        varnamechr+=1;
    return D, Dcols;


def addfirstcol(D,Dcols,n4lvl):
    # Get 4lvl generators
    nbf = np.log2(D.shape[0]).astype(int);
    fourlvlgens = np.array(fn.generators(nbf));
    fourlvlgen = list(chain(*fourlvlgens[0:n4lvl]));
    # Enumerate 2lvl generators
    candiCols = [x for x in fn.n2fn(nbf) if x not in fourlvlgen and len(x) > 1];
    val,ind = np.unique( [getIntType(i,fourlvlgens[0:n4lvl]) for i in candiCols],return_index=True);
    # Retrieve corresponding columns
    ind.sort();
    nisoCols = [candiCols[x] for x in ind];
    # Add to the design
    fullD = (fracfact(' '.join(fn.n2fn(nbf)))+1)//2;
    cat = [];
    for i in nisoCols:
        cat.append({"design" : np.c_[D,fullD[:,[fn.n2fn(nbf).index(i)]]],
                     "cols" : Dcols + [i],
                     "word" : i});
    fullcat = [];
    for i in candiCols:
        fullcat.append({"design" : np.c_[D,fullD[:,[fn.n2fn(nbf).index(i)]]],
                        "cols" : Dcols + [i],
                        "word" : i});
    return cat, candiCols


def addfactor(cat,candicols,nruns):
    nbf = np.log2(nruns).astype(int); 
    fullD = (fracfact(' '.join(fn.n2fn(nbf)))+1)//2;
    # Find all the lower generators
    fullcat = [];
    for i in cat:
        cols = [x for x in candicols if candicols.index(x) > candicols.index(i['word']) ];
        for j in cols:
            fullcat.append({"design" : np.c_[i['design'],fullD[:,[fn.n2fn(nbf).index(j)]]],
                     "cols" : i['cols'] +[j],
                     'word' : j});
            
    # Select non-isomorphic designs
    isocat = fn.isoSelection(fullcat);  
   
    return fullcat,isocat
    

def searchTable(nruns,n4lvl,n2lvl):
    D, Dcols = getstartD(nruns,n4lvl)
    # Define cases
    if n4lvl == 1:
        if n2lvl == 1:
            Dcat = {'design' : D[:,0:2],
                    'cols' : Dcols[0:2],
                    'words' : Dcols[0:2][-1]}
            return Dcat;
        elif n2lvl == 2:
            Dcat = {'design' : D,
                    'cols' : Dcols,
                    'words' : Dcols[-1]}
            return Dcat;
        else:
            Dcat, candi = addfirstcol(D,Dcols,n4lvl);
            if n2lvl == 3:
                return Dcat;
            else: 
                naddedfac = n2lvl-3;
                for nfac in range(naddedfac):
                    Dcat, isoDcat = addfactor(Dcat,candi,nruns)
                return Dcat;
    else:
        Dcat, candi = addfirstcol(D,Dcols,n4lvl);
        if n2lvl == 1:
            return Dcat;
        else:
            naddedfac = n2lvl-1;
            for nfac in range(naddedfac):
                Dcat, isoDcat = addfactor(Dcat,candi,nruns)
            return Dcat;
        


sumtable = pd.DataFrame(columns = ['n4','n2','Dcat','Regcat']);
nruns = 16
for n4 in range(1,6):
    n2max = nruns-1-3*n4;
    for n2 in range(1,n2max+1):
        Dcat = searchTable(nruns,n4,n2)
        sumtable = sumtable.append(pd.DataFrame([[n4,n2,len(Dcat),len(Dcat)]],
                                                    columns = ['n4','n2','Dcat','Regcat']),
                                        ignore_index=True);
        
#.to_pickle('Summary/searchtable.pkl')


