import functions as fn
import numpy as np
from pyDOE2 import fracfact
import pandas as pd
from itertools import combinations,chain



def bruteForceD(nruns,n4lvl,n2lvl,catType='iso'):
    if catType not in ['full','iso']:
        raise ValueError('Only "full" "iso" catalogs available');
    if not np.log2(nruns).is_integer() or nruns < 4:
        raise ValueError('Nruns must be a positive power of 2');
    if n4lvl not in range(0,int((nruns-1)/3)):
        raise ValueError('N4lvl must be between 0 and (N-1)/3');
    if n2lvl not in range(0,int(nruns-1)):
        raise ValueError('N2lvl must be between 0 and (N-1)');
    if n4lvl*3+n2lvl > nruns-1:
        raise ValueError('Too many factors')
    l = iter([nruns,n4lvl,n2lvl]);
    if any(l) < 0:
        raise ValueError('Nruns, n4lvl and n2lvl must all be positive integer');
    
    nbf = np.log2(nruns).astype(int)
    # Get starting Design 
    fourlvlgens = fn.generators(nbf)[:n4lvl];
    fourlvlgen = list(chain(*fourlvlgens));
    startDcols = [i for i in fn.n2fn(nbf) if len(i) == 1 or i in fourlvlgen] ;
    startD = (fracfact(' '.join(startDcols))+1)//2;
    
    # Make four-level variable
    D=startD.copy();
    Dcols = startDcols.copy();
    varnamechr = 65
    for gen in fourlvlgens:
        D,Dcols = fn.make4lvl(D,Dcols,[Dcols.index(i) for i in gen],chr(varnamechr),zerocoding=True)
        varnamechr+=1;
    
    # Add added factors
    if n4lvl == 1:
        if n2lvl <= 2:
            Dcatalog =[{'design' : D[:,:n2lvl+1],
                       'cols' : Dcols[:n2lvl+1]}];
            return Dcatalog
        else:
            fullD = (fracfact(' '.join(fn.n2fn(nbf)))+1)//2;
            candiCols = [i for i in fn.n2fn(nbf) if i not in startDcols];
            enum = combinations(candiCols,n2lvl-2);
            Dcatalog = [];
            for i in list(enum):
               Dcatalog.append(
                   {"design" : np.c_[D,fullD[:,[fn.n2fn(nbf).index(x) for x in [y for y in i]]]],
                    "cols" : Dcols + [y for y in i],
                    "name" : 'D_4(' + str(n4lvl) + ')2(' + str(n2lvl) + ')'}
                   );
    else:
        fullD = (fracfact(' '.join(fn.n2fn(nbf)))+1)//2;
        candiCols = [i for i in fn.n2fn(nbf) if i not in startDcols];
        enum = combinations(candiCols,n2lvl);
        Dcatalog = [];
        for i in list(enum):
           Dcatalog.append(
               {"design" : np.c_[D,fullD[:,[fn.n2fn(nbf).index(x) for x in [y for y in i]]]],
                "cols" : Dcols + [y for y in i],
                "name" : 'D_4(' + str(n4lvl) + ')2(' + str(n2lvl) + ')'}
               );
               
    if catType == 'full':
        return Dcatalog
    elif catType == 'iso':
        return fn.isoSelection(Dcatalog);
        

cat = bruteForceD(16,2,5,'iso');

sumtable = pd.DataFrame(columns = ['n4','n2','Dcat','Isocat']);
nruns = 16
for n4 in range(1,6):
    n2max = nruns-1-3*n4;
    for n2 in range(1,n2max+1):
        Dcat = bruteForceD(nruns,n4,n2,'full');
        if len(Dcat) == 0:
            sumtable.append(pd.DataFrame([[n4,n2,0,0]]),ignore_index=True);
        else:
            isocat = fn.isoSelection(Dcat);
            sumtable = sumtable.append(pd.DataFrame([[n4,n2,len(Dcat),len(isocat)]],
                                                    columns = ['n4','n2','Dcat','Isocat']),
                                       ignore_index=True);

sumtable.to_pickle('Summary/bruteforce.pkl')

    
    