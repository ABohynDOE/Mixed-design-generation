import functions as fn
import numpy as np
import pandas as pd

def regularOA(n2lvl,n4lvl):
    # Generate all OAs
    OA = fn.genOA(16,n4lvl,n2lvl);
    Dcat = [];
    
    # For each OA, check regularity and store the design
    for i in OA:
        if fn.regCheck(np.array(i),n4lvl,zerocoding=True):
            Dcat.append({"design" : np.array(i),
                         "Nfac" : n2lvl+n4lvl}
                        );
        else:
            pass
    return Dcat,OA


sumtable = pd.DataFrame(columns = ['n4','n2','Dcat','Regcat']);
nruns = 16
for n4 in range(1,6):
    n2max = nruns-1-3*n4;
    for n2 in range(1,n2max+1):
        Regcat,Dcat = regularOA(n2,n4);
        sumtable = sumtable.append(pd.DataFrame([[n4,n2,len(Dcat),len(Regcat)]],
                                                    columns = ['n4','n2','Dcat','Regcat']),
                                        ignore_index=True);
        
sumtable.to_pickle('Summary/regularOA.pkl')


