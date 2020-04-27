import functions as fn
from pyDOE2 import fracfact
import numpy as np
import pandas as pd

def pseudoword(word,labellist):
    out = word.replace(labellist[0],'p');
    out = out.replace(labellist[1],'q');
    return out.replace('pq','r');

def normalword(word,labellist=['p','q','r']):
    out = word.replace(labellist[0],'a');
    out = out.replace(labellist[1],'b');
    return out.replace(labellist[2],'ab')

def getIntType(char,gen=['a','b','ab']):
    val=0
    l = len(char);
    if char.find(gen[0]) != -1 or char.find(gen[1]) != -1:
        val = 1;
    if char.find(gen[2]) != -1:
        val = 1;
        l -= 1;
    return (2*l+val)
    
def intTypeSort(collist,fourlvlgen=['a','b','ab']):
    out = [pseudoword(x,fourlvlgen) for x in collist];
    out = sorted(out);
    out = [normalword(x) for x in out]
    return sorted(out,key=getIntType)
    

nruns=16
nbf = np.log2(nruns).astype(int)
# Get starting Design 
fourlvlgen = ['a','b','ab'];
startDcols = [i for i in fn.n2fn(nbf) if len(i) == 1 or i in fourlvlgen] ;
startD = (fracfact(' '.join(startDcols))+1)//2;

# Make four-level variable
D,Dcols = fn.make4lvl(array = startD,
                collist = startDcols,
                colindex = [startDcols.index(i) for i in fourlvlgen],
                fourlvlvarname = 'A', zerocoding=True)
#Dcols = ['A'] + [i for i in startDcols if i not in fourlvlgen];

# Select remaining columns and sort the generators
candiCols = [i for i in fn.n2fn(nbf) if i not in startDcols];
#candiCols = intTypeSort(candiCols);

# Compute interaction type
inttype = [getIntType(i,fourlvlgen) for i in candiCols]
val,ind = np.unique(inttype,return_index=True)

# Retrieve corresponding columns
ind.sort();
nisoCols = [candiCols[x] for x in ind];

# Add them to the design
fullD = (fracfact(' '.join(fn.n2fn(nbf)))+1)//2;
Dcat = [];
for i in nisoCols:
    Dcat.append({"design" : np.c_[D,fullD[:,[fn.n2fn(nbf).index(i)]]],
                 "cols" : Dcols + [i],
                 "word" : i});

sumtable = pd.DataFrame([[4,len(Dcat),len(Dcat)]],
                        columns = ['NFac','NDesigns','NFinal']);

def addfactors(Dcat,naddedfac,sumtable):
    # Find all the lower generators
    for nfac in range(0,naddedfac-1):
        newcat = [];
        for i in Dcat:
            cols = [x for x in candiCols if candiCols.index(x) > candiCols.index(i['word']) ];
            for j in cols:
                newcat.append({"design" : np.c_[i['design'],fullD[:,[fn.n2fn(nbf).index(j)]]],
                         "cols" : i['cols'] +[j],
                         'word' : j});
                
        #Fill the summary table
        Dcat = fn.isoSelection(newcat);  
        sumtable = sumtable.append(pd.DataFrame([[nfac+5,len(newcat),len(Dcat)]],
                                                columns =['NFac','NDesigns','NFinal']),
                               ignore_index=True);      
    return Dcat,sumtable

Dcat,sumtable = addfactors(Dcat,10,sumtable);
sumtable.to_pickle('Summary/searchTable.pkl')
