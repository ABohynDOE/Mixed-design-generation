import numpy as np
import oapackage as oa
from HadamardTools import selectIsomorphismClasses
from sklearn.preprocessing import PolynomialFeatures


def isrep(array,full=True):
    """
    Check if a n-runs array is a replicate of a smaller design

    Parameters
    ----------
    array : array
        Design with n runs.
    
    full : bool
        Check for full replicate or if some rows are replicates. Default is full.

    Returns
    -------
    bool
        True if the array is a replicate.

    """
    val = False;
    if len(np.unique(array,axis=0)) == len(array)/2:
        print('Array is a full replicate');
        val = True;
    if full:
        if len(np.unique(array,axis=0)) != len(array):
            print('%i rows are replicated'%(len(array)-len(np.unique(array,axis=0))));
            val = True;
    return val; 

def make2lvl(array,colindex,zerocoding=False):
    subA = array[:,colindex];
    Dtemp = np.delete(array,colindex,1);
    if zerocoding:
        zeroval = 0;
    else:
        zeroval = -1;
    a = [];
    b = [];
    ab = [];
    for i in subA:
        if i == 0:
            a.append(zeroval);
            b.append(zeroval);
            ab.append(1);
        elif i == 1:
            a.append(zeroval);
            b.append(1);
            ab.append(zeroval);
        elif i == 2:
            a.append(1);
            b.append(zeroval);
            ab.append(zeroval);
        elif i == 3:
            a.append(1);
            b.append(1);
            ab.append(1);    
            
            
    return np.c_[np.array(a),np.array(b),np.array(ab),Dtemp];


def make4lvl(array,collist,colindex,fourlvlvarname,zerocoding=False):
    gencols = [collist[i] for i in colindex];
    udcollist = [fourlvlvarname] + [i for i in collist if i not in gencols]
    subA = array[:,colindex];
    Dtemp = np.delete(array,colindex,1);
    if zerocoding:
        lvlval = 0;
    else:
        lvlval = -1;
    mat = [];
    for i in range(array.shape[0]):
        if subA[i,0] == lvlval:
            if subA[i,1] == lvlval:
                mat.append(0);
            else:
                mat.append(1);
        else:
            if subA[i,1] == lvlval:
                mat.append(2);
            else:
                mat.append(3);
    return np.c_[np.array(mat),Dtemp],udcollist;


def findBF(array):
    """
    Return the indices of the basic factors columns.

    Parameters
    ----------
    array : numpy.ndarray
        Input design with k factors.

    Returns
    -------
    colindex : list
        List of the indices of the basic factors from 0 to k.

    """
    import random
    ind = [];
    for i in range(array.shape[1]):
        ind.append(i);
    colindex = random.choices(ind,k=3)
    return colindex


def yates(nfac):
    """
    Generate the yates order matrix for generalized interactions involving nfac factors

    Parameters
    ----------
    nfac : int
        Number of factors.

    Returns
    -------
    array
        Yates order matrix.

    """
    x = np.zeros((nfac,count(nfac)),dtype=int);
    index = 0;
    for i in range(0,count(nfac)):
        if (np.log2(i+1)).is_integer():
            x[index,i]=1;
            index+=1;
        else:
            colref=((2**np.floor(np.log2(i+1)))-1).astype(int);
            coladd=(i-colref-1).astype(int);
            x[:,i] = (x[:,colref]+x[:,coladd])%2;
    return x;            
    
   

def count(n):
    """
    Count number of ith order interactions for n factors, with i=1,..,n.

    Parameters
    ----------
    n : int
        number of factors.

    Returns
    -------
    int
        number of generalized interactions.

    """
    if n == 1:
        return 1
    else:
        return 2*count(n-1)+1


def genOA(nruns,n4lvl,n2lvl,strength=2):
    """
    Generate a list of OA of the array_link class.

    Parameters
    ----------
    nruns : int
        Number of runs.
    n4lvl : int
        Number of four-level factors.
    n2lvl : int
        Number of two-level factors.
    strength : int, optional
        Strength of the OA. The default is 2.

    Returns
    -------
    arrays : list
        List of the OA with runsize nruns.

    """
    number_of_factors = n4lvl+n2lvl;
    factor_levels = np.repeat(np.array([4,2]),[n4lvl,n2lvl],axis=0).tolist(); 
    arrayclass=oa.arraydata_t(factor_levels, nruns, strength, number_of_factors)
    ll2=[arrayclass.create_root()]
    arrays = ll2
    for extension_column in range(3, number_of_factors+1):
        extensions=oa.extend_arraylist(arrays, arrayclass)
        #print('extended to %d arrays with %d columns' % (len(extensions), extension_column))
        arrays=extensions
    return arrays

def n2fn(n):
    """
    Generates the generalized interactions up to n factors

    Parameters
    ----------
    n : int
        number of factors.

    Returns
    -------
    list
        List of the interactions.

    """
    if n==1:
        return ['a']
    else:
        l = n2fn(n-1)
        return  l+[chr(96+n)]+[i+chr(96+n) for i in l]
    
def sortLexo(my_string): 
  
    # Split the my_string till where space is found. 
    words = my_string.split() 
      
    # sort() will sort the strings. 
    words.sort() 
  
    # Iterate i through 'words' to print the words 
    # in alphabetical manner. 
    for i in words: 
        print( i )

def isoSelection(cat):
    al = [oa.array_link(x['design'].astype(int)) for x in cat];
    ind,isoClass = selectIsomorphismClasses(al, verbose=2);
    vals,zz = np.unique(ind, return_index=True);
    zz.sort();
    return [cat[x] for x in list(zz)];


def regCheck(array,n4lvl,zerocoding=True):         
    ar = array.copy();
    if n4lvl == 1:
        mat = make2lvl(ar,colindex=0,zerocoding=zerocoding);
        if zerocoding:
            mat = mat*2-1;
        bf = mat[:,[0,1,3]];
        # Compute the generalized intercations
        P = PolynomialFeatures(3,interaction_only=True,include_bias=False);
        G = P.fit_transform(bf);
        
        # Test each column left in D against G
        # and add the first orthogonal column to the design.
        bfFound = False;
        colTest = list(mat[:,1:].T);
        for i in colTest:
            if np.matmul(i.T,G).sum() == 0:
                bfFound = True;
                break
        
        if bfFound:
            bf = np.c_[bf,i];
        else:
            return False;
        
    elif n4lvl >= 2:
        mat= ar;
        for i in range(n4lvl):
            ind = i*3;
            mat = make2lvl(mat,colindex=ind,zerocoding=zerocoding);
        if zerocoding:
            mat = mat*2-1;
        bf = mat[:,[0,1,3,4]];

    # Generate full interactions with these 4 cols
    P = PolynomialFeatures(4,interaction_only=True,include_bias=False);
    B = P.fit_transform(bf);
    
    # Compute inner product
    A = np.matmul(B.T,mat);
    valTupl = (ar.shape[0],0,-(ar.shape[0]));
    val = True;
    for i in list(np.unique(A)):
        if int(i) not in valTupl:
            val =  False;
            break
    return val;

# def regCheck(array,zerocoding=True): 
#     ar = array.copy();
#     if zerocoding:
#         ar[:,1:] = ar[:,1:]*2-1;

#     # Select a,b and the first column of D as basic factors
#     D = ar[:,1:];
#     mat = make2lvl(ar,colindex=0);
#     bf = np.c_[mat[:,[0,1]],D[:,1]];
    
#     # Compute the generalized intercations
#     P = PolynomialFeatures(3,interaction_only=True,include_bias=False);
#     G = P.fit_transform(bf);
    
#     # Test each column left in D against G
#     # and add the first orthogonal column to the design.
#     bfFound = False;
#     colTest = list(D[:,1:].T);
#     for i in colTest:
#         if np.matmul(i.T,G).sum() == 0:
#             bfFound = True;
#             break
    
#     if bfFound:
#         bf = np.c_[bf,i];
#     else:
#         return False;
    
#     # Generate full interactions with these 4 cols
#     P = PolynomialFeatures(4,interaction_only=True,include_bias=False);
#     B = P.fit_transform(bf);
    
#     #Generate full two-level D
#     D = np.c_[mat,D];
    
#     # Compute inner product
#     A = np.matmul(B.T,D);
#     valTupl = (ar.shape[0],0,-(ar.shape[0]));
#     val = True;
#     for i in list(np.unique(A)):
#         if int(i) not in valTupl:
#             val =  False;
#             break
#     return val;

def permutation(x,a):
    return a[x:] + a[:x];


def generators(nbf):
    if nbf <= 0:
        raise ValueError('Cannot have negative or null number of basic factors')
    elif nbf > 4:
        raise ValueError('Not programmed yet')
    else:
        if nbf in [1,2,3]:
            return ['a','b','ab'];
        else:
            l = [];
            for i in range(1,int(nbf/2+1)):
                l.append([chr(96+(2*i-1)),chr(96+(2*i)),chr(96+(2*i-1))+chr(96+(2*i))]);
            for i in range(3):
                a = permutation(i,l[1]);
                l.append([l[0][i]+a[i] for i in range(len(l[0]))])
            return l