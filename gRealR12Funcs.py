import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
import scipy.special as sspecial
import mpmath
from mpmath import mp
import  pandas as pd
mp.dps=100


#this script contains functions for V(x)=x^{2}-igx^{5}, region I-II
#using g to calculate real eigenvalues

def ret5AdjacentPairs(g,E):
    """

    :param g: const
    :param E: trial eigenvalue
    :return: 5 adjacent pairs of roots, the first root x2 has smaller angle than the second root x1,
    return [[x2, x1]]
    """
    coefs = [-1j * g, 0, 0, 1, 0, -E]
    rootsAll = np.roots(coefs)
    # print(rootsAll)
    rootsSortedByAngle = sorted(rootsAll, key=np.angle, reverse=True)
    rst = []
    length = len(rootsSortedByAngle)
    for j in range(0, length):
        rst.append([rootsSortedByAngle[j], rootsSortedByAngle[(j + 1) % length]])

    return rst


def ret5SeparatedPairs(g,E):
    """

    :param g: const
    :param E: trial eigenfunction
    :return: 5 pairs of roots separated by another root, the first root x2 has smaller angle than the
    second root x1, return [[x2, x1]]
    """
    coefs = [-1j * g, 0, 0, 1, 0, -E]
    rootsAll = np.roots(coefs)
    rootsSortedByAngle = sorted(rootsAll, key=np.angle)
    rst = []
    length = len(rootsSortedByAngle)
    for j in range(0, length):
        rst.append([rootsSortedByAngle[j], rootsSortedByAngle[(j + 2) % length]])

    return rst

def f(z, g, E):
    '''
    :param g: const
    :param z: point on x2x1
    :param E: trial eigenvalue
    :return: f value
    '''
    return (1j * g * z ** 5 - z ** 2 + E) ** (1 / 2)
def fBranchOther(z,g,E):
    '''

    :param z: point on x2x1
    :param g: const
    :param E: trial eigenvalue
    :return: f value on another branch
    '''
    return -(1j * g * z ** 5 - z ** 2 + E) ** (1 / 2)

def integralQuadrature(g,E,x1,x2):
    '''

    :param g: const
    :param E: trial eigenvalue
    :param x1: ending point
    :param x2: starting point
    :return:
    '''
    a1 = np.real(x1)
    b1 = np.imag(x1)

    a2 = np.real(x2)
    b2 = np.imag(x2)


    slope = (b1 - b2) / (a1 - a2)
    gFunc=lambda y:f(y+1j*(slope*(y-a2)+b2),g,E)
    return (1+1j*slope)*mpmath.quad(gFunc,[a2,a1])

def integralQuadratureAnotherBranch(g,E,x1,x2):
    '''

    :param g: const
    :param E: trial eigenvalue
    :param x1: ending point
    :param x2: starting point
    :return:
    '''
    a1 = np.real(x1)
    b1 = np.imag(x1)

    a2 = np.real(x2)
    b2 = np.imag(x2)

    slope = (b1 - b2) / (a1 - a2)
    gFunc = lambda y: fBranchOther(y + 1j * (slope * (y - a2) + b2), g, E)
    return (1 + 1j * slope) * mpmath.quad(gFunc, [a2, a1])

def eqnFiveAdjacentPairs(EIn,*data):
    """
    computes adjacent pairs
    :param EIn: trial eigenvalue, real
    :param data: (n, g)
    :return:
    """
    n,g=data
    E = EIn[0]
    adjPairsAll=ret5AdjacentPairs(g,E)
    retValsCis=[]#in the order x2, x1
    retValsTrans=[]# in the order x1,x2
    retValsCisAnother=[]#in the order x2, x1, another branch
    retValsTransAnother=[]#in the order x1, x2, another branch

    #fill cis
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadrature(g,E,x1Tmp,x2Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsCis.append(rstTmp)
    #fill trans
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadrature(g,E,x2Tmp,x1Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsTrans.append(rstTmp)

    # #fill cis another
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadratureAnotherBranch(g,E,x1Tmp,x2Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsCisAnother.append(rstTmp)

    # #fill trans another
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadratureAnotherBranch(g,E,x2Tmp,x1Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsTransAnother.append(rstTmp)

    retCombined=retValsCis+retValsTrans+retValsCisAnother+retValsTransAnother
    retSorted=sorted(retCombined,key=np.abs)
    root0=retSorted[0]
    return np.abs(root0)

def computeOneSolutionWith5AdjacentPairs(inData):
    """

    :param inData: [n, g, Eest]
    :return: [n, g, E]
    """
    n, g, Eest = inData
    E ,infoDict,iErr,msg= sopt.fsolve(eqnFiveAdjacentPairs, Eest, args=(n, g), maxfev=100, xtol=1e-8,full_output=True)
    if iErr==1:
        return [n, g, E]
    else:
        return [n,g,-100]


def eqnFiveSepPairs(EIn,*data):
    """
    compute separated pairs
    :param EIn: trial eigenvalue, real
    :param data: (n, g)
    :return:
    """
    n, g = data
    E = EIn[0]
    sepPairsAll=ret5SeparatedPairs(g,E)
    retValsCis=[]#in the order x2,x1
    retValsTrans=[]# in the order x1, x2
    retValsCisAnother=[]#in the order x2, x1, another branch
    retValsTransAnother=[]#in the order x1, x2, another branch
    #fill cis
    for pairTmp in sepPairsAll:
        x2Tmp, x1Tmp=pairTmp
        intValTmp=integralQuadrature(g,E,x1Tmp,x2Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsCis.append(rstTmp)
    #fill trans
    for pairTmp in sepPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadrature(g,E,x2Tmp,x1Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsTrans.append(rstTmp)
    #fill cis another
    for pairTmp in sepPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadratureAnotherBranch(g,E,x1Tmp,x2Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsCisAnother.append(rstTmp)
    #fill trans another
    for pairTmp in sepPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadratureAnotherBranch(g,E,x2Tmp,x1Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsTransAnother.append(rstTmp)

    retCombined=retValsCis+retValsTrans+retValsCisAnother+retValsTransAnother
    retSorted=sorted(retCombined,key=np.abs)
    root0=retSorted[0]
    return np.abs(root0)



def computeOneSolutionWith5SepPairs(inData):
    """

    :param inData: [n, g, Eest]
    :return: [n, g,E]
    """
    n, g, Eest = inData
    E= sopt.fsolve(eqnFiveSepPairs, Eest, args=(n, g), maxfev=100, xtol=1e-6)[0]
    return [n, g,E]