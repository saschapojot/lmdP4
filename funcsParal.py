import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
import scipy.special as sspecial
import mpmath
from mpmath import mp
mp.dps=100


def Sep5Pairs(lmd,F):
    """

    :param lmd: const
    :param F: trial eigenvalue
    :return: 5 separated pair of roots, the first has smaller angle than the second, the order is [x2, x1]
    """
    coefs = [-1j, 0, 0, lmd, 0, -F]
    rootsAll = np.roots(coefs)
    rootsSortedByAngle = sorted(rootsAll, key=np.angle)
    rst = []
    for j in range(0, len(rootsSortedByAngle)):
        rst.append([rootsSortedByAngle[j], rootsSortedByAngle[(j + 2) % len(rootsSortedByAngle)]])
    return rst

def Adj5Pairs(lmd,F):
    """

    :param lmd: const
    :param F: trial eigenvalue
    :return: 5 adjacent pair of roots, the first has smaller angle than the second, the order is [x2, x1]
    """
    coefs = [-1j, 0, 0, lmd, 0, -F]
    rootsAll = np.roots(coefs)
    rootsSortedByAngle = sorted(rootsAll, key=np.angle)
    rst = []
    for j in range(0, len(rootsSortedByAngle)):
        rst.append([rootsSortedByAngle[j], rootsSortedByAngle[(j + 1) % len(rootsSortedByAngle)]])
    return rst

def f(z,lmd,F):
    """

    :param z: point on x2x1
    :param lmd: const
    :param F: trial eigenvalue
    :return: f value
    """
    return (1j*z**5-lmd*z**2+F)**1/2

def fBranchAnother(z,lmd,F):
    """

    :param z: point on x2x1
    :param lmd: const
    :param F: trial eigenvalue
    :return: f on another branch
    """

    return -(1j*z**5-lmd*z**2+F)**1/2


def integralQuad(lmd,F,x1,x2):
    """

    :param lmd: const
    :param F: trial eigenvalue
    :param x1: ending point
    :param x2: starting point
    :return:
    """
    a1 = np.real(x1)
    b1 = np.imag(x1)

    a2 = np.real(x2)
    b2 = np.imag(x2)


    slope = (b1 - b2) / (a1 - a2)
    gFunc=lambda y:f(y+1j*(slope*(y-a2)+b2),lmd,F)
    return (1+1j*slope)*mpmath.quad(gFunc,[a2,a1])


def integralQuadBranchAnother(lmd,F,x1,x2):
    """

    :param lmd: const
    :param F: trial eigenvalue
    :param x1: ending point
    :param x2: starting point
    :return:
    """

    a1 = np.real(x1)
    b1 = np.imag(x1)

    a2 = np.real(x2)
    b2 = np.imag(x2)

    slope = (b1 - b2) / (a1 - a2)
    gFunc = lambda y: fBranchAnother(y + 1j * (slope * (y - a2) + b2), lmd, F)
    return (1 + 1j * slope) * mpmath.quad(gFunc, [a2, a1])

def eqn5AdjPairs(FIn,*data):
    """
    computes adjacent pairs
    :param FIn: trial eigenvalue, in the form of [re, im]
    :param data: (n, lmd)
    :return:
    """
    n,lmd=data
    F=FIn[0]+FIn[1]
    adjPairsAll=Adj5Pairs(lmd,F)
    retValsCis=[]#in the order x2, x1
    retValsTrans=[]# in the order x1,x2
    retValsCisAnother=[]#in the order x2, x1, another branch
    retValsTransAnother=[]#in the order x1, x2, another branch
    #fill cis
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuad(lmd,F,x1Tmp,x2Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsCis.append(rstTmp)
    #fill trans
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuad(lmd,F,x2Tmp,x1Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsTrans.append(rstTmp)

    #fill cis another
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadBranchAnother(lmd,F,x1Tmp,x2Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsCisAnother.append(rstTmp)

    #fill trans another
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadBranchAnother(lmd,F,x2Tmp,x1Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsTransAnother.append(rstTmp)
    retCombined=[]
    retCombined.extend(retValsCis)
    retCombined.extend(retValsTrans)
    retCombined.extend(retValsCisAnother)
    retCombined.extend(retValsTransAnother)
    retSorted=sorted(retCombined,key=np.abs)
    root0=retSorted[0]
    return np.real(root0), np.imag(root0)

def computeOneSolutionWith5AdjPairs(inData):
    """

    :param inData: [n, lambda, Fest]
    :return: [n, lambda, re(F), im(F)]
    """
    n, lmd,FEst=inData
    FVecTmp=sopt.fsolve(eqn5AdjPairs,[np.real(FEst),np.imag(FEst)],args=(n,lmd),maxfev=100,xtol=1e-3)
    return [n, lmd, FVecTmp[0], FVecTmp[1]]

def selectTrueRoot(exactRoots,asympRoot):
    """

    :param exactRoots: roots solved by np.roots
    :param asympRoot: asymtotic root
    :return:
    """
    diffAbs=[np.abs(elem-asympRoot) for elem in exactRoots]
    inds=np.argsort(diffAbs)
    return exactRoots[inds[0]]

def eqnLargeLambdaLowerSymPair(FIn,*data):
    """

    :param FIn:trial eigenvalue, in the form of [re, im]
    :param data: (n, lmd)
    :return:
    """
    n, lmd = data
    F = FIn[0] + FIn[1]
    coefEqn=[-1j,0,0,lmd,0,-F]
    trueRootsAll=np.roots(coefEqn)

    ###############asymp roots when large lambda >>|F|^(3/5)
    y3n0s = [np.exp(1j * (4 * n * np.pi - np.pi) / 6) * lmd ** (1 / 3) for n in range(0, 3)]

    y3ns = [ytmp + F * 1 / (2 * lmd * ytmp - 5 * 1j * ytmp ** 4) for ytmp in y3n0s]

    selectedRoots=[selectTrueRoot(trueRootsAll,elem) for elem in y3ns]

    sortedRootsByAngle=sorted(selectedRoots,key=np.angle)
    #taking lower pair
    x2Tmp=sortedRootsByAngle[0]
    x1Tmp=sortedRootsByAngle[1]
    retVals=[]
    #cis
    tmpCis=integralQuad(lmd,F,x1Tmp,x2Tmp)-(n+1/2)*np.pi
    retVals.append(tmpCis)

    #trans
    tmpTrans=integralQuad(lmd,F,x2Tmp,x1Tmp)-(n+1/2)*np.pi
    retVals.append(tmpTrans)
    #cis another
    tmpCisAnother=integralQuadBranchAnother(lmd,F,x1Tmp,x2Tmp)-(n+1/2)*np.pi
    retVals.append(tmpCisAnother)
    #trans another
    tmpTransAnother=integralQuadBranchAnother(lmd,F,x2Tmp,x1Tmp)-(n+1/2)*np.pi
    retVals.append(tmpTransAnother)

    sortedRet=sorted(retVals,key=np.abs)
    root0=sortedRet[0]
    return np.real(root0),np.imag(root0)

def computeOneSolWithLargeLambdaLower(inData):
    """

    :param inData: [n, lambda, Fest]
    :return: [n, lambda, re(F), im(F)]
    """
    n, lmd, FEst = inData
    FVecTmp = sopt.fsolve(eqnLargeLambdaLowerSymPair, [np.real(FEst), np.imag(FEst)], args=(n, lmd), maxfev=100, xtol=1e-6)
    return [n, lmd, FVecTmp[0], FVecTmp[1]]


def eqnLargeLambdaUpperSymPair(FIn,*data):
    """

    :param FIn: trial eigenvalue, in the form of [re, im]
    :param data: (n, lmd)
    :return:
    """
    n, lmd = data
    F = FIn[0] + FIn[1]
    coefEqn = [-1j, 0, 0, lmd, 0, -F]
    trueRootsAll = np.roots(coefEqn)

    zAll=[(F/lmd)**(1/2),-(F/lmd)**(1/2)]
    selectedRoots=[selectTrueRoot(trueRootsAll,zAll)]
    sortedRootsByAngle = sorted(selectedRoots, key=np.angle)
    # taking lower pair
    x2Tmp = sortedRootsByAngle[0]
    x1Tmp = sortedRootsByAngle[1]
    retVals = []
    # cis
    tmpCis = integralQuad(lmd, F, x1Tmp, x2Tmp) - (n + 1 / 2) * np.pi
    retVals.append(tmpCis)

    # trans
    tmpTrans = integralQuad(lmd, F, x2Tmp, x1Tmp) - (n + 1 / 2) * np.pi
    retVals.append(tmpTrans)
    # cis another
    tmpCisAnother = integralQuadBranchAnother(lmd, F, x1Tmp, x2Tmp) - (n + 1 / 2) * np.pi
    retVals.append(tmpCisAnother)
    # trans another
    tmpTransAnother = integralQuadBranchAnother(lmd, F, x2Tmp, x1Tmp) - (n + 1 / 2) * np.pi
    retVals.append(tmpTransAnother)

    sortedRet = sorted(retVals, key=np.abs)
    root0 = sortedRet[0]
    return np.real(root0), np.imag(root0)


def computeLargeLambdaUpper(inData):
    """

      :param inData: [n, lambda, Fest]
      :return: [n, lambda, re(F), im(F)]
      """
    n, lmd, FEst = inData
    FVecTmp = sopt.fsolve(eqnLargeLambdaUpperSymPair, [np.real(FEst), np.imag(FEst)], args=(n, lmd), maxfev=100,
                          xtol=1e-6)
    return [n, lmd, FVecTmp[0], FVecTmp[1]]
