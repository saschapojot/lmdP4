import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
import scipy.special as sspecial
import mpmath
from mpmath import mp
mp.dps=100
import pandas as pd
#this script calculates real F, with large lambda value, use WKB lower pair
#the asymptotic relation uses lambda>>|F|^{3/5}

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

    :param FIn: trial eigenvalue, real
    :param data: (n,lmd)
    :return:
    """
    n,lmd=data
    F=FIn[0]
    coefEqn = [-1j, 0, 0, lmd, 0, -F]
    trueRootsAll = np.roots(coefEqn)
    ###############asymp roots when large lambda >>|F|^(3/5)
    y3n0s = [np.exp(1j * (4 * n * np.pi - np.pi) / 6) * lmd ** (1 / 3) for n in range(0, 3)]

    y3ns = [ytmp + F * 1 / (2 * lmd * ytmp - 5 * 1j * ytmp ** 4) for ytmp in y3n0s]

    selectedRoots = [selectTrueRoot(trueRootsAll, elem) for elem in y3ns]

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

    return np.abs(root0)


def computeOneSolWithLargeLambdaLower(inData):
    """

    :param inData: [n, lambda, Fest]
    :return: [n, lambda, F]
    """
    n, lmd, FEst = inData
    F=sopt.fsolve(eqnLargeLambdaLowerSymPair,FEst,args=(n,lmd),maxfev=100,xtol=1e-6)[0]
    return [n,lmd,F]


def eqnSmallLambdaLowerSymPair(FIn, *data):
    """

    :param FIn: trial eigenvalue, real
    :param data: (n, lmd)
    :return:
    """
    n,lmd=data
    F=FIn[0]
    coefEqn = [-1j, 0, 0, lmd, 0, -F]
    trueRootsAll = np.roots(coefEqn)
    ###############asymp roots when large lambda <<|F|^(3/5)
    y0ns=[np.exp(-1j*7/10*np.pi)*F**(1/5),np.exp(-1j*3/10*np.pi)*F**(1/5)]
    yns=[elem-1j/(5*elem**2)*lmd for elem in y0ns]
    selectedRoots = [selectTrueRoot(trueRootsAll, elem) for elem in yns]

    sortedRootsByAngle = sorted(selectedRoots, key=np.angle)
    # taking lower pair
    x2Tmp = sortedRootsByAngle[0]
    x1Tmp = sortedRootsByAngle[1]
    if np.abs(np.real(x1Tmp-x2Tmp))<1e-10:
        print(x1Tmp)
        print(x2Tmp)
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
    return np.abs(root0)


def computeOneSolWithSmallLambdaLower(inData):
    """

    :param inData: [n, lambda, Fest]
    :return: [n, lambda, F]
    """
    n, lmd, FEst = inData
    F=sopt.fsolve(eqnSmallLambdaLowerSymPair,FEst,args=(n,lmd),maxfev=100,xtol=1e-6)[0]
    return [n,lmd,F]