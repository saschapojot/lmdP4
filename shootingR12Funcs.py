import numpy as np
import scipy.optimize as sopt

import matplotlib.pyplot as plt
from multiprocessing import Pool
from datetime import datetime
import math
import warnings
warnings.simplefilter("error")
eps=1e-8

def Q(y, lmd, F):
    return lmd * y ** 2 - 1j * y ** 5 - F


def f1(y, lmd, F):

    return -1 / 4 * (2 * lmd * y - 5 * 1j * y ** 4) / Q(y, lmd, F) + Q(y, lmd, F) ** (1 / 2)


def oneStepRk4(lmd, F, h, yCurr, zCurr, vCurr):
    """

    :param lmd:
    :param F:
    :param h:
    :param yCurr:
    :param zCurr:
    :param vCurr:
    :return: zNext, vNext
    """
    M0 = h * Q(yCurr, lmd, F) * zCurr
    M1 = h * Q(yCurr + 1 / 2 * h, lmd, F) * (zCurr + 1 / 2 * h * vCurr)
    M2 = h * Q(yCurr + 1 / 2 * h, lmd, F) * (zCurr + 1 / 2 * h * vCurr + 1 / 4 * h * M0)
    M3 = h * Q(yCurr + h, lmd, F) * (zCurr + h * vCurr + 1 / 2 * h * M1)

    zNext = zCurr + h * vCurr + 1 / 6 * h * (M0 + M1 + M2)
    vNext = vCurr + 1 / 6 * (M0 + 2 * M1 + 2 * M2 + M3)

    return zNext, vNext


def selectStartingPoint(y0,lmd,F):
    if np.abs(Q(y0,lmd,F))<eps:
        return 2/3*y0
    else:
        return y0

LEst=4
def calcBoundaryValue(F,*data):
    F=F[0]
    lmd=data[0]


    theta=(- 1/ 4*(1/2)-1/14*(1/2)) * np.pi
    y0=LEst*np.exp(1j*theta)


    while np.abs(Q(y0,lmd,F))<eps:
        y0=selectStartingPoint(y0,lmd,F)

    hAbsEst=1e-4
    Ls=np.abs(y0)
    Ns=int(Ls/hAbsEst)
    h=-Ls/Ns*np.exp(1j*theta)
    yAll=[y0]
    zAll=[1]
    vAll=[f1(y0,lmd,F)]


    for j in range(0,Ns):
        yCurr=yAll[-1]
        zCurr=zAll[-1]
        vCurr=vAll[-1]
        zNext,vNext=oneStepRk4(lmd,F,h,yCurr,zCurr,vCurr)

        yNext=yCurr+h
        yAll.append(yNext)
        zAll.append(zNext)
        vAll.append(vNext)
    #boundary condition: Re(vN/zN)=0

    return np.real(vAll[-1]/zAll[-1])


def computeOneSolution(inData):
    """

    :param inData: [lmd, FEst]
    :return:
    """
    lmd,FEst=inData
    # try:
    #     FVal=sopt.fsolve(calcBoundaryValue,FEst,args=(lmd),maxfev=100,xtol=1e-6)[0]
    # except RuntimeWarning:
    #     print("warning catched")
    #     return [lmd,-100]
    FVal,infoDict,ier,msg = sopt.fsolve(calcBoundaryValue, FEst, args=(lmd), maxfev=100, xtol=1e-8,full_output=True)
    if ier==1:
        return [lmd,FVal[0]]
    else:
        return [lmd,-100]
