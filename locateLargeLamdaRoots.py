import numpy as np

lmd=20

th=0.8
F=40*np.exp(1j*th*np.pi)
###########compute real roots
coefEqn=[-1j,0,0,lmd,0,-F]
trueRootsAll=np.roots(coefEqn)


###############asymp roots when large lambda >>|F|^(3/5)
y3n0s=[np.exp(1j*(4*n*np.pi-np.pi)/6)*lmd**(1/3) for n in range(0,3)]

y3ns=[ytmp+F*1/(2*lmd*ytmp-5*1j*ytmp**4) for ytmp in y3n0s]


def selectTrueRoot(exactRoots,asympRoot):
    """

    :param exactRoots: roots solved by np.roots
    :param asympRoot: asymtotic root
    :return:
    """
    diffAbs=[np.abs(elem-asympRoot) for elem in exactRoots]
    inds=np.argsort(diffAbs)
    return exactRoots[inds[0]]


for elem in y3ns:
    print("asymp="+str(elem))

    print("exact="+str(selectTrueRoot(trueRootsAll,elem)))