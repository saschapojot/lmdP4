from funcsParal import *


num=50

startLambda=0

stopLambda=20

lmdAll=np.linspace(start=startLambda,stop=stopLambda,num=num)


threadNum = 24
# energyLevelMax = 4
levelStart=0
levelEnd=3
levelsAll = range(levelStart, levelEnd + 1)
inDataAll=[]

for nTmp in levelsAll:
    for lmdTmp in lmdAll:
        FEst=(nTmp+1/2)*np.pi
        inDataAll.append([nTmp,lmdTmp,FEst])



##########################parallel computation part for Adj, may be memory consuming
tParalStart=datetime.now()
pool1=Pool(threadNum)
retAllAdj=pool1.map(computeOneSolutionWith5AdjPairs,inDataAll)
tParalEnd=datetime.now()
print("parallel WKB time for adj pairs: ",tParalEnd-tParalStart)

tPltStart = datetime.now()

# # plot WKB
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_ylabel("E")
ax.set_xlabel("g")
ax.set_title("Eigenvalues for potential $V(x)=\lambda x^{2}-ix^{5}$")
#data serialization for Adj
nAdjSctVals=[]
lmdAdjSctVals=[]
ERealAdjSctVals=[]
EImagAdjSctVals=[]
for itemTmp in retAllAdj:
    nTmp,lmdTmp,ERe,EIm=itemTmp
    nAdjSctVals.append(nTmp)
    lmdAdjSctVals.append(lmdTmp)
    ERealAdjSctVals.append(ERe)
    EImagAdjSctVals.append(EIm)

adjWKBRealPartSct=ax.scatter(lmdAdjSctVals,ERealAdjSctVals,color="red",marker=".",label="WKB real part adj")
plt.legend("tmp120.png")