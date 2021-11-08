from funcsParal import *


num=1000
startG=1e-3
stopG=1e-0
gnIndAll = np.linspace(start=np.log10(startG), stop=np.log10(stopG), num=num)
gAll = [10 ** elem for elem in gnIndAll]

lmdAll=[gTmp**(-4/7) for gTmp in gAll]


threadNum = 24
# energyLevelMax = 4
levelStart=0
levelEnd=7
levelsAll = range(levelStart, levelEnd + 1)
inDataAll=[]

for nTmp in levelsAll:
    for lmdTmp in lmdAll:
        FEst=(nTmp+1/2)*np.pi
        inDataAll.append([nTmp,lmdTmp,FEst])



##########################parallel computation part for Adj, may be memory consuming
# tParalStart=datetime.now()
# pool1=Pool(threadNum)
# retAllAdj=pool1.map(computeOneSolutionWith5AdjPairs,inDataAll)
# tParalEnd=datetime.now()
# print("parallel WKB time for adj pairs: ",tParalEnd-tParalStart)
# ############################end
#####################parallel computation part for large lambda, lower pair, may be memory consuming
tLmdStart=datetime.now()
pool1=Pool(threadNum)
retAllLargeLmd=pool1.map(computeOneSolWithLargeLambdaLower,inDataAll)
tLmdEnd=datetime.now()
print("parallel WKB time for large $\lambda$ lower: ",tLmdEnd-tLmdStart)
#####################end

##################parallel computation part for large lambda, upper pair, may be memory consuming
tUpperStart=datetime.now()
pool2=Pool(threadNum)
retAllUpperlargeLmd=pool2.map(computeLargeLambdaUpper,inDataAll)
tUpperEnd=datetime.now()
print("parallel WKB time for large $\lambda$ upper: ",tUpperEnd-tUpperStart)



#####################end
tPltStart = datetime.now()

# # plot WKB
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_ylabel("E")
# plt.yscale('symlog')
ax.set_xscale("log")
ax.set_xlabel("g")
ax.set_title("Eigenvalues for potential $V(x)=\lambda x^{2}-ix^{5}$")
#data serialization for Adj, scatter for Adj
# nAdjSctVals=[]
# lmdAdjSctVals=[]
# ERealAdjSctVals=[]
# EImagAdjSctVals=[]
# for itemTmp in retAllAdj:
#     nTmp,lmdTmp,ERe,EIm=itemTmp
#     nAdjSctVals.append(nTmp)
#     lmdAdjSctVals.append(lmdTmp)
#     ERealAdjSctVals.append(ERe)
#     EImagAdjSctVals.append(EIm)
#
# adjWKBRealPartSct=ax.scatter(lmdAdjSctVals,ERealAdjSctVals,color="red",marker=".",label="WKB real part adj")
############data serialization for large lmd lower, plot large lmd
nSctLargeLmd=[]
lmdSctLargeLmd=[]
FRealLargeLmd=[]
FImagLargeLmd=[]
#####transform back to E and g
ERealLargeLmd=[]
gSctLargeLmd=[]
for itemTmp in retAllLargeLmd:
    nTmp,lmdTmp,FRe,FIm=itemTmp
    if np.abs(FRe)>30:
        continue
    # if np.abs(FIm)>1:
    #     continue
    nSctLargeLmd.append(nTmp)
    lmdSctLargeLmd.append(lmdTmp)
    FRealLargeLmd.append(FRe)
    FImagLargeLmd.append(FIm)

for j in range(0,len(lmdSctLargeLmd)):
    lmdTmp=lmdSctLargeLmd[j]
    FReTmp=FRealLargeLmd[j]
    gTmp=lmdTmp**(-7/4)
    EReTmp=FReTmp*gTmp**(2/7)
    gSctLargeLmd.append(gTmp)
    ERealLargeLmd.append(EReTmp)

largeLmdSct=ax.scatter(gSctLargeLmd,ERealLargeLmd,color="blue",marker=".",label="WKB large $\lambda$ lower")
##########data serialization for large lambda, upper
nUppSctLargeLmd=[]
lmdUpperSctLargeLmd=[]
FRealUpperSctLargeLmd=[]
FImagUpperSctLargeLmd=[]
######transform to E and g
ERealUpperLargeLmd=[]
gUpperSctLargeLmd=[]

for itemTmp in retAllUpperlargeLmd:
    nTmp,lmdTmp,FReTmp,FImTmp=itemTmp
    nUppSctLargeLmd.append(nTmp)
    lmdUpperSctLargeLmd.append(lmdTmp)
    FRealUpperSctLargeLmd.append(FReTmp)
    FImagUpperSctLargeLmd.append(FImTmp)

for j in  range(0,len(lmdUpperSctLargeLmd)):
    lmdTmp=lmdUpperSctLargeLmd[j]
    FReTmp=FRealUpperSctLargeLmd[j]
    gTmp=lmdTmp**(-7/4)
    EReTmp=FReTmp*gTmp**(2/7)
    gUpperSctLargeLmd.append(gTmp)
    ERealUpperLargeLmd.append(EReTmp)

upperLargeLmdSct=ax.scatter(gUpperSctLargeLmd,ERealUpperLargeLmd,color="green",marker=".",label="WKB large $\lambda$ upper")








##############
plt.legend()
plt.savefig("tmp120.png")
plt.close()