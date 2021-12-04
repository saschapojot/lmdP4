from realLowerFuncs1 import *


num=30
startG=1e-3
stopG=1e-1
gnIndAll = np.linspace(start=np.log10(startG), stop=np.log10(stopG), num=num)
gAll = [10 ** elem for elem in gnIndAll]

lmdAll=[gTmp**(-4/7) for gTmp in gAll]
threadNum = 24
# energyLevelMax = 4
levelStart=0
levelEnd=15
levelsAll = range(levelStart, levelEnd + 1)
inDataAll=[]

for nTmp in levelsAll:
    for lmdTmp in lmdAll:
        FEst=(nTmp+1/2)*np.pi
        inDataAll.append([nTmp,lmdTmp,FEst])


#####################parallel computation part for large lambda, lower pair, may be memory consuming
tLmdStart=datetime.now()
pool1=Pool(threadNum)
retAllLargeLmd=pool1.map(computeOneSolWithLargeLambdaLower,inDataAll)
tLmdEnd=datetime.now()
print("parallel WKB time for large lambda lower: ",tLmdEnd-tLmdStart)
######################parallel computation part for small lambda, lower pair, may be memory consuming
# tSmallStart=datetime.now()
# pool2=Pool(threadNum)
# retAllSmallLmd=pool2.map(computeOneSolWithSmallLambdaLower,inDataAll)
# tSmallEnd=datetime.now()
# print("parallel WKB time for small lambda lower: ",tSmallEnd-tSmallEnd)
############data serialization for large lmd lower, plot large lmd
nSctLargeLmd=[]
lmdSctLargeLmd=[]
FRealLargeLmd=[]
#####transform back to E and g, large lambda
ERealLargeLmd=[]
gSctLargeLmd=[]
############data serialization for small lmd lower, plot small lmd
nSctSmallLmd=[]
lmdSctSmallLmd=[]
FRealSmallLmd=[]
#####transform back to E and g, small lambda
ERealSmallLmd=[]
gSctSmallLmd=[]
###################
for itemTmp in retAllLargeLmd:
    nTmp,lmdTmp,FRe=itemTmp
    nSctLargeLmd.append(nTmp)
    lmdSctLargeLmd.append(lmdTmp)
    FRealLargeLmd.append(FRe)

for j in range(0,len(lmdSctLargeLmd)):
    lmdTmp=lmdSctLargeLmd[j]
    FReTmp=FRealLargeLmd[j]
    gTmp=lmdTmp**(-7/4)
    EReTmp=FReTmp*lmdTmp**(-1/2)
    gSctLargeLmd.append(gTmp)
    ERealLargeLmd.append(EReTmp)
##########
# for itemTmp in retAllSmallLmd:
#     nTmp,lmdTmp,FRe=itemTmp
#     nSctSmallLmd.append(nTmp)
#     lmdSctSmallLmd.append(lmdTmp)
#     FRealSmallLmd.append(FRe)
# for j in range(0,len(lmdSctSmallLmd)):
#     lmdTmp=lmdSctSmallLmd[j]
#     FReTmp=FRealSmallLmd[j]
#     gTmp=lmdTmp**(-7/4)
#     EReTmp=FReTmp*lmdTmp**(-1/2)
#     gSctSmallLmd.append(gTmp)
#     ERealSmallLmd.append(EReTmp)

#load shooting data
shootingDf=pd.read_csv("shootingR12.csv")
shootingDf=shootingDf.drop(shootingDf.columns[0],axis=1)
#########################

gShooting=shootingDf["g"]
EShooting=shootingDf["E"]

####################plotting

tPltStart = datetime.now()

# # plot WKB
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_ylabel("E")
# plt.yscale('symlog')
ax.set_xscale("log")
ax.set_xlabel("g")
ax.set_title("Eigenvalues for potential $V(x)=\lambda x^{2}-ix^{5}$")


lowerLargeLmdSct=ax.scatter(gSctLargeLmd,ERealLargeLmd,color="green",marker="x",label="WKB large $\lambda$ lower")
lowerSmallLmdSct=ax.scatter(gSctSmallLmd,ERealSmallLmd,color="red",marker="+",label="WKB small $\lambda$ lower")
shootingSct=ax.scatter(gShooting,EShooting,color="blue",marker=".",label="Shooting")



plt.legend()



plt.savefig("tmp00.png")