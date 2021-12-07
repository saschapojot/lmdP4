from gRealR12Funcs import *


num=60
startG=1e-4
stopG=1e1
gnIndAll = np.linspace(start=np.log10(startG), stop=np.log10(stopG), num=num)
gAll = [10 ** elem for elem in gnIndAll]


threadNum = 24
# energyLevelMax = 4
levelStart=0
levelEnd=3
levelsAll = range(levelStart, levelEnd + 1)
inDataAll=[]

for nTmp in levelsAll:
    for gTmp in gAll:
        EEst=(nTmp+1/2)*np.pi
        inDataAll.append([nTmp,gTmp,EEst])


# ###########parallel computation part for adj, may be memory consuming
tWKBParalStart = datetime.now()
pool1 = Pool(threadNum)
retAllAdj=pool1.map(computeOneSolutionWith5AdjacentPairs,inDataAll)
tWKBParalEnd = datetime.now()
print("parallel WKB time for adj pairs: ", tWKBParalEnd - tWKBParalStart)


tPltStart = datetime.now()

# # plot WKB
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_ylabel("E")
ax.set_xlabel("g")
ax.set_title("Eigenvalues for potential $V(x)=x^{2}-igx^{5}$")

# data serialization for adj
nSctValsAdj = []
gSctValsAdj= []
ERealSctValsAdj = []


#data serialization for adj
for itemTmp in retAllAdj:
    nTmp, gTmp, ERe = itemTmp
    nSctValsAdj.append(nTmp)
    gSctValsAdj.append(gTmp)
    ERealSctValsAdj.append(ERe)

sctRealPartWKBAdj = ax.scatter(gSctValsAdj, ERealSctValsAdj, color="green", marker=".", s=50, label="WKB real part adj")
plt.legend()
tPltEnd = datetime.now()
print("plotting time: ", tPltEnd - tPltStart)




plt.savefig("tmp11.png")