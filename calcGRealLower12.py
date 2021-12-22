from gRealR12Funcs import *


num=15
startG=1e-4
stopG=1e1
gnIndAll = np.linspace(start=np.log10(startG), stop=np.log10(stopG), num=num)
gAll = [10 ** elem for elem in gnIndAll]


threadNum = 24
# energyLevelMax = 4
levelStart=2
levelEnd=levelStart
levelsAll = range(levelStart, levelEnd + 1)
inDataAll=[]

for nTmp in levelsAll:
    for gTmp in gAll:
        EEst=(nTmp+1/2)*np.pi
        inDataAll.append([nTmp,gTmp,EEst])

############################################################################
######Adj computation
# ###########parallel computation  for adj, may be memory consuming
tWKBParalStart = datetime.now()
pool1 = Pool(threadNum)
retAllAdj=pool1.map(computeOneSolutionWith5AdjacentPairs,inDataAll)
tWKBParalEnd = datetime.now()
print("parallel WKB time for adj pairs: ", tWKBParalEnd - tWKBParalStart)
#################end of parallel computation
###############serial computation for adj, may be time consuming
# tWKBSerialStart=datetime.now()
# retAllAdj=[]
# for itmTmp in inDataAll:
#     n,g,E=computeOneSolutionWith5AdjacentPairs(itmTmp)
#     retAllAdj.append([n,g,E])
# tWKBSerialEnd=datetime.now()
# print("Serial WKB for adj pairs: ",tWKBSerialEnd-tWKBSerialStart)
#################end of serial computation
####################################end of adj computation
tPltStart = datetime.now()

# # # plot WKB
# fig, ax = plt.subplots(figsize=(20, 20))
# ax.set_ylabel("E")
# ax.set_xlabel("g")
# ax.set_xscale("log")
# ax.set_title("Eigenvalues for potential $V(x)=x^{2}-igx^{5}$")

# data serialization for adj
nSctValsAdj = []
gSctValsAdj= []
ERealSctValsAdj = []

#data serialization for adj
for itemTmp in retAllAdj:
    nTmp, gTmp, ERe = itemTmp
    if ERe<0 or ERe>40:
        continue
    nSctValsAdj.append(nTmp)
    gSctValsAdj.append(gTmp)
    ERealSctValsAdj.append(ERe)
###########################
#write data of adj to csv
adjDfData=np.array([gSctValsAdj,ERealSctValsAdj]).T
adjDf=pd.DataFrame(adjDfData,columns=["g","E"])
adjDf.to_csv("level"+str(levelStart)+"adjGRealR12.csv")

#load shooting data
# prefix="startL4"
# shootingDf=pd.read_csv(prefix+"shootingR12.csv")
# shootingDf=shootingDf.drop(shootingDf.columns[0],axis=1)
#########################

# gShooting=shootingDf["g"]
# EShooting=shootingDf["E"]
#
# shootingSct=ax.scatter(gShooting,EShooting,color="blue",marker="1",label="Shooting")
# sctRealPartWKBAdj = ax.scatter(gSctValsAdj, ERealSctValsAdj, color="fuchsia", marker="+", s=50, label="WKB real part adj")
# plt.legend()
# tPltEnd = datetime.now()
# print("plotting time: ", tPltEnd - tPltStart)
#
#
#
#
# plt.savefig("tmp12.png")