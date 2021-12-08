from gRealR12Funcs import *


num=30
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



#############parallel computation for sep, may be memory consuming
tParalSepStart=datetime.now()
pool2=Pool(threadNum)
retAllSep=pool2.map(computeOneSolutionWith5SepPairs,inDataAll)
tParalSepEnd=datetime.now()
print("parallel WKB time for sep pairs: ",tParalSepEnd-tParalSepStart)

#################################end of parallel computation


#data serialization for sep
nSctValsSep=[]
gSctValsSep=[]
ERealSctValsSep=[]

#data serialization for sep
for itemTmp in retAllSep:
    nTmp,gTmp,ERe,EIm=itemTmp
    if ERe<0 or ERe>40:
        continue
    nSctValsSep.append(nTmp)
    gSctValsSep.append(gTmp)
    ERealSctValsSep.append(ERe)
###########
#######write data of sep to csv
sepDfData=np.array([gSctValsSep,ERealSctValsSep]).T
sepDf=pd.DataFrame(sepDfData,columns=["g","E"])
sepDf.to_csv("sepGRealR12.csv")

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