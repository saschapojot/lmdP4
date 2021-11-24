from shootingR12Funcs import *
import pandas as pd



# lmd0=3
# FEst0=1
# inData0=[lmd0,FEst0]
#
# F0Val=computeOneSolution(inData0)
# print(F0Val)

####values of g and E
num=3000
startG=1e-4
stopG=1e0

gnIndAll = np.linspace(start=np.log10(startG), stop=np.log10(stopG), num=num)
gAll = [10 ** elem for elem in gnIndAll]
EMax=60
#convert to lambda and F
inDataAll=[] #contains [lambda, FEst]
for g in gAll:
    for E in range(1,EMax):
        lmd=g**(-4/7)
        FEst=E*g**(-2/7)
        inDataAll.append([lmd,FEst])


threadNum=24

pool1=Pool(threadNum)
tShootingStart=datetime.now()
retAll=pool1.map(computeOneSolution,inDataAll)
tShootingEnd=datetime.now()

print("shooting time: ",tShootingEnd-tShootingStart)

#data serialization
gShootingVals=[]
EShootingVals=[]
for itemTmp in retAll:
    lmd,F=itemTmp
    gTmp=lmd**(-7/4)
    ETmp=F*lmd**(-1/2)
    gShootingVals.append(gTmp)
    EShootingVals.append(ETmp)



# plot shooting
fig, ax = plt.subplots(figsize=(20, 20))

# ax.set_xscale('log')
# ax.set_yscale('symlog')
ax.set_xscale("log")
ax.set_ylabel("E")
ax.set_xlabel("g")
ax.set_title("Shooting eigenvalues for potential $V(x)=x^{2}-igx^{5}$, region I-II")

shootingScatter=ax.scatter(gShootingVals,EShootingVals,color="blue",marker=".",s=50,label="shooting")
plt.legend()

plt.savefig("shootingR12.png")
plt.close()

dataPdFrame=np.array([gShootingVals,EShootingVals]).T
dfgE=pd.DataFrame(dataPdFrame,columns=["g","E"])

dfgE.to_csv("shootingR12.csv")