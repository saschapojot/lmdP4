import matplotlib.pyplot as plt
import pandas as pd



#load shooting data
prefix="startL4"
shootingDf=pd.read_csv(prefix+"shootingR12.csv",index_col=0)

gShooting=shootingDf["g"]
EShooting=shootingDf["E"]
########################


#load adj data

adjDf=pd.read_csv("adjGRealR12.csv",index_col=0)

gAdj=adjDf["g"]
EAdj=adjDf["E"]

#load sep data
sepDf=pd.read_csv("sepGRealR12.csv",index_col=0)
gSep=sepDf["g"]
ESep=sepDf["E"]

# # plot
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_ylabel("E")
# plt.yscale('symlog')
ax.set_xscale("log")
ax.set_xlabel("g")
ax.set_title("Eigenvalues for potential $V(x)=\lambda x^{2}-ix^{5}$")


sctShooting=ax.scatter(gShooting,EShooting,color="blue",marker=".",s=40,label="Shooting")
sctAdj=ax.scatter(gAdj,EAdj,color="fuchsia",marker="+",s=50,label="WKB adj real")
sctSep=ax.scatter(gSep,ESep,color="green",marker="x",s=50,label="WKB sep real")

plt.legend()
plt.savefig("tmp12.png")