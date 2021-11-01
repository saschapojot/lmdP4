import numpy as np
import matplotlib.pyplot as plt

lmd=40

th=0.8
F=4*np.exp(1j*th*np.pi)
###########compute real roots
coefEqn=[-1j,0,0,lmd,0,-F]
trueRootsAll=np.roots(coefEqn)


fig,ax=plt.subplots(figsize=(20,20))
#######################true roots, data serialization
trueRootsReal=[]
trueRootsImag=[]
for rtTmp in trueRootsAll:
    trueRootsReal.append(np.real(rtTmp))
    trueRootsImag.append(np.imag(rtTmp))


####################asymptotic roots when large lambda>>|F|^(3/5)
# lmdLargeRoots=[]
y3n0s=[np.exp(1j*(4*n*np.pi-np.pi)/6)*lmd**(1/3) for n in range(0,3)]

y3ns=[ytmp+F*1/(2*lmd*ytmp-5*1j*ytmp**4) for ytmp in y3n0s]
z2ns=[(F/lmd)**(1/2),-(F/lmd)**(1/2)]
lmdLargeRoots=y3ns+z2ns
#data serialization
lmdLargeRootsReal=[]
lmdLargeRootsImag=[]
for rtTmp in lmdLargeRoots:
    lmdLargeRootsReal.append(np.real(rtTmp))
    lmdLargeRootsImag.append(np.imag(rtTmp))


############################asymptotic roots for small lambda<<|F|^(3/5)
# y5ns=[]
y5n0s=[np.exp(1j*(np.pi+4*n*np.pi)/10)*F**(1/5) for n in range(0,5)]
y5ns=[ytmp-1j/(5*ytmp**2)*lmd for ytmp in y5n0s]

#data serialization
smallLmdRootsReal=[]
smallLmdRootsImag=[]
for rtTmp in y5ns:
    smallLmdRootsReal.append(np.real(rtTmp))
    smallLmdRootsImag.append(np.imag(rtTmp))

#################plot Stokes wedges
N=100
R=35


rtsAll=[]
rtsAll.extend(trueRootsAll)
rtsAll.extend(y5ns)
rtsAll.extend(lmdLargeRoots)
rs=np.linspace(start=0,stop=np.max(np.abs(rtsAll))*1.1,num=N)


a1=3/14*np.pi

a2=-1/14*np.pi

a3=-5/14*np.pi

a4=-9/14*np.pi

a5=-13/14*np.pi

a6=-17/14*np.pi

p0x=[0]*len(rs)
p0y=rs

p1x=[elem*np.sign(np.cos(a1)) for elem in rs]
p1y=[xtmp*np.tan(a1) for xtmp in p1x]

p2x=[elem*np.sign(np.cos(a2)) for elem in rs]
p2y=[xtmp*np.tan(a2) for xtmp in p2x]

p3x=[elem*np.sign(np.cos(a3)) for elem in rs]
p3y=[xtmp*np.tan(a3) for xtmp in p3x]

p4x=[elem*np.sign(np.cos(a4)) for elem in rs]
p4y=[xtmp*np.tan(a4) for xtmp in p4x]

p5x=[elem*np.sign(np.cos(a5)) for elem in rs]
p5y=[xtmp*np.tan(a5) for xtmp in p5x]

p6x=[elem*np.sign(np.cos(a6)) for elem in rs]
p6y=[xtmp*np.tan(a6) for xtmp in p6x]


b0=1/4*np.pi

b1=-1/4*np.pi

b2=-3/4*np.pi

b3=-5/4*np.pi

q0x=[elem*np.sign(np.cos(b0)) for elem in rs]
q0y=[xtmp*np.tan(b0) for xtmp in q0x]

q1x=[elem*np.sign(np.cos(b1)) for elem in rs]
q1y=[xtmp*np.tan(b1) for xtmp in q1x]

q2x=[elem*np.sign(np.cos(b2)) for elem in rs]
q2y=[xtmp*np.tan(b2) for xtmp in q2x]

q3x=[elem*np.sign(np.cos(b3)) for elem in rs]
q3y=[xtmp*np.tan(b3) for xtmp in q3x]






ax.spines['bottom'].set_color('grey')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_color('grey')
ax.spines['left'].set_position('center')
# ax.set_yticks([])
# ax.set_xticks([])
ax.plot(p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y,p4x,p4y,p5x,p5y,p6x,p6y,color="blue")
ax.plot(q0x,q0y,q1x,q1y,q2x,q2y,q3x,q3y,color="red")
#right I-I
ax.fill_between(p1x, p1y, p2y,color='gainsboro')
#left I-I
ax.fill_between(p6x,p6y,p5y,color="gainsboro")
#right I-II
ax.fill_between(p2x,p2y,q1y,color="aquamarine")
#left I-II
ax.fill_between(p5x,p5y,q2y,color="aquamarine")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#color filling
p=int(N/2)
#region right I-I

tx=p1x[p]
ty=1/3*(2*p1y[p]+p2y[p])
ax.text(tx,ty,"I-I",fontsize=16)
#region left I-I
tx=p6x[p]
ty=1/3*(2*p6y[p]+p5y[p])
ax.text(tx,ty,"I-I",fontsize=16)

#region right I-II
tx=p2x[p]
ty=1/3*(2*p2y[p]+q1y[p])
ax.text(tx,ty,"I-II",fontsize=16)
#region left I-II
tx=p5x[p]
ty=1/3*(2*p5y[p]+q2y[p])
ax.text(tx,ty,"I-II",fontsize=16)
##############################plot true roots

trueSct=ax.scatter(trueRootsReal,trueRootsImag,color="black",s=40,label="exact roots")
###############################plot asymp large lambda

lmdLargeSct=ax.scatter(lmdLargeRootsReal,lmdLargeRootsImag,color="red",marker="x",s=20,label="large $\lambda$")
###############################plot asymp small lambda
lmdSmallSct=ax.scatter(smallLmdRootsReal,smallLmdRootsImag,color="blue",marker="s",label="small $\lambda$")

#######################save fig
plt.legend()
plt.title("$\lambda=$"+str(lmd)+", $F=$"+str(F))
plt.savefig("tmp.png")
plt.close()