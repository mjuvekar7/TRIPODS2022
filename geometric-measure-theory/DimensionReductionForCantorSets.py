"""
@ZSheng & Geometric Group

We are looking for a way to generate a 2 dimension cantor set and try to reduce its dimension to a lower one. As it is proved 'doable'
in the paper Fractal dimension, approximation and data sets. The function for x in A is.... the function for x in set B is... some one please
helps me to put the function in...

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import itertools
from itertools import product
x=[]
y=[]
nvalue=4    #corresponds to the value of n


def xinA(list,n,k,power):  # for the case x in A
    for an in [0, 1]:
        k = k + an*(4**(-power))

        if power < n:
            xinA(x, n, k, power + 1)

        else :
            x.append(k)


def xinB(list,n,k,power):  # for the case x in B
    for an in [0, 2]:
        k = k + an * (4 ** (-power))

        if power < n:
            xinB(y, n, k, power + 1)

        else:
            y.append(k)

xinA(x,nvalue,0,1)

xinB(y,nvalue,0,1)

#print(y)

#if u wanna reduce such sets into a lower dim, you may simply add xinA and xinB up.

DimReducedset = x+y

cartesian = list(product(x, y))  # cartesian product to make the set becomes 2D

print(cartesian)

listoflist = np.array(cartesian)

output='2D Cantor set '+str(nvalue)
dataset = pd.DataFrame({'col1': listoflist[:, 0], 'col2': listoflist[:, 1]})
#print(dataset)
fig = plt.figure()
plt.get_current_fig_manager().set_window_title(output)
ax = fig.add_subplot(1, 1, 1)
xcomponent = listoflist[:,0]
ycomponent = listoflist[:,1]
ax.scatter(xcomponent, ycomponent,s=1)
plt.ylabel('y')
plt.xlabel('x')
plt.show()




