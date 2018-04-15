from tree_model import regression_tree
import numpy as np
X = np.array([[2,1,-1,0]]).transpose()
Y = np.array([ 0.25664347,-1.67945097,-0.75309126,-0.50370735])
rt = regression_tree()
rt.fit(X,Y)
X = np.array([[1,0,-1,100]]).transpose()
print(rt.predict(X))
rt.print()
print("\n\n\n")
print(rt.predict(X))