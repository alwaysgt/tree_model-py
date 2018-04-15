# Structure

This package is used for implementing regression tree and classification tree. 



## Regression Tree

It reads in a shape\[n_sample\]\[n_features\] matrix X, and a shape\[n_sample\] response vector Y. Then it generate a regression tree according to least square error principal. 

## Classification Tree

Still on the way......

# Usage 

First put the file `tree_model.py` in the directory in which you run your python programme. 

The way to use it is similar to that in *sklearn*, here's an example

```python
from tree_model import regression_tree
rt = regression_tree(max_depth = 2) # the default value is 2
rt.fit(X,Y)
rt.predict(X)
```

We also provide a class function `rt.print()` to visualize the tree it generates.

