import numpy as np

def ColumnRowSection(foreground):
    dims = np.shape(foreground)
    X = dims[0]
    Y = dims[1]
    labeled = np.zeros((X, Y))
    labels = []
    label = 1

    # Column scan to find vertical boundary
    for c in range(Y):
        space = 0
        for r in range(X):
            if foreground[c][r] == 0:
                space = space + 1
                continue
            if space > 100:
                labeled = 1
                labels.append(labeled)
            space = 0
            
    return labels, labeled