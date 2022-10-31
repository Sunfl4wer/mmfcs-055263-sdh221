from multiprocessing.connection import wait
import numpy as np
import unionfind as uf

def getFilter(n):
    filter = []
    for i in range(-n, 1):
        for j in range(-n, 1):
            if i == j and i == 0:
                continue
            filter.append([i,j])
    return filter

def FindNeighbors(x, y):
    filters = getFilter(15)
    neighbors = []
    for f in filters:
        nx = x+f[0]
        ny = y+f[1]
        if nx < 0 or ny < 0:
            continue
        neighbors.append([nx, ny])
    return neighbors

def TwoPass(foreground):
    dims = np.shape(foreground)
    X = dims[0]
    Y = dims[1]
    labeled = np.zeros((X, Y))
    labels = []
    label = 0

    labelDJS = uf.UnionFind()
    # First pass to labels pixel based on north, north-west and west neighbors
    for r in range(X):
        for c in range(Y):
            if not foreground[r][c]:
                continue

            neighbors = FindNeighbors(r, c)
            foregroundNeighbors = []
            for n in neighbors:
                if foreground[n[0]][n[1]]:
                    foregroundNeighbors.append(n)

            if len(foregroundNeighbors) == 0:
                label+=1
                labeled[r][c] = label
                labelDJS.add(label)
                continue

            n = foregroundNeighbors[0]
            minLabel = labeled[n[0]][n[1]]
            for i in range(1, np.shape(foregroundNeighbors)[0]):
                nn = foregroundNeighbors[i]
                if labeled[n[0]][n[1]] != labeled[nn[0]][nn[1]]:
                    labelDJS.union(labeled[n[0]][n[1]], labeled[nn[0]][nn[1]])
                minLabel = min(minLabel, labeled[nn[0]][nn[1]])
            labeled[r][c] = minLabel

    # Second pass to relabeled pixel based on the minimum value in the set
    for r in range(X):
        for c in range(Y):
            if not foreground[r][c]:
                continue
            
            rootLabel = labelDJS.find(labeled[r][c])
            labeled[r][c] = rootLabel
            labels.append(rootLabel)
    
    labels = np.sort(np.unique(labels))

    return labels, labeled