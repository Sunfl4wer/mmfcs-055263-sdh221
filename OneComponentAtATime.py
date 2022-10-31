import numpy as np

def getFilter(n):
    filter = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            if i == j and i == 0:
                continue
            filter.append([i,j])
    return filter

def FindNeighbors(x, y, X, Y):
    filters = getFilter(15)
    neighbors = []
    for f in filters:
        p = [x+f[0], y+f[1]]
        if p[0] >= 0 and p[0] < X and p[1] >= 0 and p[1] < Y:
            neighbors.append(p)
    return neighbors

def OneComponentAtATime(foreground):
    dims = np.shape(foreground)
    X = dims[0]
    Y = dims[1]
    labeled = np.zeros((X, Y))
    labels = []
    label = 1
    for r in range(X):
        for c in range(Y):
            if not foreground[r][c] or labeled[r][c]:
                continue
            q = []
            labeled[r][c] = label
            q.append([r,c])
            while len(q) != 0:
                pixel = q.pop(0)
                x = pixel[0]
                y = pixel[1]
                neighbors = FindNeighbors(x, y, X, Y)
                for n in neighbors:
                    nx = n[0]
                    ny = n[1]
                    if foreground[nx][ny] and not labeled[nx][ny]:
                        labeled[nx][ny] = label
                        q.append([nx, ny])
            labels.append(label)
            label+=1
    return labels, labeled


