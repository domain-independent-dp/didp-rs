def read(filename):
    with open(filename) as f:
        line = f.readline().rstrip().split()
        n = int(line[0])
        line = f.readline().rstrip().split()
        node_weights = [int(i) for i in line]
        edge_weights = {}
        for i in range(n):
            line = [int(w) for w in f.readline().rstrip().split()]
            for j, w in enumerate(line):
                if j > i and w > 0:
                    edge_weights[i, j] = w
        return n, node_weights, edge_weights
