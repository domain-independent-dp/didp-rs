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


def validate(n, node_weights, edge_weights, solution, cost):
    actual_cost = 0
    clean = set()

    for i in solution:
        if i < 0 or i > n - 1:
            print("Node {} does not exist".format(i))
            return False
        if i in clean:
            print("{} is already clean".format(i))
            return False

        n_robots = node_weights[i]
        for j in range(n):
            if (i, j) in edge_weights:
                n_robots += edge_weights[i, j]
            elif (j, i) in edge_weights:
                n_robots += edge_weights[j, i]
        for j in range(n):
            if j in clean:
                for k in range(n):
                    if k != i and k not in clean:
                        if (j, k) in edge_weights:
                            n_robots += edge_weights[j, k]
                        elif (k, j) in edge_weights:
                            n_robots += edge_weights[k, j]

        actual_cost = max(actual_cost, n_robots)
        clean.add(i)

    if len(clean) != n:
        print("The number of swept nodes is {}, but should be {}".format(len(clean), n))
        return False

    if actual_cost != cost:
        print(
            "The cost of solution {} mismatches the actual cost {}".format(
                cost, actual_cost
            )
        )
        return False

    return True
