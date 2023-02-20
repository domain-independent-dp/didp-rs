import math


def tsplib_round(x):
    return math.floor(x + 0.5)


def read_euc2d(n, f):
    line = f.readline()
    while True:
        if line.startswith("NODE_COORD_SECTION"):
            break
        line = f.readline()
    nodes = []
    edges = {}
    x = {}
    y = {}
    for _ in range(n):
        row = f.readline().split()
        i = int(row[0])
        nodes.append(i)
        x[i] = float(row[1])
        y[i] = float(row[2])
    for i in nodes:
        for j in nodes:
            if i >= j:
                continue
            edges[i, j] = tsplib_round(
                math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            )
            edges[j, i] = edges[i, j]
    return nodes, edges, True


def read_euc3d(n, f):
    line = f.readline()
    while True:
        if line.startswith("NODE_COORD_SECTION"):
            break
        line = f.readline()
    nodes = []
    edges = {}
    x = {}
    y = {}
    z = {}
    for _ in range(n):
        row = f.readline().split()
        i = int(row[0])
        nodes.append(i)
        x[i] = float(row[1])
        y[i] = float(row[2])
        z[i] = float(row[3])
    for i in nodes:
        for j in nodes:
            if i >= j:
                continue
            edges[i, j] = tsplib_round(
                math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)
            )
            edges[j, i] = edges[i, j]
    return nodes, edges, True


def read_geo(n, f):
    line = f.readline()
    while True:
        if line.startswith("NODE_COORD_SECTION"):
            break
        line = f.readline()
    PI = 3.141592
    nodes = []
    edges = {}
    latitude = {}
    longtitude = {}
    for _ in range(n):
        row = f.readline().split()
        i = int(row[0])
        nodes.append(i)
        x = float(row[1])
        deg = int(x)
        min = x - deg
        latitude[i] = PI * (deg + 5.0 * min / 3.0) / 180.0
        y = float(row[2])
        deg = int(y)
        min = y - deg
        longtitude[i] = PI * (deg + 5.0 * min / 3.0) / 180.0
    RRR = 6378.388
    for i in nodes:
        for j in nodes:
            if i >= j:
                continue
            q1 = math.cos(longtitude[i] - longtitude[j])
            q2 = math.cos(latitude[i] - latitude[j])
            q3 = math.cos(latitude[i] + latitude[j])
            edges[i, j] = int(
                RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
            )
            edges[j, i] = edges[i, j]
    return nodes, edges, True


def read_att(n, f):
    line = f.readline()
    while True:
        if line.startswith("NODE_COORD_SECTION"):
            break
        line = f.readline()
    nodes = []
    edges = {}
    x = {}
    y = {}
    for _ in range(n):
        row = f.readline().split()
        i = int(row[0])
        nodes.append(i)
        x[i] = float(row[1])
        y[i] = float(row[2])
    for i in nodes:
        for j in nodes:
            if i >= j:
                continue
            xd = x[i] - x[j]
            yd = y[i] - y[j]
            rij = math.sqrt((xd**2 + yd**2) / 10.0)
            tij = tsplib_round(rij)
            if tij < rij:
                edges[i, j] = tij + 1
            else:
                edges[i, j] = tij
            edges[j, i] = edges[i, j]
    return nodes, edges, True


def read_ceil2d(n, f):
    line = f.readline()
    while True:
        if line.startswith("NODE_COORD_SECTION"):
            break
        line = f.readline()
    nodes = []
    edges = {}
    x = {}
    y = {}
    for _ in range(n):
        row = f.readline().split()
        i = int(row[0])
        nodes.append(i)
        x[i] = float(row[1])
        y[i] = float(row[2])
    for i in nodes:
        for j in nodes:
            if i >= j:
                continue
            edges[i, j] = math.ceil(math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2))
            edges[j, i] = edges[i, j]
    return nodes, edges, True


def read_full_matrix(n, f):
    line = f.readline()
    while True:
        if line.startswith("EDGE_WEIGHT_SECTION"):
            break
        line = f.readline()
    edges = {}
    line = f.readline()
    entries = [int(c) for c in line.split()]
    position = 0
    symmetric = True
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if position >= len(entries):
                line = f.readline()
                entries = [int(c) for c in line.split()]
                position = 0
            edges[i, j] = entries[position]
            position += 1
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            if edges[i, j] != edges[j, i]:
                symmetric = False
                break
    return edges, symmetric


def read_upper_row(n, f):
    line = f.readline()
    while True:
        if line.startswith("EDGE_WEIGHT_SECTION"):
            break
        line = f.readline()
    edges = {}
    line = f.readline()
    entries = [int(c) for c in line.split()]
    position = 0
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            if position >= len(entries):
                line = f.readline()
                entries = [int(c) for c in line.split()]
                position = 0
            edges[i, j] = entries[position]
            edges[j, i] = edges[i, j]
            position += 1
    return edges, True


def read_lower_row(n, f):
    line = f.readline()
    while True:
        if line.startswith("EDGE_WEIGHT_SECTION"):
            break
        line = f.readline()
    edges = {}
    line = f.readline()
    entries = [int(c) for c in line.split()]
    position = 0
    for i in range(2, n + 1):
        for j in range(1, i):
            if position >= len(entries):
                line = f.readline()
                entries = [int(c) for c in line.split()]
                position = 0
            edges[i, j] = entries[position]
            edges[j, i] = edges[i, j]
            position += 1
    return edges, True


def read_upper_diag_row(n, f):
    line = f.readline()
    while True:
        if line.startswith("EDGE_WEIGHT_SECTION"):
            break
        line = f.readline()
    edges = {}
    line = f.readline()
    entries = [int(c) for c in line.split()]
    position = 0
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            if position >= len(entries):
                line = f.readline()
                entries = [int(c) for c in line.split()]
                position = 0
            edges[i, j] = entries[position]
            if i != j:
                edges[j, i] = edges[i, j]
            position += 1
    return edges, True


def read_lower_diag_row(n, f):
    line = f.readline()
    while line:
        if line.startswith("EDGE_WEIGHT_SECTION"):
            break
        line = f.readline()
    entries = []
    line = f.readline()
    edges = {}
    line = f.readline()
    entries = [int(c) for c in line.split()]
    position = 0
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            if position >= len(entries):
                line = f.readline()
                entries = [int(c) for c in line.split()]
                position = 0
            edges[i, j] = entries[position]
            if i != j:
                edges[j, i] = edges[i, j]
            position += 1
    return edges, True


def read_edges(edge_weight_type, edge_weight_format, n, f):
    if edge_weight_type == "EUC_2D":
        return read_euc2d(n, f)

    if edge_weight_type == "EUC_3D":
        return read_euc3d(n, f)

    if edge_weight_type == "GEO":
        return read_geo(n, f)

    if edge_weight_type == "ATT":
        return read_att(n, f)

    if edge_weight_type == "CEIL_2D":
        return read_ceil2d(n, f)

    nodes = list(range(1, n + 1))

    if edge_weight_format == "FULL_MATRIX":
        return nodes, *read_full_matrix(n, f)

    if edge_weight_format == "UPPER_ROW":
        return nodes, *read_upper_row(n, f)

    if edge_weight_format == "LOWER_ROW":
        return nodes, *read_lower_row(n, f)

    if edge_weight_format == "UPPER_DIAG_ROW":
        return nodes, *read_upper_diag_row(n, f)

    if edge_weight_format == "LOWER_DIAG_ROW":
        return nodes, *read_lower_diag_row(n, f)

    return None, None, None


def read_tsp(filename):
    with open(filename) as f:
        n = None
        edge_weight_type = None
        edge_weight_format = None
        while (
            n is None
            or edge_weight_type is None
            or (edge_weight_type == "EXPLICIT" and edge_weight_format is None)
        ):
            line = f.readline()
            if line.startswith("TYPE"):
                assert "TSP" in line.split() or "ATSP" in line.split()
            if line.startswith("DIMENSION"):
                n = int(line.split()[-1])
            if line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split()[-1]
            if line.startswith("EDGE_WEIGHT_FORMAT"):
                edge_weight_format = line.split()[-1]

        nodes, edges, symmetric = read_edges(edge_weight_type, edge_weight_format, n, f)
        return n, nodes, edges, symmetric


def read_demand(n, f):
    line = f.readline()
    while line:
        if line.startswith("DEMAND_SECTION"):
            break
        line = f.readline()

    demand = {}

    for _ in range(n):
        line = f.readline().split()
        demand[int(line[0])] = int(line[1])

    return demand


def read_depots(n, f):
    line = f.readline()
    while line:
        if line.startswith("DEPOT_SECTION"):
            break
        line = f.readline()

    depots = []
    d = int(f.readline().rstrip())
    while d != -1:
        depots.append(d)
        d = int(f.readline().rstrip())

    return depots


def read_cvrp(filename):
    with open(filename) as f:
        n = None
        edge_weight_type = None
        edge_weight_format = None
        capacity = None
        while (
            n is None
            or edge_weight_type is None
            or capacity is None
            or (edge_weight_type == "EXPLICIT" and edge_weight_format is None)
        ):
            line = f.readline()
            if line.startswith("TYPE"):
                assert "CVRP" in line.split()
            if line.startswith("DIMENSION"):
                n = int(line.split()[-1])
            if line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split()[-1]
            if line.startswith("CAPACITY"):
                capacity = int(line.split()[-1])
            if line.startswith("EDGE_WEIGHT_FORMAT"):
                edge_weight_format = line.split()[-1]

        nodes, edges, symmetric = read_edges(edge_weight_type, edge_weight_format, n, f)
        demand = read_demand(n, f)
        depots = read_depots(n, f)
        assert len(depots) == 1
        return n, nodes, edges, capacity, demand, depots[0], symmetric


def validate_cvrp(n, nodes, edges, capacity, demand, depot, solution, cost, k=None):
    if solution[0] != depot:
        print(
            "The tour does not start from the depot {} but from {}".format(
                depot, solution[0]
            )
        )
        return False
    previous = solution[0]
    actual_cost = 0
    load = 0
    n_vehicles = 0
    visited = set([depot])
    for i in solution[1:]:
        if previous == depot:
            n_vehicles += 1
            if k is not None and n_vehicles > k:
                print(
                    "The number of vehicles {} exceeds the limit {}".format(
                        n_vehicles, k
                    )
                )
                return False

        if i not in nodes:
            print("No such customer {}".format(i))
            return False

        if (previous, i) not in edges:
            print("No such edge ({}, {})".format(previous, i))
            return False

        actual_cost += edges[previous, i]

        if i == depot:
            load = 0
        else:
            if i in visited:
                print("Customer {} is already visited".format(i))
                return False
            visited.add(i)
            load += demand[i]
            if load > capacity:
                print(
                    "load {} at customer {} exceeds the capacity {}".format(
                        load, i, capacity
                    )
                )
                return False

        previous = i

    if previous != depot:
        print(
            "The tour does not return to the depot {} but to {}".format(depot, previous)
        )
        return False

    if len(visited) != n:
        print(
            "The number of visited customers is {}, but should be {}".format(
                len(visited), n
            )
        )

    if actual_cost != cost:
        print(
            "The cost of the solution {} mismatches the actual cost {}".format(
                cost, actual_cost
            )
        )
        return False

    return True
