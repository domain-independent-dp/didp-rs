def read(input_file):
    with open(input_file) as f:
        line = f.readline()
        pair = line.split()
        n_patterns = int(pair[0])
        n_items = int(pair[1])
        item_to_patterns = [[] for _ in range(n_items)]
        pattern_to_items = [[] for _ in range(n_patterns)]

        for j in range(n_patterns):
            line = f.readline()
            row = [int(c) for c in line.split()]
            for i in range(n_items):
                if row[i] == 1:
                    item_to_patterns[i].append(j)
                    pattern_to_items[j].append(i)

    return item_to_patterns, pattern_to_items


def compute_item_to_neighbors(item_to_patterns, pattern_to_items):
    m = len(item_to_patterns)
    item_to_neighbors = [set() for _ in range(m)]
    for i in range(m):
        item_to_neighbors[i].add(i)
        for j in item_to_patterns[i]:
            for k in pattern_to_items[j]:
                item_to_neighbors[i].add(k)
    return item_to_neighbors


def item_order_to_pattern_order(item_to_patterns, items):
    produced = set()
    solution = []
    for i in items:
        for j in item_to_patterns[i]:
            if j not in produced:
                solution.append(j)
                produced.add(j)
    return solution


def validate(item_to_patterns, pattern_to_items, solution, cost):
    actual_cost = 0
    produced = set()
    open = set()
    for i in solution:
        if i < 0 or i > len(pattern_to_items) - 1:
            print("Pattern {} does not exist".format(i))
            return False
        if i in produced:
            print("Pattern {} is already produced".format(i))
            return False

        produced.add(i)
        for j in pattern_to_items[i]:
            if j not in open:
                open.add(j)
        actual_cost = max(actual_cost, len(open))
        closed = []
        for j in open:
            if all([k in produced for k in item_to_patterns[j]]):
                closed.append(j)
        for j in closed:
            open.remove(j)

    if len(produced) != len(pattern_to_items):
        print(
            "The number of produced patterns is {}, but should be {}".format(
                len(produced), len(pattern_to_items)
            )
        )
        return False

    if cost != actual_cost:
        print(
            "The cost of the solution {} mismatches the actual cost {}".format(
                cost, actual_cost
            )
        )
        return False

    return True
