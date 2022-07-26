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
