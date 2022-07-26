import math


def read(filename):
    weights = []
    with open(filename) as f:
        n = int(f.readline().strip("\n"))
        c = int(f.readline().strip("\n"))
        for i in range(n):
            weights.append(int(f.readline().strip("\n")))

    return n, c, weights


def first_fit_decreasing(c, weights):
    weights_decreasing = sorted(weights, reverse=True)
    bins = [0]

    for w in weights_decreasing:
        no_fit = True
        for i, r in enumerate(bins):
            if r + w <= c:
                bins[i] += w
                no_fit = False
                break
        if no_fit:
            bins.append(w)

    ub = len(bins)

    return ub, weights_decreasing


def continuous_lb(c, weights):
    return math.ceil(sum(weights) / c)
