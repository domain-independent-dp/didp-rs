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


def validate(n, c, weights, solution, cost):
    if cost != len(solution):
        print(
            "The cost of solution {} mismatches the actual cost {}".format(
                cost, len(solution)
            )
        )
    packed = {}
    for (bin, items) in enumerate(solution):
        weight_sum = 0
        for i in items:
            if i < 0 or i > n - 1:
                print("item {} does not exist".format(i))
            if i in packed:
                print(
                    "item {} in bin {} is already scheduled in bin {}",
                    i,
                    bin,
                    packed[i],
                )
                return False
            packed[i] = bin
            weight_sum += weights[i]
        if weight_sum > c:
            print(
                "The sum of weight in bin {} exceeds the capacity of {}".format(bin, c)
            )
            return False

    if len(packed) != n:
        print(
            "The number of packed items is {}, but should be {}".format(len(packed), n)
        )

    return True
