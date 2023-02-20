def read(filename):
    with open(filename) as f:
        line = f.readline().strip()
        while line != "<end>":
            if line == "<number of tasks>":
                number_of_tasks = int(f.readline().strip())
            if line == "<cycle time>":
                cycle_time = int(f.readline().strip())
            if line == "<task times>":
                task_times = {}
                for _ in range(number_of_tasks):
                    pair = f.readline().strip().split()
                    task_times[int(pair[0])] = int(pair[1])
            if line == "<precedence relations>":
                predecessors = {i: [] for i in range(1, number_of_tasks + 1)}
                followers = {i: [] for i in range(1, number_of_tasks + 1)}
                relation = f.readline().strip()
                while len(relation) > 0:
                    relation = relation.split(",")
                    predecessor = int(relation[0])
                    follower = int(relation[1])
                    predecessors[follower].append(predecessor)
                    followers[predecessor].append(follower)
                    relation = f.readline().strip()

            line = f.readline().strip()

    return number_of_tasks, cycle_time, task_times, predecessors, followers


def validate(number_of_tasks, cycle_time, task_times, predecessors, solution, cost):
    if cost != len(solution):
        print(
            "The cost of solution {} mismatches the actual cost {}".format(
                cost, len(solution)
            )
        )
    scheduled = {}
    for (i, tasks) in enumerate(solution):
        time = 0
        for j in tasks:
            if j < 1 or j > number_of_tasks:
                print("task {} in station {} does not exist".format(j, i))
                return False
            if j in scheduled:
                print(
                    "task {} in station {} is already scheduled in station {}".format(
                        j, i, scheduled[j]
                    )
                )
                return False
            for k in predecessors[j]:
                if k not in scheduled and k not in tasks:
                    print(
                        "task {}, which is a predecessor of task {} in station {}, is not scheduled".format(
                            k, j, i
                        )
                    )
                    return False
            scheduled[j] = i
            time += task_times[j]
        if time > cycle_time:
            print(
                "station {} has total time {}, which exceeds the cycle time".format(
                    i + 1, time
                )
            )
            return False

    if len(scheduled) != number_of_tasks:
        print(
            "The number of scheduled tasks is {}, but should be {}".format(
                len(scheduled), number_of_tasks
            )
        )

    return True
