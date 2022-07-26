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
