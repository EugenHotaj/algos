"""Algorithms to solve the makespan problem.

The jobs are taken from "GoCJ: Google Cloud Jobs Dataset” by Hussain et al.
"""

import argparse
import heapq


def grahams_algorithm(jobs, n_machines):
    """Uses Graham's algorithm to arrive at a makespan solution.

    Graham's algorithm greedily selects the least loaded machine to schedule
    the current job on. A minheap is used to efficiently determine the least
    loaded machine.

    The solution <= 2 * (1 - 1/n_machines) * optimal.
    """
    loads = [0] * n_machines
    for job in jobs:
        least_load = loads[0]
        heapq.heappushpop(loads, least_load + job)
    return heapq.nlargest(1, loads)[0]


def longest_processing_time(jobs, n_machines):
    """Uses Longest Processing Time (LPT) to arrive at a makespan solution.

    The solution <= (4/3 - 1/(3*n_machines)) * optimal.
    """
    jobs = sorted(jobs, reverse=True)
    return grahams_algorithm(jobs, n_machines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['graham', 'lpt'], help='Algorithm to use')
    parser.add_argument('--n_machines', type=int, required=True,
                        help='Number of available machines to schedule jobs on')
    FLAGS = parser.parse_args()
    assert FLAGS.n_machines > 0, '--n_machines must be positive'

    with open('GoCJ_Dataset_1000.txt', 'r') as file_:
        jobs = [int(job) for job in file_.readlines()]

    largest_job = max(jobs)
    ideal_load = sum(jobs) / FLAGS.n_machines
    lower_bound = max(largest_job, ideal_load)

    if FLAGS.algorithm == 'graham':
        solution = grahams_algorithm(jobs, FLAGS.n_machines)
    elif FLAGS.algorithm == 'lpt':
        solution = longest_processing_time(jobs, FLAGS.n_machines)
    else:
        raise ValueError('This should not be possible.')
    ratio = solution / lower_bound
    print(f'lower_bound={lower_bound}, solution={solution} ({ratio:.2%})')
