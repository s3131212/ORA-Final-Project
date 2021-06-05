from scipy.stats import norm
import random
import numpy as np
from pprint import pprint

def demand_generation_normal(length, do_random=False):
    a = (np.array(range(0, length)) - length // 2) / ( random.uniform(3, 9) if do_random else 6 )
    return (norm().pdf(a) * ( random.uniform(60, 140) if do_random else 100 )).astype(int)

jobs = list(range(0, 2))
scenarios = list(range(0, 3))
periods = list(range(0, 24))
demands = [ [ demand_generation_normal(len(periods), do_random=True) for j in jobs ] for s in scenarios ] # demand of worker needed on job j at period t in scenario s
schedules = [(0, 6), (3, 9), (6, 12), (9, 15), (12, 18), (15, 21), (18, 24)]
schedulesIncludePeriods = [[ ( 1 if i[0] <= t < i[1] else 0 ) for t in periods ] for i in schedules] # schedule i include period t or not (binary)
workerNumWithJobSkills = [ 200 for j in jobs ] # number of workers who have the skill for job j
workerNumWithBothSkills = 10 # number of workers who have both skills
costOfHiring = [10 for i in periods] # cost of hiring a worker to work at period t
costOfSwitching = 5 # cost of letting a worker change jobs at the middle of a day

print("demands:")
pprint(demands)