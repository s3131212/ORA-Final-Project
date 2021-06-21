from scipy.stats import norm
import random
import numpy as np
from pprint import pprint
import pandas as pd                # use DataFrame
import matplotlib.pyplot as plt    # draw the plot


DATA_PATH = "./data/"


class TestData():
    # init all the stuff that seldem change
    def __init__(self, nJob=2, nPeriods=24, nScenario=3, scenarioP=None,
        nWorker=90, twoSkill_worker=30, outsourcingLimit = 10,
        cHire=10, cSwitch=5, cOutsourcing=100,
        schedule="6 hours, per 3 hours", do_random=True):
        self.jobs = list(range(nJob))
        self.scenarios = list(range(nScenario))
        self.periods = list(range(nPeriods))

        # P of scenarios
        if type(scenarioP) != list or len(scenarioP) != self.scenarios:  
            # if not a given a list and P, same P for every scenario
            self.scenarioProbabilities = [ 1 / len(self.scenarios) for s in self.scenarios ]
        else:  # if given a list of P, use the ratio as P
            totalP = sum(scenarioP)  # if sum of P is not 1, keep the ratio between scenarios
            self.scenarioProbabilities = [t / totalP for t in scenarioP]

        # number of workers
        if type(nWorker)==list and len(nWorker)>=nJob:
            self.workerNumWithJobSkills = nWorker[0:nJob] # number of workers who have the skill for job j, may different by job
        else:
            self.workerNumWithJobSkills = [ nWorker for j in self.jobs ] # number of workers who have the skill for job j
        self.workerNumWithBothSkills = twoSkill_worker
        self.outsourcingLimit = [[ outsourcingLimit for t in self.periods ] for j in self.jobs]
        
        # cost
        self.costOfHiring = [cHire for i in self.periods] # cost of hiring a worker to work at period t
        self.costOfSwitching = cSwitch # cost of letting a worker change jobs at the middle of a day
        self.costOfOutsourcing = [[ cOutsourcing for t in self.periods ] for j in self.jobs] # Cost of satisfied an unit of demand by outsourcing
    
        # schedule
        if schedule == "6 hours, per 3 hours":
            self.schedules = [(0, 6), (3, 9), (6, 12), (9, 15), (12, 18), (15, 21), (18, 24), (21, 3)]
        elif schedule == "8 hours, per 4 hours":
            self.schedules = [(1, 9), (5, 13), (9, 17), (13, 21), (17, 1), (21, 5)]
        self.schedulesIncludePeriods = [
            [ ( 1 if i[0] <= t < i[1] else 0 ) if i[0] <= i[1] else ( 1 if i[0] <= t or t < i[1] else 0 ) for t in self.periods ] for i in self.schedules
        ] # schedule i include period t or not (binary)
    


def demand_generation_normal(length, offset=0, do_random=False):
    a = np.roll(np.array(range(0, length)) - length // 2, offset if not do_random else int(random.uniform(-length // 2, length // 2))) 
    a = a / ( random.uniform(3, 9) if do_random else 6 )
    return (norm().pdf(a) * ( random.uniform(60, 140) if do_random else 100 )).astype(int)


def generate_data(nJob=2, nPeriods=24, nScenario=3, scenarioP=None,
        nWorker=90, twoSkill_worker=30, outsourcingLimit = 10,
        cHire=10, cSwitch=5, cOutsourcing=100,
        schedule="6 hours, per 3 hours", do_random=True,
        demandTimes = 1,
        dataPath = None):
    if dataPath==None:
        data = TestData(nJob, nPeriods, nScenario, scenarioP,
            nWorker, twoSkill_worker, outsourcingLimit,
            cHire, cSwitch, cOutsourcing,
            schedule, do_random)
        # data.jobs = list(range(0, 2))
        # data.scenarios = list(range(0, 3))
        # data.scenarioProbabilities = [ 1 / len(data.scenarios) for s in data.scenarios ]
        # data.periods = list(range(0, 24))
        data.demands = [[ demandTimes * demand_generation_normal(len(data.periods), offset=s * (j * 2 - 1) * 4 ,do_random=do_random) * (s + 1) // (4 + j) for j in data.jobs ] for s in data.scenarios ] # demand of worker needed on job j at period t in scenario s
        # data.schedules = [(0, 6), (3, 9), (6, 12), (9, 15), (12, 18), (15, 21), (18, 24), (21, 3)]
        # data.schedulesIncludePeriods = [
        #     [ ( 1 if i[0] <= t < i[1] else 0 ) if i[0] <= i[1] else ( 1 if i[0] <= t or t < i[1] else 0 ) for t in data.periods ] for i in data.schedules
        # ] # schedule i include period t or not (binary)
        # data.workerNumWithJobSkills = [ 90 for j in data.jobs ] # number of workers who have the skill for job j
        # data.workerNumWithBothSkills = 30 # number of workers who have both skills
        # data.costOfHiring = [10 for i in data.periods] # cost of hiring a worker to work at period t
        # data.costOfSwitching = 5 # cost of letting a worker change jobs at the middle of a day
        # data.costOfOutsourcing = [[ 100 for t in data.periods ] for j in data.jobs] # Cost of satisfied an unit of demand by outsourcing for job j on period t
        # data.outsourcingLimit = [[ 10 for t in data.periods ] for j in data.jobs] # numbers on outsourcing workers for job j on period t
        #print("demands:")
        #pprint(data.demands)
        return data
    else:
        info = pd.read_csv(dataPath, header=1)
        data = TestData(nJob, nPeriods, 
            nScenario, scenarioP,
            nWorker, twoSkill_worker, outsourcingLimit,
            cHire, cSwitch, cOutsourcing,
            schedule, do_random)
        return data  


def save_and_plot(dfSample, n=None, frontierCol=None,
    color = "black", size = 10, labels = None, path = "./result/", model = "basic"):
    if n==None:
        sample_n = len(dfSample)
    else:
        sample_n = n
    fileName = model + "_s" + str(sample_n)

    # save results
    dfSample.to_csv(path + fileName + ".csv")
    print("Results have saved as", path + fileName + ".csv")

    # draw scatter-plot
    if frontierCol!=None:
        frotierFilter = (dfSample[frontierCol] == True)
        dominatedFilter = (dfSample[frontierCol] == False)
        # dominated points
        plt.scatter(dfSample['Cost'][dominatedFilter], dfSample['Redundant'][dominatedFilter],
            c = color[dominatedFilter],  s = size*0.7, label = labels,
            marker =  "x", alpha = 0.3)  # all points
        # frontier points
        plt.scatter(dfSample["Cost"][frotierFilter], dfSample["Redundant"][frotierFilter],
            c = color[frotierFilter],  s = size, label = "Frontier",
            marker =  "o", alpha = 0.5)  # frontier points
    else:
        # all points
        plt.scatter(dfSample['Cost'], dfSample['Redundant'],
            c = color,  s = size, label = labels,
            marker =  "o", alpha = 0.5)  # all points        
    plt.xlabel('Cost')       # set the name of x axis
    plt.ylabel('Redundant')
    plt.title("Scatter Plot of " + str(sample_n) + " sample") # set title
    plt.savefig(path + fileName + ".png", dpi=None)
    print("Plot has saved as", path + fileName + ".png")



def generate_and_save(n=1000, filePath="./data/",
    nJob=2, nPeriods=24, nScenario=3, scenarioP=None,
    nWorker=90, twoSkill_worker=30, outsourcingLimit = 10,
    cHire=10, cSwitch=5, cOutsourcing=100,
    schedule="6 hours, per 3 hours", do_random=True):
    # generate basic data
    data = generate_data(nJob, nPeriods, nScenario, scenarioP,
        nWorker, twoSkill_worker, outsourcingLimit,
        cHire, cSwitch, cOutsourcing, schedule, do_random)
    # get data from class
    menbers = [attr for attr in dir(data) if not callable(getattr(data, attr)) 
        and not attr.startswith("__")]   # get all not function menber in a class
    for m in menbers:
        if m != "demands":
            print(m)
            # data to DataFrame
            df = pd.DataFrame(getattr(data, m)) # get value by menber name  
            # data to csv
            fileName = filePath + m + ".csv"
            df.to_csv(fileName)
            print("Data generated has saved as", fileName)

    for i in range(n):
        # generate data
        data = generate_data(nJob, nPeriods, nScenario, scenarioP,
            nWorker, twoSkill_worker, outsourcingLimit,
            cHire, cSwitch, cOutsourcing, schedule, do_random)
        # data to DataFrame
        df = pd.DataFrame(data.demands)
        # data to csv
        fileName = filePath + "demands_"+ str(i) + ".csv"
        df.to_csv(fileName)
        print("Data generated has saved as", fileName)
    print("All", n, "data have saved.")


if __name__ == '__main__':
    generate_and_save(n=2, filePath="n1000")
