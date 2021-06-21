import gurobipy as gp
from gurobipy import GRB
import sys
from pprint import pprint
import pandas as pd                # use DataFrame
import time                        # caculate time spend

# Parameter
from dataloader import * 
import dataloader2

class Basic:
    def __init__(self, data):
        self.data = data
    def solve_model(self):
        m = gp.Model("basic")
        
        # Variable
        x = m.addVars(len(self.data.schedules), name="x", vtype=GRB.INTEGER) # number of worker work on schedule i (integer)
        y = m.addVars(len(self.data.scenarios), len(self.data.jobs), len(self.data.periods), name="y", vtype=GRB.INTEGER) # number of worker work on job j on period t (integer)
        z = m.addVars(len(self.data.scenarios), len(self.data.jobs), len(self.data.periods), name="z") # number of changes in job j at the start of period t(integer)
        r = m.addVars(len(self.data.scenarios), len(self.data.jobs), len(self.data.periods), name="r") # number of extra manpower on job j, period t (integer)
        l = m.addVars(len(self.data.scenarios), len(self.data.jobs), len(self.data.periods), name="l", vtype=GRB.INTEGER) # number of outsourcing worker on job j, period t (integer)


        # Model
        m.addConstrs(
            ((gp.quicksum(y[s, j, t] for j in self.data.jobs) == gp.quicksum(x[i] * self.data.schedulesIncludePeriods[i][t] for i in range(len(self.data.schedules))))\
            for t in self.data.periods for s in self.data.scenarios), name='分配到工作的人數要符合上班人數') # 分配到工作的人數要符合上班人數
        m.addConstrs((y[s, j, t] + l[s, j, t] >= self.data.demands[s][j][t] for j in self.data.jobs for t in self.data.periods for s in self.data.scenarios), name="值班人數要滿足需求") # 值班人數要滿足需求
        m.addConstrs((y[s, j, t] <= self.data.workerNumWithJobSkills[j] for j in self.data.jobs for t in self.data.periods for s in self.data.scenarios), name="任一時段，每個技能的值班人數 <= 總持有技能人數") # 任一時段，每個技能的值班人數 <= 總持有技能人數
        m.addConstrs((l[s, j, t] <= self.data.outsourcingLimit[j][t] for j in self.data.jobs for t in self.data.periods for s in self.data.scenarios), name="任一時段，每個外包人數 <= 可用外包人數") # 任一時段，每個外包人數 <= 可用外包人數
        m.addConstr((gp.quicksum(x[i] for i in range(len(self.data.schedules))) <= sum(self.data.workerNumWithJobSkills) - self.data.workerNumWithBothSkills), name="總上班人數 <= 總員工數") # 總上班人數 <= 總員工數
        m.addConstrs(
            (r[s, j, t] == y[s, j, t] + l[s, j, t] - self.data.demands[s][j][t] \
            for j in self.data.jobs for t in self.data.periods for s in self.data.scenarios), name='redundant') # redundant
        m.addConstrs(
            (z[s, j, t] >= y[s, j, t] - y[s, j, t - 1] \
            for j in self.data.jobs for t in range(1, len(self.data.periods)) for s in self.data.scenarios), name='中途轉換次數(取絕對值)_1') # 中途轉換次數(取絕對值)
        m.addConstrs(
            (z[s, j, t] >= y[s, j, t - 1] - y[s, j, t] \
            for j in self.data.jobs for t in range(1, len(self.data.periods)) for s in self.data.scenarios), name='中途轉換次數(取絕對值)_2') # 中途轉換次數(取絕對值)

        # Objective Function
        cost = m.addVar(name="Cost")
        m.addConstr(cost ==\
            gp.quicksum((
                gp.quicksum(gp.quicksum(y[s, j, t] for j in self.data.jobs) * self.data.costOfHiring[t] for t in self.data.periods) +
                gp.quicksum(z[s, j, t] for j in self.data.jobs for t in self.data.periods) * self.data.costOfSwitching +
                gp.quicksum(l[s, j, t] * self.data.costOfOutsourcing[j][t] for j in self.data.jobs for t in self.data.periods)
            ) * self.data.scenarioProbabilities[s] for s in self.data.scenarios))

        redundant = m.addVar(name="Redundant")
        m.addConstr(redundant == gp.quicksum(r[s, j, t] for j in self.data.jobs for t in self.data.periods for s in self.data.scenarios))

        m.setObjective(cost, GRB.MINIMIZE)

        m.write('workforce1.lp')

        # Optimize
        m.optimize()
        status = m.status
        if status == GRB.UNBOUNDED:
            raise "Unbounded"
        elif status == GRB.OPTIMAL:
            return {
                "Cost": cost.x,
                "Redundant": redundant.x
                #"Redundant Variance": redundantVariance.x
            }
        elif status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
            raise f'Optimization was stopped with status {status}'
    
    def drive(self):
        return self.solve_model()



################################################################################
SAMPLE_N = 1000         # how many samples
################################################################################


if __name__ == '__main__':
    print("Start to run Basic Model, ", SAMPLE_N, "samples.")
    startTime = time.time()   # record start time
    samples = pd.DataFrame()  # define a DataFrame to put results
    
    # loops for each sample
    for i in range(SAMPLE_N):
        sampleTime = time.time()  # caluate a single time of time

        # solve model
        data = dataloader2.generate_data(do_random = True)
        basic_model = Basic(data)                           # solve model
        newSample = basic_model.drive()                     # return result of model
        
        # record related data
        newSample['sample'] = i       # record which sample it is
        newSample['Time(sec)'] = time.time() - sampleTime   # record time spend
        samples = samples.append(newSample, ignore_index = True)  # save in DataFrame
        print("sample", i, "solved.")

    # save as csv and draw the plot
    dataloader2.save_and_plot(samples, color = [1]*len(samples), path = "./result/", model = "basic")
    print('Total time:', time.time() - startTime)
