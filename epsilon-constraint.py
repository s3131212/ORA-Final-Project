import gurobipy as gp
from gurobipy import GRB
import sys
from pprint import pprint
import itertools
import pandas as pd                # use DataFrame
import time                        # caculate time spend
import matplotlib.pyplot as plt    # draw the plot

# Parameter
from dataloader import * 
import dataloader2



def solve_model(data, objfunc, scenario, epsilon=None):
    m = gp.Model("assignment")
    # Variable
    x = m.addVars(len(data.schedules), name="x", vtype=GRB.INTEGER) # number of worker work on schedule i (integer)
    y = m.addVars(len(data.jobs), len(data.periods), name="y", vtype=GRB.INTEGER) # number of worker work on job j on period t (integer)
    z = m.addVars(len(data.jobs), len(data.periods), name="z") # number of changes in job j at the start of period t(integer)
    r = m.addVars(len(data.jobs), len(data.periods), name="r") # number of extra manpower on job j, period t (integer)
    l = m.addVars(len(data.jobs), len(data.periods), name="l", vtype=GRB.INTEGER) # number of outsourcing worker on job j, period t (integer)

    # Model
    m.addConstrs(
        ((gp.quicksum(y[j, t] for j in data.jobs) == gp.quicksum(x[i] * data.schedulesIncludePeriods[i][t] for i in range(len(data.schedules))))\
        for t in data.periods), name='分配到工作的人數要符合上班人數') # 分配到工作的人數要符合上班人數
    m.addConstrs((y[j, t] + l[j, t] >= data.demands[scenario][j][t] for j in data.jobs for t in data.periods), name="值班人數要滿足需求") # 值班人數要滿足需求
    m.addConstrs((y[j, t] <= data.workerNumWithJobSkills[j] for j in data.jobs for t in data.periods), name="任一時段，每個技能的值班人數 <= 總持有技能人數") # 任一時段，每個技能的值班人數 <= 總持有技能人數
    m.addConstrs((l[j, t] <= data.outsourcingLimit[j][t] for j in data.jobs for t in data.periods), name="任一時段，每個外包人數 <= 可用外包人數") # 任一時段，每個外包人數 <= 可用外包人數
    m.addConstr((gp.quicksum(x[i] for i in range(len(data.schedules))) <= sum(data.workerNumWithJobSkills) - data.workerNumWithBothSkills), name="總上班人數 <= 總員工數") # 總上班人數 <= 總員工數
    m.addConstrs(
        (r[j, t] == y[j, t] + l[j, t] - data.demands[scenario][j][t] \
        for j in data.jobs for t in data.periods), name='redundant') # redundant
    m.addConstrs(
        (z[j, t] >= y[j, t] - y[j, t - 1] \
        for j in data.jobs for t in range(1, len(data.periods))), name='中途轉換次數(取絕對值)_1') # 中途轉換次數(取絕對值)
    m.addConstrs(
        (z[j, t] >= y[j, t - 1] - y[j, t] \
        for j in data.jobs for t in range(1, len(data.periods))), name='中途轉換次數(取絕對值)_2') # 中途轉換次數(取絕對值)

    # Objective Function
    cost = m.addVar(name="Cost")
    m.addConstr(cost == \
            gp.quicksum(gp.quicksum(y[j, t] for j in data.jobs) * data.costOfHiring[t] for t in data.periods) +
            gp.quicksum(z[j, t] for j in data.jobs for t in data.periods) * data.costOfSwitching +
            gp.quicksum((l[j, t] * data.costOfOutsourcing[j][t] for j in data.jobs for t in data.periods))
        )

    redundant = m.addVar(name="Redundant")
    m.addConstr(redundant == gp.quicksum(r[j, t] for j in data.jobs for t in data.periods))

    #redundantVariance = m.addVar(name="Redundant Variance")
    #m.addConstr(redundantVariance ==\
    #    gp.quicksum((r[j, t] - gp.quicksum(r[j_, t_] for j_ in jobs for t_ in data.periods)) ** 2 for j in data.jobs for t in data.periods))
    
    if objfunc == "Cost":
        m.setObjective(cost, GRB.MINIMIZE)
    elif objfunc == "Redundant":
        m.setObjective(redundant, GRB.MAXIMIZE)
    #elif objfunc == "Redundant Variance":
    #    m.setObjective(redundantVariance, GRB.MINIMIZE)
    else:
        raise f"Wrong Objective Function {objfunc}"
    
    # Epsilon
    if objfunc == "Cost" and epsilon is not None:
        m.addConstr(redundant >= epsilon["Redundant"])
        #m.addConstr(redundantVariance <= epsilon["Redundant Variance"])

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

################################################################################
SAMPLE_N = 25         # how many samples
objFuncs = ["Cost", "Redundant"]
epsilon_r = 6#3
# basic cost, switch:hire:outsource = 5:10:100
ch = 20      # cost of hire, basic 10
csw = 10             # cost of switching job, basic 5
ol = 10                # outsourcingLimit, basic 10
dt = 1                  # times demand to make demand larger, basic 1
################################################################################

def calculateEpsilonInterpolation(objFunc, r, epsilon_r, objValTable):
    objVals = [objValTable[objFuncOriginal][objFunc] for objFuncOriginal in objFuncs]
    return min(objVals) + (max(objVals) - min(objVals)) * r / epsilon_r


# for s in scenarios:
#     # objValTable[obj1][obj2] = 把 obj1 的 optimal solution 帶進 obj2 所得
#     objValTable = { objFunc: solve_model(objFunc, scenario=s) for objFunc in objFuncs } 
#     pprint(objValTable)

#     epsilon = { objFunc: [ calculateEpsilonInterpolation(objFunc, r, epsilon_r, objValTable) for r in range(epsilon_r + 1) ] for objFunc in objFuncs[1:] }
#     print("epsilon:", epsilon)
#     epsilon_pairs = list(itertools.product(*[[(t[0], e) for e in t[1]] for t in epsilon.items()]))
#     print("epsilon_pairs:", epsilon_pairs)

#     solutions = list()
#     for epsilon_pair in epsilon_pairs:
#         solutions.append(solve_model("Cost", epsilon=dict(epsilon_pair), scenario=s))

#     print("solutions:")
#     pprint(solutions)




if __name__ == '__main__':
    print("Start to run Epsilon Basic Model", SAMPLE_N, "samples.")
    startTime = time.time()   # record start time
    solutions = pd.DataFrame()  # define a DataFrame to put results of each epsilon, each sample
    
    # loops for each sample
    for i in range(SAMPLE_N):
        sampleTime = time.time()  # caluate a single time of time
        data = dataloader2.generate_data(do_random = (SAMPLE_N!=1), 
            cHire=ch, cSwitch=csw, outsourcingLimit=ol, demandTimes=dt)

        # solve model
        for s in data.scenarios:
            # objValTable[obj1][obj2] = 把 obj1 的 optimal solution 帶進 obj2 所得
            objValTable = { objFunc: solve_model(data, objFunc, scenario=s) for objFunc in objFuncs } 
            # print("objValTable:")
            # pprint(objValTable)

            epsilon = { objFunc: [ calculateEpsilonInterpolation(objFunc, r, epsilon_r, objValTable) for r in range(epsilon_r + 1) ] for objFunc in objFuncs[1:] }
            # print("epsilon:", epsilon)
            epsilon_pairs = list(itertools.product(*[[(t[0], e) for e in t[1]] for t in epsilon.items()]))
            # print("epsilon_pairs:", epsilon_pairs)

            solutions_inSample = list()
            for epsilon_pair in epsilon_pairs:
                sol = solve_model(data, "Cost", epsilon=dict(epsilon_pair), scenario=s)
                sol['epsilon'] = epsilon_pair[0][1]   # save the epsilon
                sol['binding(1)'] = sol['Redundant'] == epsilon_pair[0][1]  # binding on objective or not
                sol['sample'] = i       # record which sample it is
                sol['scenario'] = s     # record what scenario it is                 # record demand
                sol['Time(sec)'] = time.time() - sampleTime   # record time spend  
                solutions_inSample.append(sol)

            # save result           
            solutions = solutions.append(solutions_inSample, ignore_index = True)

            # draw a solution
            df = pd.DataFrame(solutions_inSample)
            plt.plot(df['Cost'], df['Redundant'],
                linestyle='-', markersize=1, alpha = 0.5)  # all points
            highlightfilter = (df['binding(1)'] == True)
            plt.scatter(df['Cost'][highlightfilter], df['Redundant'][highlightfilter],
                marker='x', s=2)

        print("sample", i, "solved.")


    # save as csv and draw the plot
    dataloader2.save_and_plot(solutions, n=SAMPLE_N,
        frontierCol="binding(1)",
        color = solutions['scenario'],
        labels = ["scenario " + str(i) for i in data.scenarios],
        path = "./result/", model = "epsilon_r"+str(epsilon_r)\
            + "_cH" + str(ch)\
            +"_cS" + str(csw) + "_osL" + str(ol) + "_d" + str(dt))
    print('Total time:', time.time() - startTime)