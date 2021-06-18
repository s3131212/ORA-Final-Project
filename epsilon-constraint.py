import gurobipy as gp
from gurobipy import GRB
import sys
from pprint import pprint
import itertools

# Parameter
from dataloader import * 

def solve_model(objfunc, scenario, epsilon=None):
    m = gp.Model("assignment")
    # Variable
    x = m.addVars(len(schedules), name="x", vtype=GRB.INTEGER) # number of worker work on schedule i (integer)
    y = m.addVars(len(jobs), len(periods), name="y", vtype=GRB.INTEGER) # number of worker work on job j on period t (integer)
    z = m.addVars(len(jobs), len(periods), name="z") # number of changes in job j at the start of period t(integer)
    r = m.addVars(len(jobs), len(periods), name="r") # number of extra manpower on job j, period t (integer)
    l = m.addVars(len(jobs), len(periods), name="l", vtype=GRB.INTEGER) # number of outsourcing worker on job j, period t (integer)

    # Model
    m.addConstrs(
        ((gp.quicksum(y[j, t] for j in jobs) == gp.quicksum(x[i] * schedulesIncludePeriods[i][t] for i in range(len(schedules))))\
        for t in periods), name='分配到工作的人數要符合上班人數') # 分配到工作的人數要符合上班人數
    m.addConstrs((y[j, t] + l[j, t] >= demands[scenario][j][t] for j in jobs for t in periods), name="值班人數要滿足需求") # 值班人數要滿足需求
    m.addConstrs((y[j, t] <= workerNumWithJobSkills[j] for j in jobs for t in periods), name="任一時段，每個技能的值班人數 <= 總持有技能人數") # 任一時段，每個技能的值班人數 <= 總持有技能人數
    m.addConstrs((l[j, t] <= outsourcingLimit[j][t] for j in jobs for t in periods), name="任一時段，每個外包人數 <= 可用外包人數") # 任一時段，每個外包人數 <= 可用外包人數
    m.addConstr((gp.quicksum(x[i] for i in range(len(schedules))) <= sum(workerNumWithJobSkills) - workerNumWithBothSkills), name="總上班人數 <= 總員工數") # 總上班人數 <= 總員工數
    m.addConstrs(
        (r[j, t] == y[j, t] + l[j, t] - demands[scenario][j][t] \
        for j in jobs for t in periods), name='redundant') # redundant
    m.addConstrs(
        (z[j, t] >= y[j, t] - y[j, t - 1] \
        for j in jobs for t in range(1, len(periods))), name='中途轉換次數(取絕對值)_1') # 中途轉換次數(取絕對值)
    m.addConstrs(
        (z[j, t] >= y[j, t - 1] - y[j, t] \
        for j in jobs for t in range(1, len(periods))), name='中途轉換次數(取絕對值)_2') # 中途轉換次數(取絕對值)

    # Objective Function
    cost = m.addVar(name="Cost")
    m.addConstr(cost == \
            gp.quicksum(gp.quicksum(y[j, t] for j in jobs) * costOfHiring[t] for t in periods) +
            gp.quicksum(z[j, t] for j in jobs for t in periods) * costOfSwitching +
            gp.quicksum((l[j, t] * costOfOutsourcing[j][t] for j in jobs for t in periods))
        )

    redundant = m.addVar(name="Redundant")
    m.addConstr(redundant == gp.quicksum(r[j, t] for j in jobs for t in periods))

    #redundantVariance = m.addVar(name="Redundant Variance")
    #m.addConstr(redundantVariance ==\
    #    gp.quicksum((r[j, t] - gp.quicksum(r[j_, t_] for j_ in jobs for t_ in periods)) ** 2 for j in jobs for t in periods))
    
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


objFuncs = ["Cost", "Redundant"]
epsilon_r = 3

def calculateEpsilonInterpolation(objFunc, r, epsilon_r, objValTable):
    objVals = [objValTable[objFuncOriginal][objFunc] for objFuncOriginal in objFuncs]
    return min(objVals) + (max(objVals) - min(objVals)) * r / epsilon_r

for s in scenarios:
    # objValTable[obj1][obj2] = 把 obj1 的 optimal solution 帶進 obj2 所得
    objValTable = { objFunc: solve_model(objFunc, scenario=s) for objFunc in objFuncs } 
    pprint(objValTable)

    epsilon = { objFunc: [ calculateEpsilonInterpolation(objFunc, r, epsilon_r, objValTable) for r in range(epsilon_r + 1) ] for objFunc in objFuncs[1:] }
    print("epsilon:", epsilon)
    epsilon_pairs = list(itertools.product(*[[(t[0], e) for e in t[1]] for t in epsilon.items()]))
    print("epsilon_pairs:", epsilon_pairs)

    solutions = list()
    for epsilon_pair in epsilon_pairs:
        solutions.append(solve_model("Cost", epsilon=dict(epsilon_pair), scenario=s))

    print("solutions:")
    pprint(solutions)