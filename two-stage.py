import gurobipy as gp
from gurobipy import GRB
import sys
from pprint import pprint
import itertools

# Parameter
from dataloader import * 

def stage_1():
    m = gp.Model("assignment")
    # Variable
    x = m.addVars(len(schedules), name="x", vtype=GRB.INTEGER) # number of worker work on schedule i (integer)
    y = m.addVars(len(scenarios), len(jobs), len(periods), name="y", vtype=GRB.INTEGER) # number of worker work on job j on period t (integer)
    z = m.addVars(len(scenarios), len(periods), name="z") # number of worker changing job at the start of period t(integer)
    r = m.addVars(len(scenarios), len(jobs), len(periods), name="r") # number of extra manpower on job j, period t (integer)

    # Model
    m.addConstrs(
        ((gp.quicksum(y[s, j, t] for j in jobs) == gp.quicksum(x[i] * schedulesIncludePeriods[i][t] for i in range(len(schedules))))\
        for t in periods for s in scenarios), name='分配到工作的人數要符合上班人數') # 分配到工作的人數要符合上班人數
    m.addConstrs((y[s, j, t] >= demands[s][j][t] for j in jobs for t in periods for s in scenarios), name="值班人數要滿足需求") # 值班人數要滿足需求
    m.addConstrs((y[s, j, t] <= workerNumWithJobSkills[j] for j in jobs for t in periods for s in scenarios), name="任一時段，每個技能的值班人數 <= 總持有技能人數") # 任一時段，每個技能的值班人數 <= 總持有技能人數
    m.addConstr((gp.quicksum(x[i] for i in range(len(schedules))) <= sum(workerNumWithJobSkills) - workerNumWithBothSkills), name="總上班人數 <= 總員工數") # 總上班人數 <= 總員工數
    m.addConstrs(
        (r[s, j, t] == y[s, j, t] - demands[s][j][t] \
        for j in jobs for t in periods for s in scenarios), name='redundant') # redundant
    m.addConstrs(
        (z[s, t] >= y[s, j, t] - y[s, j, t - 1] \
        for j in jobs for t in range(1, len(periods)) for s in scenarios), name='中途轉換次數(取絕對值)_1') # 中途轉換次數(取絕對值)
    m.addConstrs(
        (z[s, t] >= y[s, j, t - 1] - y[s, j, t] \
        for j in jobs for t in range(1, len(periods)) for s in scenarios), name='中途轉換次數(取絕對值)_2') # 中途轉換次數(取絕對值)

    # Objective Function
    cost = m.addVar(name="Cost")
    m.addConstr(cost ==\
        gp.quicksum(x[i] * schedulesIncludePeriods[i][t] * costOfHiring[t] for i in range(len(schedules)) for t in periods) + \
        gp.quicksum(gp.quicksum(z[s, t] * costOfSwitching for t in periods) * scenarioProbabilities[s] for s in scenarios))

    m.setObjective(cost, GRB.MINIMIZE)
        
    # Optimize
    m.optimize()
    status = m.status
    if status == GRB.UNBOUNDED:
        raise Exception("Unbounded")
    elif status == GRB.OPTIMAL:
        print('The optimal objective is %g' % m.objVal)
        return {
            "x": [ x[i].x for i in range(len(schedules))],
            "y": [[[ y[s, j, t].x for t in periods] for j in jobs ] for s in scenarios],
            "Cost": cost.x
        }
    elif status == GRB.INFEASIBLE:
        raise Exception(f'Infeasible')
    else:
        raise Exception(f'Optimization was stopped with status {status}')
    return False


def stage_2(scenario, x, objfunc, epsilon=None):
    m = gp.Model("assignment")
    # Variable
    y = m.addVars(len(jobs), len(periods), name="y", vtype=GRB.INTEGER) # number of worker work on job j on period t (integer)
    z = m.addVars(len(periods), name="z") # number of worker changing job at the start of period t(integer)
    r = m.addVars(len(jobs), len(periods), name="r") # number of extra manpower on job j, period t (integer)

    # Model
    m.addConstrs(
        ((gp.quicksum(y[j, t] for j in jobs) == gp.quicksum(x[i] * schedulesIncludePeriods[i][t] for i in range(len(schedules))))\
        for t in periods), name='分配到工作的人數要符合上班人數') # 分配到工作的人數要符合上班人數
    m.addConstrs((y[j, t] >= demands[scenario][j][t] for j in jobs for t in periods), name="值班人數要滿足需求") # 值班人數要滿足需求
    m.addConstrs((y[j, t] <= workerNumWithJobSkills[j] for j in jobs for t in periods), name="任一時段，每個技能的值班人數 <= 總持有技能人數") # 任一時段，每個技能的值班人數 <= 總持有技能人數
    m.addConstr((gp.quicksum(x[i] for i in range(len(schedules))) <= sum(workerNumWithJobSkills) - workerNumWithBothSkills), name="總上班人數 <= 總員工數") # 總上班人數 <= 總員工數
    m.addConstrs(
        (r[j, t] == y[j, t] - demands[scenario][j][t] \
        for j in jobs for t in periods), name='redundant') # redundant
    m.addConstrs(
        (z[t] >= y[j, t] - y[j, t - 1] \
        for j in jobs for t in range(1, len(periods))), name='中途轉換次數(取絕對值)_1') # 中途轉換次數(取絕對值)
    m.addConstrs(
        (z[t] >= y[j, t - 1] - y[j, t] \
        for j in jobs for t in range(1, len(periods))), name='中途轉換次數(取絕對值)_2') # 中途轉換次數(取絕對值)

    # Objective Function
    cost = m.addVar(name="Cost")
    m.addConstr(cost == gp.quicksum((z[t] * costOfSwitching for t in periods)) )

    redundant = m.addVar(name="Redundant")
    m.addConstr(redundant == gp.quicksum(r[j, t] for j in jobs for t in periods))

    if objfunc == "Cost":
        m.setObjective(cost, GRB.MINIMIZE)
    elif objfunc == "Redundant":
        m.setObjective(redundant, GRB.MAXIMIZE)
    else:
        raise Exception(f"Wrong Objective Function {objfunc}")
    
    # Epsilon
    if objfunc == "Cost" and epsilon is not None:
        m.addConstr(redundant >= epsilon["Redundant"])

    # Optimize
    m.optimize()
    status = m.status
    if status == GRB.UNBOUNDED:
        raise Exception("Unbounded")
    elif status == GRB.OPTIMAL:
        return {
            "Cost": cost.x,
            "Redundant": redundant.x
            #"Redundant Variance": redundantVariance.x
        }
    elif status == GRB.INFEASIBLE:
        raise Exception(f'Infeasible')
    else:
        raise Exception(f'Optimization was stopped with status {status}')
    return False

stage_1_result = stage_1()
print("stage 1 results:")
pprint(stage_1_result, width=512)


objFuncs = ["Cost", "Redundant"]
epsilon_r = 3

def calculateEpsilonInterpolation(objFunc, r, epsilon_r, objValTable):
    objVals = [objValTable[objFuncOriginal][objFunc] for objFuncOriginal in objFuncs]
    return min(objVals) + (max(objVals) - min(objVals)) * r / epsilon_r

for scenario in scenarios:
    print("\n\n\n")
    print(f"================================ scenario {scenario} ================================")
    # objValTable[obj1][obj2] = 把 obj1 的 optimal solution 帶進 obj2 所得
    objValTable = { objFunc: stage_2(scenario, stage_1_result['x'], objFunc) for objFunc in objFuncs } 
    pprint(objValTable)

    epsilon = { objFunc: [ calculateEpsilonInterpolation(objFunc, r, epsilon_r, objValTable) for r in range(epsilon_r + 1) ] for objFunc in objFuncs[1:] }
    print("epsilon:", epsilon)
    epsilon_pairs = list(itertools.product(*[[(t[0], e) for e in t[1]] for t in epsilon.items()]))
    print("epsilon_pairs:", epsilon_pairs)

    solutions = list()
    for epsilon_pair in epsilon_pairs:
        solutions.append(stage_2(scenario, stage_1_result['x'], "Cost", epsilon=dict(epsilon_pair)))

    print("solutions:")
    pprint(solutions)