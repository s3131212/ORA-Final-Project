import gurobipy as gp
from gurobipy import GRB
import sys
from pprint import pprint

m = gp.Model("assignment")

# Parameter
from dataloader import * 

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
m.setObjective(
    gp.quicksum(
        (gp.quicksum(gp.quicksum(y[s, j, t] for j in jobs) * costOfHiring[t] for t in periods) +
        gp.quicksum((z[s, t] for t in periods)) * costOfSwitching)
    for s in scenarios), GRB.MINIMIZE)

m.write('workforce1.lp')

# Optimize
m.optimize()
status = m.status
if status == GRB.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')
    sys.exit(0)
if status == GRB.OPTIMAL:
    print('The optimal objective is %g' % m.objVal)
    print([ x[i].x for i in range(len(schedules))])
    sys.exit(0)
if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)
    sys.exit(0)

# do IIS
print('The model is infeasible; computing IIS')
m.computeIIS()
if m.IISMinimal:
    print('IIS is minimal\n')
else:
    print('IIS is not minimal\n')
print('\nThe following constraint(s) cannot be satisfied:')
for c in m.getConstrs():
    if c.IISConstr:
        print('%s' % c.constrName)

