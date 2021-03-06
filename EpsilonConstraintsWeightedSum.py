import gurobipy as gp
from gurobipy import GRB
import sys
from pprint import pprint
import itertools

# Parameter
from dataloader import * 

class EpsilonConstraintsWeightedSum:
    def __init__(self, data):
        self.data = data
    def solve_model(self, objfunc, epsilon=None):
        m = gp.Model("assignment")
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
        
        if objfunc == "Cost":
            m.setObjective(cost, GRB.MINIMIZE)
        elif objfunc == "Redundant":
            m.setObjective(redundant, GRB.MAXIMIZE)
        else:
            raise f"Wrong Objective Function {objfunc}"
        
        # Epsilon
        if objfunc == "Cost" and epsilon is not None:
            m.addConstr(redundant >= epsilon["Redundant"])

        # Optimize
        m.optimize()
        status = m.status
        if status == GRB.UNBOUNDED:
            raise Exception("Unbounded")
        elif status == GRB.OPTIMAL:
            #print('The optimal objective is %g' % m.objVal)
            return {
                "Cost": cost.x,
                "Redundant": redundant.x
            }
        elif status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
            raise Exception(f'Optimization was stopped with status {status}')

    def drive(self, epsilon_r=3):
        objFuncs = ["Cost", "Redundant"]

        # objValTable[obj1][obj2] = 把 obj1 的 optimal solution 帶進 obj2 所得
        objValTable = { objFunc: self.solve_model(objFunc) for objFunc in objFuncs } 
        #pprint(objValTable)

        def calculateEpsilonInterpolation(objFunc, r, epsilon_r, objValTable):
            objVals = [objValTable[objFuncOriginal][objFunc] for objFuncOriginal in objFuncs]
            return min(objVals) + (max(objVals) - min(objVals)) * r / epsilon_r

        epsilon = { objFunc: [ calculateEpsilonInterpolation(objFunc, r, epsilon_r, objValTable) for r in range(epsilon_r + 1) ] for objFunc in objFuncs[1:] }
        epsilon_pairs = list(itertools.product(*[[(t[0], e) for e in t[1]] for t in epsilon.items()]))

        solutions = list()
        for epsilon_pair in epsilon_pairs:
            solutions.append(self.solve_model("Cost", epsilon=dict(epsilon_pair)))
        
        return solutions

if __name__ == '__main__':
    epsilon_constraint_model = EpsilonConstraintsWeightedSum(generate_data())
    print(epsilon_constraint_model.drive())