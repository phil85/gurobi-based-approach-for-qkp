import numpy as np
import pandas as pd
import gurobipy as gp


def run_gurobi_approach(nodes, edges, weights, budgets, params):

    # Setup model
    m = gp.Model()

    if 'output_flag' in params:
        m.setParam('OutputFlag', params['output_flag'])
    if 'time_limit' in params:
        m.setParam('TimeLimit', params['time_limit'])

    x = m.addVars(nodes, vtype=gp.GRB.BINARY)
    y = m.addVars(edges, vtype=gp.GRB.BINARY)
    m.setObjective(gp.quicksum(edges[i, j] * y[i, j] for i, j in edges), gp.GRB.MAXIMIZE)
    budget_constraint = m.addConstr(gp.quicksum(weights[i] * x[i] for i in nodes) <= 0)
    m.addConstrs(y[i, j] <= x[i] for i, j in edges)
    m.addConstrs(y[i, j] <= x[j] for i, j in edges)

    # Initialize results
    results = pd.DataFrame()

    for budget in budgets:

        # Set budget
        budget_constraint.rhs = budget

        # Optimize model
        m.optimize()

        # Get results
        result = pd.Series(dtype=object)
        try:
            result['items'] = [i for i in nodes if x[i].x > 0.5]
            result['ofv'] = m.objVal
            result['cpu'] = m.Runtime
            result['mip_gap'] = m.MipGap
        except:
            result['items'] = np.array([], dtype=int)
            result['ofv'] = 0
            result['cpu'] = params['time_limit']
            result['mip_gap'] = -1

        result['budget'] = budget
        result['budget_fraction'] = '{:.4f}'.format(budget / sum(weights))
        result['total_weight'] = sum([weights[i] for i in result['items']])
        result['approach'] = 'gurobi'

        # Convert to dataframe
        result = result.to_frame().transpose()

        # Append results to result
        results = pd.concat((results, result))

    return results


