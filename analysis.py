from Basic import Basic
from EpsilonConstraintsWeightedSum import EpsilonConstraintsWeightedSum
from TwoStage import TwoStage
from dataloader import *
from pprint import pprint

# Does 2-stage model perform better than basic model?
print("\n\n\n=== Does 2-stage model perform better than basic model? ===")
testdata = generate_data(do_random=False)

basic_model = Basic(testdata)
two_stage_model = TwoStage(testdata)

print("Basic:")
pprint(basic_model.drive())
print("Two Stage:")
pprint(two_stage_model.drive())


# How the cost of switching jobs affect the number of switching jobs and overall performance
print("\n\n\n=== How the cost of switching jobs affect the number of switching jobs and overall performance ===")
testdata = generate_data(do_random=False)
testdata.costOfSwitching = 5
two_stage_model = TwoStage(testdata)
print("Low cost of switching jobs")
pprint(two_stage_model.drive(scenario=0))

testdata.costOfSwitching = 50
two_stage_model = TwoStage(testdata)
print("High cost of switching jobs")
pprint(two_stage_model.drive(scenario=0))


# Whether availability of outsourcing affect the overall performance
print("\n\n\n=== Whether availability of outsourcing affect the overall performance ===")
testdata = generate_data(do_random=False)
testdata.outsourcingLimit = [[ 100 for t in testdata.periods ] for j in testdata.jobs]
two_stage_model = TwoStage(testdata)
print("Allow outsourcing")
pprint(two_stage_model.drive())

testdata.outsourcingLimit = [[ 0 for t in testdata.periods ] for j in testdata.jobs]
two_stage_model = TwoStage(testdata)
print("Disallow outsourcing")
pprint(two_stage_model.drive())