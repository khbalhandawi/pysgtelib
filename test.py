import numpy as np  
import os
from build.pysgtelib import Matrix, TrainingSet, Surrogate_Factory, Surrogate_Ensemble
from build.pysgtelib import Models, Metrics, KernelType, DistanceType, WeightType

data_dir = "data"

# Load training data
X = np.loadtxt(os.path.join(data_dir,"X.txt"), delimiter=",", dtype=str, ndmin=2)
Z = np.loadtxt(os.path.join(data_dir,"Z.txt"), delimiter=",", dtype=str, ndmin=2)
XX = np.loadtxt(os.path.join(data_dir,"XX.txt"), delimiter=",", dtype=str, ndmin=2)

# Create a Matrix instance  
# Xm = Matrix('X', X.shape[0], X.shape[1], X)
# Zm = Matrix('Z', Z.shape[0], Z.shape[1], Z)
# XXm = Matrix('XX', XX.shape[0], XX.shape[1], XX)

TS = TrainingSet(X,Z)

# test string dict
model = {
    "TYPE" : "PRS",
    "DEGREE" : "2",
    "RIDGE" : "0.001"
}

S = Surrogate_Factory(TS,model)
S.build()
ZZ = S.predict(XX)

# test normal dict

model = {
    "TYPE" : Models.KS,
    "KERNEL" : "OPTIM",
    "KERNEL_COEF" : "OPTIM",
    "DISTANCE_TYPE" : "OPTIM",
    "METRIC" : Metrics.RMSECV,
    "BUDGET" : 500
}

S = Surrogate_Factory(TS,model)
S.build(optimize=False)
ZZ = S.predict(XX)
E = S.get_metric(Metrics.RMSECV,0)

p = S.get_param()
print("================\nModel default\n================")
print("kernel:", p.get_kernel_type())
print("kernel_coeff:", p.get_kernel_coef())
print("distance:", p.get_distance_type())
print("ERROR:", E)


S = Surrogate_Factory(TS,model)
S.build(optimize=True)
S.optimize_parameters()
E = S.get_metric(Metrics.RMSECV,0)

p = S.get_param()
print("================\nModel optimized\n================")
print("kernel:", p.get_kernel_type())
print("kernel_coeff:", p.get_kernel_coef())
print("distance:", p.get_distance_type())
print("ERROR:", E)

# Test Ensemble models
models = [
    {
        "TYPE" : Models.PRS,
        "DEGREE" : 2,
        "RIDGE" : 1e-3
    },
    {
        "TYPE" : Models.KRIGING,
        "RIDGE" : 1e-3,
        "DISTANCE" : DistanceType.NORM2
    },
    {
        "TYPE" : Models.LOWESS,
        "KERNEL" : KernelType.D1,
        "KERNEL_COEF" : 0.2,
        "DISTANCE" : DistanceType.NORM2
    },  
]

model = {
    "TYPE" : Models.ENSEMBLE,
    "WEIGHT" : "OPTIM",
    "METRIC" : Metrics.OECV,
    "PRESET" : "DEFAULT",
    "BUDGET" : 500
}
S:Surrogate_Ensemble = Surrogate_Factory(TS,model)
print(S.model_list_display())

S.model_list_remove_all()
for model in models:
    S.model_list_add(model)

print(S.model_list_display())

S.build()
ZZ = S.predict(XX)
E = S.get_metric(Metrics.OECV,0)

S.optimize_parameters()

p = S.get_param()
w = p.get_weight()

print(p.display_x())

# Test Ensemble models with pretrained models

