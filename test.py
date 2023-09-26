import numpy as np  
import os
from build.pysgtelib import Matrix, TrainingSet, Surrogate_Factory
from build.pysgtelib import Models, Metrics

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

model = "TYPE KRIGING DISTANCE OPTIM RIDGE OPTIM METRIC OECV"
S = Surrogate_Factory(TS,model)
S.build()
ZZ = S.predict(XX)
 
E = S.get_metric(Metrics.OECV,0)
p = S.get_param()
print("Model default")
print("ridge:", p.get_ridge())
print("distance:", p.get_distance_type())
print("error:", E)

# Zh = S.get_Zh()
# Sh = S.get_Sh()

S.optimize_parameters()
p = S.get_param()
print("Model optimized")
print("ridge:", p.get_ridge())
print("distance:", p.get_distance_type())
print("error:", E)

E = S.get_metric(Metrics.OECV,0)
