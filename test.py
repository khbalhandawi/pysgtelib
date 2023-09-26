import numpy as np  
import os
from build.pysgtelib import Matrix, TrainingSet, Surrogate_Factory

data_dir = "data"

# Load training data
X = np.loadtxt(os.path.join(data_dir,"X.txt"), delimiter=",", dtype=str, ndmin=2)
Z = np.loadtxt(os.path.join(data_dir,"Z.txt"), delimiter=",", dtype=str, ndmin=2)
XX = np.loadtxt(os.path.join(data_dir,"XX.txt"), delimiter=",", dtype=str, ndmin=2)

# Create a Matrix instance  
Xm = Matrix('X', X.shape[0], X.shape[1], X)
Zm = Matrix('Z', Z.shape[0], Z.shape[1], Z)
XXm = Matrix('XX', XX.shape[0], XX.shape[1], XX)

TS = TrainingSet(Xm,Zm)

model = "TYPE PRS DEGREE 2"

S = Surrogate_Factory(TS,model)
S.build()
ZZ = S.predict(XXm)

