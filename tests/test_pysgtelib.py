import numpy as np
import os
from pysgtelib import Matrix, TrainingSet, Surrogate_Factory, Surrogate_Ensemble
from pysgtelib import Models, Metrics, KernelType, DistanceType, WeightType

# Assuming your data files are located in a 'tests/data' directory
data_dir = "data"

def load_data(filename):
    return np.loadtxt(os.path.join(data_dir, filename), delimiter=",", dtype=str, ndmin=2)

# Test loading of data
def test_load_data():
    X = load_data("X.txt")
    Z = load_data("Z.txt")
    XX = load_data("XX.txt")
    assert X is not None
    assert Z is not None
    assert XX is not None

# Test creation of TrainingSet
def test_create_training_set():
    X = load_data("X.txt")
    Z = load_data("Z.txt")
    TS = TrainingSet(X, Z)
    assert TS is not None

# Test Surrogate Factory with PRS model
def test_surrogate_factory_prs():
    X = load_data("X.txt")
    Z = load_data("Z.txt")
    XX = load_data("XX.txt")
    TS = TrainingSet(X, Z)

    model = {
        "TYPE": "PRS",
        "DEGREE": "2",
        "RIDGE": "0.001"
    }

    S = Surrogate_Factory(TS, model)
    S.build()
    ZZ = S.predict(XX)
    assert ZZ is not None

# Test Surrogate Factory with KS model and optimization
def test_surrogate_factory_ks_optimization():
    X = load_data("X.txt")
    Z = load_data("Z.txt")
    XX = load_data("XX.txt")
    TS = TrainingSet(X, Z)

    model = {
        "TYPE": Models.KS,
        "KERNEL": "OPTIM",
        "KERNEL_COEF": "OPTIM",
        "DISTANCE_TYPE": "OPTIM",
        "METRIC": Metrics.RMSECV,
        "BUDGET": 500
    }

    S = Surrogate_Factory(TS, model)
    S.build(optimize=False)
    initial_error = S.get_metric(Metrics.RMSECV, 0)

    S.build(optimize=True)
    S.optimize_parameters()
    optimized_error = S.get_metric(Metrics.RMSECV, 0)

    assert optimized_error <= initial_error