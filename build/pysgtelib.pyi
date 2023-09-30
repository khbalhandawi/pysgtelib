from typing import Any, ClassVar, Dict, Union

from typing import overload
import numpy
AOE: Metrics
AOECV: Metrics
ARMSE: Metrics
ARMSECV: Metrics
CN: Models
D1: KernelType
D2: KernelType
D3: KernelType
D4: KernelType
D5: KernelType
D6: KernelType
D7: KernelType
DYNATREE: Models
EFIOE: Metrics
EFIOECV: Metrics
EMAX: Metrics
EMAXCV: Metrics
ENSEMBLE: Models
EXTERN: WeightType
I0: KernelType
I1: KernelType
I2: KernelType
I3: KernelType
I4: KernelType
KRIGING: Models
KS: Models
LINEAR: Models
LINV: Metrics
LOWESS: Models
NORM1: DistanceType
NORM2: DistanceType
NORM2_CAT: DistanceType
NORM2_IS0: DistanceType
NORMINF: DistanceType
NORM_0: NormType
NORM_1: NormType
NORM_2: NormType
NORM_INF: NormType
OE: Metrics
OECV: Metrics
OPTIM: WeightType
PRS: Models
PRS_CAT: Models
PRS_EDGE: Models
RBF: Models
RMSE: Metrics
RMSECV: Metrics
SELECT: WeightType
SVN: Models
TGP: Models
WTA1: WeightType
WTA3: WeightType

class DistanceType:
    __members__: ClassVar[dict] = ...  # read-only
    NORM1: ClassVar[DistanceType] = ...
    NORM2: ClassVar[DistanceType] = ...
    NORM2_CAT: ClassVar[DistanceType] = ...
    NORM2_IS0: ClassVar[DistanceType] = ...
    NORMINF: ClassVar[DistanceType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class KernelType:
    __members__: ClassVar[dict] = ...  # read-only
    D1: ClassVar[KernelType] = ...
    D2: ClassVar[KernelType] = ...
    D3: ClassVar[KernelType] = ...
    D4: ClassVar[KernelType] = ...
    D5: ClassVar[KernelType] = ...
    D6: ClassVar[KernelType] = ...
    D7: ClassVar[KernelType] = ...
    I0: ClassVar[KernelType] = ...
    I1: ClassVar[KernelType] = ...
    I2: ClassVar[KernelType] = ...
    I3: ClassVar[KernelType] = ...
    I4: ClassVar[KernelType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Matrix:
    def __init__(self, arg0: str, arg1: int, arg2: int, arg3: numpy.ndarray[numpy.float64]) -> None: ...
    def get(self, i: int, j: int) -> float: ...
    def get_nb_cols(self) -> int: ...
    def get_nb_rows(self) -> int: ...

class Metrics:
    __members__: ClassVar[dict] = ...  # read-only
    AOE: ClassVar[Metrics] = ...
    AOECV: ClassVar[Metrics] = ...
    ARMSE: ClassVar[Metrics] = ...
    ARMSECV: ClassVar[Metrics] = ...
    EFIOE: ClassVar[Metrics] = ...
    EFIOECV: ClassVar[Metrics] = ...
    EMAX: ClassVar[Metrics] = ...
    EMAXCV: ClassVar[Metrics] = ...
    LINV: ClassVar[Metrics] = ...
    OE: ClassVar[Metrics] = ...
    OECV: ClassVar[Metrics] = ...
    RMSE: ClassVar[Metrics] = ...
    RMSECV: ClassVar[Metrics] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Models:
    __members__: ClassVar[dict] = ...  # read-only
    CN: ClassVar[Models] = ...
    DYNATREE: ClassVar[Models] = ...
    ENSEMBLE: ClassVar[Models] = ...
    KRIGING: ClassVar[Models] = ...
    KS: ClassVar[Models] = ...
    LINEAR: ClassVar[Models] = ...
    LOWESS: ClassVar[Models] = ...
    PRS: ClassVar[Models] = ...
    PRS_CAT: ClassVar[Models] = ...
    PRS_EDGE: ClassVar[Models] = ...
    RBF: ClassVar[Models] = ...
    SVN: ClassVar[Models] = ...
    TGP: ClassVar[Models] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class NormType:
    __members__: ClassVar[dict] = ...  # read-only
    NORM_0: ClassVar[NormType] = ...
    NORM_1: ClassVar[NormType] = ...
    NORM_2: ClassVar[NormType] = ...
    NORM_INF: ClassVar[NormType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Surrogate:
    def __init__(self, *args, **kwargs) -> None: ...
    def build(self) -> bool: ...
    def get_Sh(self) -> numpy.ndarray[numpy.float64]: ...
    def get_Sv(self) -> numpy.ndarray[numpy.float64]: ...
    def get_Zh(self) -> numpy.ndarray[numpy.float64]: ...
    def get_Zv(self) -> numpy.ndarray[numpy.float64]: ...
    def get_in_dim(self) -> int: ...
    @overload
    def get_metric(self, arg0: Metrics) -> numpy.ndarray[numpy.float64]: ...
    @overload
    def get_metric(self, arg0: Metrics, arg1: int) -> float: ...
    @overload
    def get_metric(self, mt: Metrics, j: int) -> float: ...
    def get_out_dim(self) -> int: ...
    def get_param(self) -> Surrogate_Parameters: ...
    def optimize_parameters(self) -> bool: ...
    def predict(self, arg0: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]: ...

class Surrogate_Ensemble(Surrogate):
    def __init__(self, *args, **kwargs) -> None: ...
    def model_list_display(self) -> str: ...

class Surrogate_Parameters:
    @overload
    def __init__(self, arg0: Models) -> None: ...
    @overload
    def __init__(self, arg0: str) -> None: ...
    def get_budget(self) -> int: ...
    def get_covariance_coef(self) -> numpy.ndarray[numpy.float64]: ...
    def get_degree(self) -> int: ...
    def get_distance_type(self) -> DistanceType: ...
    def get_kernel_coef(self) -> float: ...
    def get_kernel_type(self) -> KernelType: ...
    def get_metric_type(self) -> Metrics: ...
    def get_metric_type_str(self) -> str: ...
    def get_output(self) -> str: ...
    def get_preset(self) -> str: ...
    def get_ridge(self) -> float: ...
    def get_type(self) -> Models: ...
    def get_weight(self) -> numpy.ndarray[numpy.float64]: ...
    def get_weight_type(self) -> WeightType: ...

class TrainingSet:
    def __init__(self, arg0: numpy.ndarray[numpy.float64], arg1: numpy.ndarray[numpy.float64]) -> None: ...
    def display(self) -> str: ...

class WeightType:
    __members__: ClassVar[dict] = ...  # read-only
    EXTERN: ClassVar[WeightType] = ...
    OPTIM: ClassVar[WeightType] = ...
    SELECT: ClassVar[WeightType] = ...
    WTA1: ClassVar[WeightType] = ...
    WTA3: ClassVar[WeightType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

@overload
def Surrogate_Factory(arg0: TrainingSet, arg1: str) -> Surrogate: ...
@overload
def Surrogate_Factory(arg0: Matrix, arg1: Matrix, arg2: str) -> Surrogate: ...
@overload
def Surrogate_Factory(arg0: TrainingSet, arg1: Dict[str,Union[str,Models,WeightType,KernelType,DistanceType,Metrics,float,int]]) -> Surrogate: ...
def metric_type_to_norm_type(*args, **kwargs) -> Any: ...
def metric_type_to_str(arg0) -> str: ...
def str_to_metric_type(*args, **kwargs) -> Any: ...
