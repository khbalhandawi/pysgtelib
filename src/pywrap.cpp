#include "sgtelib.hpp"
#include "Surrogate_Factory.hpp"
#include "Surrogate_Utils.hpp"
#include <fstream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  

using namespace SGTELIB;

namespace py = pybind11;

constexpr auto byref = py::return_value_policy::reference_internal;

std::string TrainingSet_display_wrapper(const TrainingSet& ts) {  
    std::ostringstream oss;  
    ts.display(oss);  
    return oss.str();  
}  

// Function to convert a NumPy array to SGTELIB::Matrix  
SGTELIB::Matrix convertNumpyToMatrix(py::array_t<double> arr, const std::string& name) {  
    auto r = arr.unchecked<2>();  // Assumes arr is a 2D array  
    int nbRows = r.shape(0);  
    int nbCols = r.shape(1);  
  
    double **p = new double*[nbRows];  
    for(int i = 0; i < nbRows; ++i) {  
        p[i] = new double[nbCols];  
        for(int j = 0; j < nbCols; ++j) {  
            p[i][j] = r(i, j);  
        }  
    }  
  
    // Create SGTELIB::Matrix with the given name, number of rows and columns, and data  
    return SGTELIB::Matrix(name, nbRows, nbCols, p);  
}

// Function to convert SGTELIB::Matrix to a NumPy array  
py::array_t<double> convertMatrixToNumpy(const SGTELIB::Matrix &ZZ) {  
    // Create a new NumPy array    
    py::array_t<double> result({ZZ.get_nb_rows(), ZZ.get_nb_cols()});    
  
    // Get a mutable view to the data    
    py::buffer_info buf_info = result.request();    
    double *ptr = static_cast<double *>(buf_info.ptr);    
  
    // Copy the data from the ZZ matrix to the NumPy array    
    for (size_t i = 0; i < ZZ.get_nb_rows(); ++i) {    
        for (size_t j = 0; j < ZZ.get_nb_cols(); ++j) {    
            ptr[i * ZZ.get_nb_cols() + j] = ZZ.get(i, j);  // Assumes that you have a get method to access elements    
        }    
    }    
    return result;  
}

PYBIND11_MODULE(pysgtelib , m) {  

    m.def("metric_type_to_str", &SGTELIB::metric_type_to_str, "Convert metric type to string");  
    m.def("metric_type_to_norm_type", &SGTELIB::metric_type_to_norm_type, "Convert metric type to norm type");  
    m.def("str_to_metric_type", &SGTELIB::str_to_metric_type, "Convert string to metric type");  

    py::class_<Matrix>(m, "Matrix")  
        .def(py::init([](const std::string &name, int nbRows, int nbCols, py::array_t<double> A) {  
            auto r = A.unchecked<2>();  // We assume A is a 2D array here  
            double **p = new double*[nbRows];  
            for(int i=0; i<nbRows; ++i) {  
                p[i] = new double[nbCols];  
                for(int j=0; j<nbCols; ++j) {  
                    p[i][j] = r(i, j);  
                }  
            }  
            return new Matrix(name, nbRows, nbCols, p);  
        }))
        // other class methods and properties can be defined here  
        .def("get_nb_cols", &Matrix::get_nb_cols)
        .def("get_nb_rows", &Matrix::get_nb_rows)
        .def("get", (double (Matrix::*)(int, int) const) &Matrix::get, py::arg("i"), py::arg("j"));

    py::class_<TrainingSet>(m, "TrainingSet")    
        .def(py::init([](py::array_t<double> X, py::array_t<double> Z) {  
            return new TrainingSet(convertNumpyToMatrix(X, "X"), convertNumpyToMatrix(Z, "Z"));  
        }))  
        .def("display", &TrainingSet_display_wrapper);

    py::enum_<SGTELIB::norm_t>(m, "NormType")  
        .value("NORM_0", SGTELIB::norm_t::NORM_0)  
        .value("NORM_1", SGTELIB::norm_t::NORM_1)  
        .value("NORM_2", SGTELIB::norm_t::NORM_2)  
        .value("NORM_INF", SGTELIB::norm_t::NORM_INF)  
        .export_values();  

    py::enum_<SGTELIB::distance_t>(m, "DistanceType")  
        .value("NORM2", SGTELIB::DISTANCE_NORM2)  
        .value("NORM1", SGTELIB::DISTANCE_NORM1)  
        .value("NORMINF", SGTELIB::DISTANCE_NORMINF)  
        .value("NORM2_IS0", SGTELIB::DISTANCE_NORM2_IS0)  
        .value("NORM2_CAT", SGTELIB::DISTANCE_NORM2_CAT)  
        .export_values();  

    py::enum_<SGTELIB::metric_t>(m, "Metrics")  
        .value("EMAX", SGTELIB::metric_t::METRIC_EMAX)  
        .value("EMAXCV", SGTELIB::metric_t::METRIC_EMAXCV)  
        .value("RMSE", SGTELIB::metric_t::METRIC_RMSE)  
        .value("ARMSE", SGTELIB::metric_t::METRIC_ARMSE)  
        .value("RMSECV", SGTELIB::metric_t::METRIC_RMSECV)  
        .value("ARMSECV", SGTELIB::metric_t::METRIC_ARMSECV)  
        .value("OE", SGTELIB::metric_t::METRIC_OE)  
        .value("OECV", SGTELIB::metric_t::METRIC_OECV)  
        .value("AOE", SGTELIB::metric_t::METRIC_AOE)  
        .value("AOECV", SGTELIB::metric_t::METRIC_AOECV)  
        .value("EFIOE", SGTELIB::metric_t::METRIC_EFIOE)  
        .value("EFIOECV", SGTELIB::metric_t::METRIC_EFIOECV)  
        .value("LINV", SGTELIB::metric_t::METRIC_LINV)  
        .export_values();  

    py::enum_<SGTELIB::kernel_t>(m, "KernelType")  
        .value("D1", SGTELIB::kernel_t::KERNEL_D1)  
        .value("D2", SGTELIB::kernel_t::KERNEL_D2)  
        .value("D3", SGTELIB::kernel_t::KERNEL_D3)  
        .value("D4", SGTELIB::kernel_t::KERNEL_D4)  
        .value("D5", SGTELIB::kernel_t::KERNEL_D5)  
        .value("D6", SGTELIB::kernel_t::KERNEL_D6)  
        .value("D7", SGTELIB::kernel_t::KERNEL_D7)  
        .value("I0", SGTELIB::kernel_t::KERNEL_I0)  
        .value("I1", SGTELIB::kernel_t::KERNEL_I1)  
        .value("I2", SGTELIB::kernel_t::KERNEL_I2)  
        .value("I3", SGTELIB::kernel_t::KERNEL_I3)  
        .value("I4", SGTELIB::kernel_t::KERNEL_I4)  
        .export_values();  

    py::enum_<SGTELIB::weight_t>(m, "WeightType")  
        .value("SELECT", SGTELIB::weight_t::WEIGHT_SELECT)  
        .value("OPTIM", SGTELIB::weight_t::WEIGHT_OPTIM)  
        .value("WTA1", SGTELIB::weight_t::WEIGHT_WTA1)  
        .value("WTA3", SGTELIB::weight_t::WEIGHT_WTA3)  
        .value("EXTERN", SGTELIB::weight_t::WEIGHT_EXTERN)  
        .export_values();  

    py::enum_<SGTELIB::model_t>(m, "Models")  
        .value("LINEAR", SGTELIB::model_t::LINEAR)  
        .value("TGP", SGTELIB::model_t::TGP)  
        .value("DYNATREE", SGTELIB::model_t::DYNATREE)  
        .value("PRS", SGTELIB::model_t::PRS)  
        .value("PRS_EDGE", SGTELIB::model_t::PRS_EDGE)  
        .value("PRS_CAT", SGTELIB::model_t::PRS_CAT)  
        .value("KS", SGTELIB::model_t::KS)  
        .value("CN", SGTELIB::model_t::CN)  
        .value("KRIGING", SGTELIB::model_t::KRIGING)  
        .value("SVN", SGTELIB::model_t::SVN)  
        .value("RBF", SGTELIB::model_t::RBF)  
        .value("LOWESS", SGTELIB::model_t::LOWESS)  
        .value("ENSEMBLE", SGTELIB::model_t::ENSEMBLE)  
        .export_values(); 

    py::class_<SGTELIB::Surrogate_Parameters>(m, "Surrogate_Parameters")  
        .def(py::init<const model_t &>())  
        .def(py::init<const std::string &>())
        .def(py::init<const std::map<std::string, SGTELIB::ParameterTypes> &>())  
        .def("get_nb_parameter_optimization", &SGTELIB::Surrogate_Parameters::get_nb_parameter_optimization, "A function that returns the number of parameters to optimize")
        .def("get_type", &SGTELIB::Surrogate_Parameters::get_type)  
        .def("get_degree", &SGTELIB::Surrogate_Parameters::get_degree)  
        .def("get_kernel_type", &SGTELIB::Surrogate_Parameters::get_kernel_type)  
        .def("get_kernel_coef", &SGTELIB::Surrogate_Parameters::get_kernel_coef)  
        .def("get_ridge", &SGTELIB::Surrogate_Parameters::get_ridge)  
        .def("get_weight", [](const SGTELIB::Surrogate_Parameters &self) {    
            return convertMatrixToNumpy(self.get_weight());   
        })  
        .def("set_weight", [](SGTELIB::Surrogate_Parameters &self, py::array_t<double> input) {  
            SGTELIB::Matrix matrix = convertNumpyToMatrix(input,"W");  
            self.set_weight(matrix);  
        }, "A function that sets the weight matrix")
        .def("get_weight_type", &SGTELIB::Surrogate_Parameters::get_weight_type)  
        .def("get_metric_type", &SGTELIB::Surrogate_Parameters::get_metric_type)  
        .def("get_metric_type_str", &SGTELIB::Surrogate_Parameters::get_metric_type_str)  
        .def("get_distance_type", &SGTELIB::Surrogate_Parameters::get_distance_type)  
        .def("get_preset", &SGTELIB::Surrogate_Parameters::get_preset)  
        .def("get_output", &SGTELIB::Surrogate_Parameters::get_output)  
        .def("get_covariance_coef", [](const SGTELIB::Surrogate_Parameters &self) {    
            return convertMatrixToNumpy(self.get_covariance_coef());   
        })  
        .def("get_budget", &SGTELIB::Surrogate_Parameters::get_budget)
        .def("display_x", [](SGTELIB::Surrogate_Parameters &self) {     
            std::ostringstream oss;   
            self.display_x(oss);  
            return oss.str(); 
        }, "A wrapper for displaying optimal model parameters");


    // this is an abstract class, cannot bind constructors
    py::class_<SGTELIB::Surrogate>(m, "Surrogate")
        .def("build", &SGTELIB::Surrogate::build, py::arg("optimize") = false)
        .def("predict", [](SGTELIB::Surrogate &self, py::array_t<double> XX_arr) {    
            SGTELIB::Matrix XX = convertNumpyToMatrix(XX_arr, "XX");  
            SGTELIB::Matrix ZZ("ZZ", self.get_in_dim(), self.get_out_dim());    
            self.predict(XX, &ZZ);      
            return convertMatrixToNumpy(ZZ);   
        }, "A wrapper for the Surrogate predict method that returns a NumPy array")
        .def("get_metric", [](SGTELIB::Surrogate &self, SGTELIB::metric_t mt) {  
            SGTELIB::Matrix E = self.get_metric(mt);    
            return convertMatrixToNumpy(E); 
        }, "A wrapper for the Surrogate get_metric method that returns a NumPy array")
        .def("get_metric", [](SGTELIB::Surrogate &self, SGTELIB::metric_t mt, const int j) {  
            double e = self.get_metric(mt,j);    
            return e;
        }, "A wrapper for the Surrogate get_metric method that returns a NumPy array")
        .def("get_metric", (double (SGTELIB::Surrogate::*)(SGTELIB::metric_t, int)) &SGTELIB::Surrogate::get_metric, py::arg("mt"), py::arg("j"), "A wrapper for the Surrogate get_metric method that returns a double")
        .def("get_Sh", [](SGTELIB::Surrogate &self) {  
            SGTELIB::Matrix Sh = self.get_matrix_Sh();    
            return convertMatrixToNumpy(Sh); 
        }, "A wrapper for the Surrogate get_matrix_Sh method that returns a NumPy array")
        .def("get_Zh", [](SGTELIB::Surrogate &self) {  
            SGTELIB::Matrix Zh = self.get_matrix_Zh();    
            return convertMatrixToNumpy(Zh); 
        }, "A wrapper for the Surrogate get_matrix_Zh method that returns a NumPy array")
        .def("get_Zv", [](SGTELIB::Surrogate &self) {  
            SGTELIB::Matrix Zv = self.get_matrix_Zv();    
            return convertMatrixToNumpy(Zv); 
        }, "A wrapper for the Surrogate get_matrix_Zv method that returns a NumPy array")
        .def("get_Sv", [](SGTELIB::Surrogate &self) {  
            SGTELIB::Matrix Sv = self.get_matrix_Sv();    
            return convertMatrixToNumpy(Sv); 
        }, "A wrapper for the Surrogate get_matrix_Sv method that returns a NumPy array")
        .def("optimize_parameters", &SGTELIB::Surrogate::optimize_parameters, 
        "A wrapper for the Surrogate optimize_parameters method that returns a bool")
        .def("get_param", &SGTELIB::Surrogate::get_param, 
        "A wrapper for the Surrogate get_param method that returns Surrogate_Parameters")
        .def("get_in_dim", &SGTELIB::Surrogate::get_in_dim, 
        "A wrapper for the Surrogate get_in_dim method that returns Surrogate_Parameters")
        .def("get_out_dim", &SGTELIB::Surrogate::get_out_dim, 
        "A wrapper for the Surrogate get_out_dim method that returns Surrogate_Parameters");
        
    // this is an abstract class, cannot bind constructors
    py::class_<SGTELIB::Surrogate_Ensemble, SGTELIB::Surrogate>(m, "Surrogate_Ensemble")  
        .def("model_list_display", [](SGTELIB::Surrogate_Ensemble &self) {     
            std::ostringstream oss;   
            self.model_list_display(oss);  
            return oss.str();  
        }, "A wrapper for model_list_display")  
        .def("model_list_preset", &SGTELIB::Surrogate_Ensemble::model_list_preset, "A method to set a preset for the model list")  
        .def("model_list_remove_all", &SGTELIB::Surrogate_Ensemble::model_list_remove_all, "A method to remove all models from the list")  
        .def("model_list_add", (void (SGTELIB::Surrogate_Ensemble::*)(const std::string &)) &SGTELIB::Surrogate_Ensemble::model_list_add, 
            "A method to add a model to the list with a string definition")  
        .def("model_list_add", (void (SGTELIB::Surrogate_Ensemble::*)(const std::map<std::string,SGTELIB::ParameterTypes>)) &SGTELIB::Surrogate_Ensemble::model_list_add, 
            "A method to add a model to the list with a map definition")
        .def("model_list_add", (void (SGTELIB::Surrogate_Ensemble::*)(SGTELIB::Surrogate *)) &SGTELIB::Surrogate_Ensemble::model_list_add, 
            "A method to add a model to the list with a Surrogate pointer")
        .def("set_weight_vector", [](SGTELIB::Surrogate_Ensemble &self, py::array_t<double> input) {  
            SGTELIB::Matrix matrix = convertNumpyToMatrix(input,"W");  
            self.set_weight_vector(matrix);  
        }, "A function that sets the weight matrix from an external numpy array");

    m.def("Surrogate_Factory", (SGTELIB::Surrogate* (*)(SGTELIB::TrainingSet&, const std::string&)) &Surrogate_Factory, py::return_value_policy::reference);  
    m.def("Surrogate_Factory", (SGTELIB::Surrogate* (*)(SGTELIB::Matrix&, SGTELIB::Matrix&, const std::string&)) &Surrogate_Factory, py::return_value_policy::reference);
    m.def("Surrogate_Factory", (SGTELIB::Surrogate* (*)(SGTELIB::TrainingSet&, const std::map<std::string,SGTELIB::ParameterTypes>&)) &Surrogate_Factory, py::return_value_policy::reference);
    m.def("Surrogate_Factory", (SGTELIB::Surrogate* (*)(SGTELIB::TrainingSet&, SGTELIB::Surrogate_Parameters&)) &Surrogate_Factory, py::return_value_policy::reference);
    m.def("set_seed", &SGTELIB::set_seed, py::arg("seed"), "A function to set the seed for SGTELIB");

}