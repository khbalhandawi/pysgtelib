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

// py::array_t<double> predict_wrapper(SGTELIB::Surrogate *s, const SGTELIB::Matrix &XX) {  
//     SGTELIB::Matrix ZZ("ZZ", s->get_in_dim(), s->get_out_dim());  
//     s->predict(XX, &ZZ);  
  
//     // Create a new NumPy array  
//     py::array_t<double> result({ZZ.get_nb_rows(), ZZ.get_nb_cols()});  
  
//     // Get a mutable view to the data  
//     py::buffer_info buf_info = result.request();  
//     double *ptr = static_cast<double *>(buf_info.ptr);  
  
//     // Copy the data from the ZZ matrix to the NumPy array  
//     for (size_t i = 0; i < ZZ.get_nb_rows(); ++i) {  
//         for (size_t j = 0; j < ZZ.get_nb_cols(); ++j) {  
//             ptr[i * ZZ.get_nb_cols() + j] = ZZ.get(i, j);  
//         }  
//     }  
      
//     return result;  
// }  

PYBIND11_MODULE(pysgtelib , m) {  
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
        .def(py::init<const SGTELIB::Matrix &, const SGTELIB::Matrix &>())
        .def("display", &TrainingSet_display_wrapper);

    py::enum_<SGTELIB::model_t>(m, "model_t")  
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
        .def(py::init<const std::string &>());  

    // this is an abstract class, cannot bind constructors
    py::class_<SGTELIB::Surrogate>(m, "Surrogate")
        .def("build", &SGTELIB::Surrogate::build)
        .def("predict", [](SGTELIB::Surrogate &self, const SGTELIB::Matrix &XX) {  
            SGTELIB::Matrix ZZ("ZZ", self.get_in_dim(), self.get_out_dim());  
            self.predict(XX, &ZZ);  
    
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
        }, "A wrapper for the Surrogate predict method that returns a NumPy array");  

    m.def("Surrogate_Factory", (SGTELIB::Surrogate* (*)(SGTELIB::TrainingSet&, const std::string&)) &Surrogate_Factory, py::return_value_policy::reference);  
    m.def("Surrogate_Factory", (SGTELIB::Surrogate* (*)(SGTELIB::Matrix&, SGTELIB::Matrix&, const std::string&)) &Surrogate_Factory, py::return_value_policy::reference);  
}