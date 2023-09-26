#-------------------------------------------------------------------------------
#	CMAKE SETTINGS
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8.12)

#-------------------------------------------------------------------------------
#	SOURCE CODE AND USER SETTINGS
#-------------------------------------------------------------------------------

add_subdirectory(pybind11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  

add_subdirectory(sgtelib)

set(PYBIND11_INCLUDE_DIRS pybind11/include)
set(SGTELIB_INCLUDE_DIRS sgtelib/include)

#-------------------------------------------------------------------------------
#	PROJECT SETTINGS
#-------------------------------------------------------------------------------

set(PROJECT_NAME "sgtelib") 
cmake_policy(SET CMP0048 NEW)
project(${PROJECT_NAME} VERSION 0.1.0)

set(TARGET_NAME "pysgtelib")

#-------------------------------------------------------------------------------
#	COMPILE FLAGS
#-------------------------------------------------------------------------------

# Common compiler flags for all build types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fPIC")

# Debug build flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
# Release build flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
	set(GUI_TYPE WIN32)
	add_compile_options(/MP)
elseif(CMAKE_COMPILER_IS_GNUCXX)
	add_compile_options(-std=c++17)
endif(MSVC)

#-------------------------------------------------------------------------------
#	ADD MODULES
#-------------------------------------------------------------------------------

# Set the path to the Anaconda Python interpreter
set(Python_ROOT_DIR $ENV{PYTHON_PATH})
set(Python_FIND_STRATEGY LOCATION)

# Now find the Python package
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

message(Python_INCLUDE_DIRS="${Python_INCLUDE_DIRS}")
message(Python_EXECUTABLE="${Python_EXECUTABLE}")
message(Python_LIBRARY="${Python_LIBRARY}")

#-------------------------------------------------------------------------------
#	GENERATE EXECUTABLE
#-------------------------------------------------------------------------------

pybind11_add_module(${TARGET_NAME} src/pywrap.cpp)
target_include_directories(${TARGET_NAME} PRIVATE
    ${Python_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIRS}
)
target_compile_definitions(${TARGET_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(${TARGET_NAME} PRIVATE sgtelib Python::Python ${LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)