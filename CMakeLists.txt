#-------------------------------------------------------------------------------
#	CMAKE SETTINGS
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.5...3.20)

#-------------------------------------------------------------------------------
#	SOURCE CODE AND USER SETTINGS
#-------------------------------------------------------------------------------

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(sgtelib)

set(PYBIND11_INCLUDE_DIRS pybind11/include)
set(SGTELIB_INCLUDE_DIRS sgtelib/include)

#-------------------------------------------------------------------------------
#	PROJECT SETTINGS
#-------------------------------------------------------------------------------

set(PROJECT_NAME "pysgtelib")
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
	set(GUI_TYPE WIN32)
	add_compile_options(/MP)
elseif(CMAKE_COMPILER_IS_GNUCXX)
	add_compile_options(-std=c++17)
endif(MSVC)

#-------------------------------------------------------------------------------
#	ADD Python MODULES
#-------------------------------------------------------------------------------
include(FetchContent)
message(STATUS "Fetching pybind11 from: https://github.com/pybind/pybind11.git")
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.13.4
	DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(pybind11)

# Ensure Python and NumPy are found
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

# Now find the Python package
message(Python_INCLUDE_DIRS="${Python_INCLUDE_DIRS}")
message(Python_EXECUTABLE="${Python_EXECUTABLE}")
message(Python_LIBRARY="${Python_LIBRARY}")

#-------------------------------------------------------------------------------
#	GENERATE EXECUTABLE
#-------------------------------------------------------------------------------

pybind11_add_module(${TARGET_NAME} src/pywrap.cpp)
target_compile_definitions(${TARGET_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(${TARGET_NAME} PRIVATE sgtelib Python3::Python Python3::Module)

#-------------------------------------------------------------------------------
#	INSTALLATION RULES
#-------------------------------------------------------------------------------

# Define output directories based on the build type
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin)

# Installation rules
set(INSTALL_DIRECTORY "${CMAKE_SOURCE_DIR}/pysgtelib")
install(TARGETS ${TARGET_NAME} sgtelib
        LIBRARY DESTINATION ${INSTALL_DIRECTORY}
        RUNTIME DESTINATION ${INSTALL_DIRECTORY})

# Set the RPATH for pysgtelib to ensure it can find libsgtelib.so in the same directory
set_target_properties(pysgtelib PROPERTIES
	INSTALL_RPATH "$ORIGIN"  # $ORIGIN refers to the directory where pysgtelib.so is located
	BUILD_WITH_INSTALL_RPATH TRUE
)

# Package configuration
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)