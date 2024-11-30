# pySGTELIB a python wrapper for SGTELIB

This project provides a python programming interface for SGTELIB using [`pybind11`](https://pybind11.readthedocs.io/en/stable/index.html), so that the `C++` methods and classes defined in [`SGTELIB`](SGTELIB) are usable in python.

## Quickstart: Building with CMake

The project uses a [CMakeLists.txt](CMakeLists.txt) file to define the build project and compile the `SGTELIB` library. You need to compile it successfully so that the python wrapper works

### Prerequisites

To complete this tutorial, you'll need:

*   A compatible operating system (e.g. Linux, macOS, Windows).
*   A compatible C++ compiler that supports at least C++14.
*   [CMake](https://cmake.org/) and a compatible build tool for building the
    project.
    *   Compatible build tools include
        [Make](https://www.gnu.org/software/make/),
        [Ninja](https://ninja-build.org/), and others - see
        [CMake Generators](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html)
        for more information.
* A [Python](https://www.python.org/) distribution to run the scripts and compile the `pybind11` library `.so` file which will be imported. Recommended distributions:
    * [Anaconda](https://www.anaconda.com/)
* A [PDM](https://pdm-project.org/latest/) installation for python environment management

If you don't already have CMake installed, see the
[CMake installation guide](https://cmake.org/install).

Note: The terminal commands in this tutorial show a Unix shell prompt and this has not been tested on Windows.

### Building the project

First clone the repository:

```bash
git clone --recurse-submodules https://github.com/khbalhandawi/pysgtelib.git
cd pysgtelib
```

Now, in `root` create a virtual environment

```bash
pdm install
```

Next, build the module using

```bash
mkdir build
cd build
cmake -S .. -DCMAKE_BUILD_TYPE=Release
cmake --build Release --config Release
cmake --install Release --config Release
```

This should create a `pysgtelib.cpython-39-x86_64-linux-gnu.so` file in your `build` directory. The file name will vary based on your python version and linux distribution.

**Optional**: Generate stub file `.pyi` to help you with intellisense and be able to understand how to use `pysgtelib`

```bash
stubgen -m pysgtelib -o .
```

You may run your python scripts now. For example [test.py](test.py)