from setuptools import setup, find_packages, Extension
from setuptools._distutils.dist import Distribution
from setuptools.command.build_ext import build_ext
import subprocess
import os
import shutil
import re
import glob

def check_for_shared_objects(build_temp):
    files_in_build_dir = os.listdir(build_temp)

    # Regular expressions for matching pysgtelib*.so and libsgtelib.so
    pysgtelib_found = any(re.match(r'pysgtelib\..*\.so$', fname) for fname in files_in_build_dir)
    libsgtelib_found = any(re.match(r'libsgtelib\.so$', fname) for fname in files_in_build_dir)

    return pysgtelib_found and libsgtelib_found

def check_cmake_version():
    try:
        # Run the command and capture the output
        result = subprocess.run(['cmake', '--version'], check=True, capture_output=True, text=True)
        output = result.stdout

        # Use regular expression to extract the version number
        version_match = re.search(r'version\s+(\d+\.\d+\.\d+)', output)
        if version_match:
            # Extract the version number and convert it into a tuple of integers
            cmake_version = tuple(map(int, version_match.group(1).split('.')))

            # Define the minimum required version
            required_version = (3, 24, 0)

            # Compare the installed CMake version with the required version
            if cmake_version <= required_version:
                raise RuntimeError(f"CMake version must be above {'.'.join(map(str, required_version))}. Found version: {'.'.join(map(str, cmake_version))}")
            else:
                print(f"Found CMake version: {'.'.join(map(str, cmake_version))}")
        else:
            raise RuntimeError("Failed to parse CMake version.")
    except subprocess.CalledProcessError:
        raise RuntimeError("CMake must be installed to build the following extensions.")
    except OSError:
        raise RuntimeError("CMake must be installed to build the following extensions.")

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):

    def __init__(self, dist: Distribution) -> None:
        super().__init__(dist)

        self.build_temp = os.path.join("build")
        if os.path.exists(self.build_temp):
            found_so = check_for_shared_objects(self.build_temp)
            if found_so:
                print("Found .so files in build, skipping build.")
                self.build_complete = True
                
            else:
                self.build_complete = False
        else:
            self.build_complete = False
            os.mkdir(self.build_temp)

    def run(self):

        if not self.build_complete:
            check_cmake_version()
            for ext in self.extensions:
                self.build_extension(ext)

        for ext in self.extensions:
            # create __init__.py file
            with open(os.path.join(ext.sourcedir, ext.name, "__init__.py"), "w") as f:
                f.write("")
            try:
                stubgen_command = ['stubgen', '-m', 'pysgtelib', '-o', '.']
                subprocess.run(stubgen_command, cwd=os.path.join(ext.sourcedir, ext.name), check=True)
                print("Stub generation successful.")
            except subprocess.CalledProcessError as e:
                print("Stub generation failed:", e)

            # Copy the .so files to the correct place for the wheel
            so_files = glob.glob(os.path.join(ext.sourcedir, ext.name, "*.so"))
            for so_file in so_files:
                target_path = os.path.join(self.build_lib, ext.name, os.path.basename(so_file))
                print(f"Copying {so_file} to {target_path}")
                shutil.copy(so_file, target_path)

            print("========================================")
            print("Extension contents")
            for file in os.listdir(os.path.join(ext.sourcedir, ext.name)):
                print(file)
            print("========================================")

    def build_extension(self, ext):
        ## User has to set "Debug" or "Release" here ##
        cfg = "Release"

        cmake_command1 = ['cmake', '-S', ext.sourcedir, '-B', self.build_temp, '-DCMAKE_BUILD_TYPE=' + cfg]
        cmake_command2 = ['cmake', '--build', self.build_temp, '--config', cfg, '--', '-j4']
        cmake_command3 = ['cmake', '--install', self.build_temp, '--config', cfg]
        
        subprocess.run(cmake_command1, check=True)
        subprocess.run(cmake_command2, check=True)
        subprocess.run(cmake_command3, check=True)

setup(
    name='pysgtelib',
    version='0.1.0',
    author='Khalil Al Handawi',
    author_email='sebastian.hampl.ext@siemens-energy.com',
    description='Machine Learning library SGTELIB implementation in Python.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where=".", exclude=["tests", "tests.*"]),
    ext_modules=[CMakeExtension('pysgtelib', sourcedir=".")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=['pybind11>=2.5.0', 'numpy>=1.21.0'],
    setup_requires=['pybind11>=2.5.0'],
    include_package_data=True,  # Ensure package data is included
    package_data={
        'pysgtelib': ['*.so', '*.pyi'], # Include any *.so and *.pyi files found within the package directories
    },
)