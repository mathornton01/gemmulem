# GeMMulEM (\[Ge\]neral \[M\]ixed \[Mul\]tinomial \[E\]xpectation \[M\]aximization)

This version of the gemmulem program is intended to be very easy to install and use, and therefore only
requires the GNU make and the CMake utility for its installation. 

To compile this utility from its sources the only tools you will need in addition to make is gcc. 
This utility can be compiled in linux, mac and Windows environment.


# Getting Started

## Prerequisites

- Linux/Mac environment  
  C/C++ compiler(gcc, clang)  
  CMake, make  

- Windows environment  
  Visual Studio 2017 or newer version  
  CMake, make  
  or, Windows subsystem for linux

## Installation

### Linux/MAC environment
1. Clone the GEMMULEM repo
```bash
git clone https://github.com/mathornton01/gemmulem.git
```

2. Create a build directory in the project and change to it, and run cmake
```bash
cd gemmulem
mkdir build
cd build
cmake ../
```

3. Run make command
```bash
make
```

4. Install GEMMULEM application and library
```bash
sudo make install
```

5. (Optional) Build and install R package  
Change to extension/R directory in the project.
```bash
cd extension/R
R CMD INSTALL rgemmulem
```

6. (Optional) Build and install python package  
Change to extension/python directory in the project.
```bash
TODO
```


### Windows environment (Using MSVC)
1. Clone the GEMMULEM repo
```bash
git clone https://github.com/mathornton01/gemmulem.git
```

2. Create a build directory in the project and change to it, and run cmake
```bash
cd gemmulem
mkdir build
cd build
cmake ../
```

3. Run build command
```bash
cmake --build . --config Release
```

4. Install GEMMULEM application and library. Run console as administrator
```bash
cmake --install . --config Release
```
GEMMULEM will be installed under "C:\Program Files (x86)\GEMMULEM" directory by default. If you want to install different directory, please use `--prefix` option
```bash
cmake --install . --config Release --prefix "D:\OTHER\FOLDER"
```

5. (Optional) Build and install R package  
Change to extension/R directory in the project.
```bash
TODO
```

6. (Optional) Build and install python package  
Change to extension/python directory in the project.
```bash
TODO
```

