<div id="top"></div>
# GeMMulEM (\[Ge\]neral \[M\]ixed \[Mul\]tinomial \[E\]xpectation \[M\]aximization)

This version of the gemmulem program is intended to be very easy to install and use, and therefore only
requires the GNU make and the CMake utility for its installation. 

To compile this utility from its sources the only tools you will need in addition to make is gcc. 
This utility can be compiled in linux, mac and Windows environment.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <ul>
          <li><a href="#linuxmac-environment">Linux/Mac environment</a></li>
          <li><a href="#windows-environment">Windows environment</a></li>
        </ul>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <ul>
      <li><a href="#gemmulem">gemmulem application</a></li>
      <li><a href="#cc-apis">C/C++ APIs</a></li>
      <li><a href="#r-apis">R APIs</a></li>
      <li><a href="#python-apis">Python APIs</a></li>
    </ul>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

# Getting Started

## Prerequisites

### Linux/Mac environment  
  C/C++ compiler(gcc, clang)  
  CMake  

### Windows environment  
  Visual Studio Community 2017 or newer version (with `Desktop development with C++` workload)  
  CMake  
  or, Windows subsystem for linux  
  Rtools(for building R package)  
  python setuptools, build package(for building python package)

<p align="right">(<a href="#top">back to top</a>)</p>

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

3. Build build command
```bash
cmake --build . --config Release
```

4. Install GEMMULEM application and library
```bash
sudo cmake --install . --config Release
```
`GEMMULEM` will be installed in `/usr/local/lib` directory by default. Use `--prefix` option to install a different directory.
```bash
cmake --install . --config Release --prefix=/home/user/other
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
cd extension/python
pip install --upgrade build
./build.sh
```

<p align="right">(<a href="#top">back to top</a>)</p>


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
GEMMULEM will be installed in `C:\Program Files (x86)\GEMMULEM` directory by default. Use `--prefix` option to install a different directory.
```bash
cmake --install . --config Release --prefix "D:\OTHER\FOLDER"
```

5. (Optional) Build and install R package  
Change to extension/R directory in the project. We assume `R.exe` is in the `PATH` environment.
```bash
cd extension/R
R CMD INSTALL rgemmulem
```

6. (Optional) Build and install python package  
Change to extension/python directory in the project.
```bash
cd extension/python
pip install --upgrade build
build.bat
```

<p align="right">(<a href="#top">back to top</a>)</p>

# Usage  
## gemmulem
```
gemmulem [options]
```
### options
- `-i/-I/--IFILE <input filename>`  
Specify input file for coarse multinomial mode. The file contains compatibility patterns and counts. Each columns are seperated by comma(,). First column is a compatiblity pattern and second is a count.  
_Both -g/e and -i may not be simultaneously specified_  
Example file with 4 category class  

  > `1011,40`  
  > `1100,45`  
  > `0010,30`  

- `-g/-G/--GFILE <input filename>`  
Specify input file for Gaussian mixture deconvolution mode. The file contains a list of values comming from a mixture of univariate gaussians.  
_Both -g/e and -i may not be simultaneously specified_  

  > `-2.64566122335398`  
  > `-24.9939621808671`  
  > `-5.25417008181354`  
  > `-27.4150444391751`  
  > `-2.23999093704485`  
  > `-0.644893131269661`  
  > `-16.9420712792649`  
  > `-17.359643633111`  

- `-e/-E/--EFILE <input filename>`  
Specify input file for Exponential mixture deconvolution mode. The file contains a list of values comming from a mixture of univariate exponentials.  
_Both -g/e and -i may not be simultaneously specified._  

  > `-5.01966537271658`  
  > `-7.4061877009785`  
  > `-2.15413929205554`  
  > `-17.0618868673418`  
  > `0.225835191938541`  
  > `-16.6095956294893`  
  > `-18.4864630074395`  

- `-o/-O/--OFILE <output filename>`  
Specify output file. By default, the timestamp is used as the file name.  

- `-r/-R/--RTOLE <relative tolerance>`  
EM Stopping Criteria.  
Default: `0.00001`  

- `-v/-V/--VERBO`  
Display of status and info messages.  

- `-k/-K/--KMIXT <value>`  
Number of mixture distribution. Valid only with `-e/-g` option.  
Default: 3  

- `-t/-T/--TERMI`  
Show results in the terminal.  

- `-c/-C/--CSEED <value>`  
Seed for random number generator.

- `-m/-M/--MAXIT <value>`  
Maximum number of EM iteration.  
Default: 1000  


### example
Run coarse multinomial mode and write results to `abn.txt` file.  
```
gemmulem -i sim.tsv -o abn.txt
```


## C/C++ APIs

## R APIs

## Python APIs



<p align="right">(<a href="#top">back to top</a>)</p>
 

# Contact  

<p align="right">(<a href="#top">back to top</a>)</p>


