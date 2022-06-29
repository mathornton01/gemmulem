# GeMMulEM (\[Ge\]neral \[M\]ixed \[Mul\]tinomial \[E\]xpectation \[M\]aximization)

This version of the gemmulem program is intended to be very easy to install and use, and therefore only
requires the GNU make utility for its installation. 

To compile this utility from its sources the only tools you will need in addition to make is gcc. 
As of now, this utility is intended to be run from a linux environment, therefore if you are using 
a windows machine you should use the windows subsystem for linux, or a simulator such as cygwin for 
installing and using gemmulem. 

```bash
git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem
sudo make install
```

This will build the gemmulem application and install the binary in the /bin/ directory. 
To remove the binary from the /bin/ directory, you can use

```bash
sudo make uninstall 
```

