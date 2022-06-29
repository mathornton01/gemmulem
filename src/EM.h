#ifndef __EM_H__
#define __EM_H__

#include <vector>

// structures 
struct emsettings{
    std::string ifilename; 
    std::string ofilename;
    std::string samfilename;
    bool verbose = false; 
    bool termcat = false;
    double rtole;
};


struct comppatcounts{
    std::vector<std::string> compatibilityPattern;
    std::vector<int> count;
};

// Function headers 
std::vector<double> expectationmaximization(struct comppatcounts, struct emsettings ems);

#endif /* __EM_H__ */
