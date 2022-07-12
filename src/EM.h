#ifndef __EM_H__
#define __EM_H__

#include <vector>
#include <string>

using namespace std;

// structures 
struct emsettings{
    string ifilename; 
    string ofilename;
    string samfilename;
    string valfilename;
    string type;
    int kmixt;
    bool verbose = false; 
    bool termcat = false;
    double rtole;
};
struct gaussianemresults{
    int numGaussians;
    int iterstaken;
    vector<double> means_init;
    vector<double> vars_init;
    vector<double> probs_init;
    vector<double> means_final;
    vector<double> vars_final;
    vector<double> probs_final;
};
struct exponentialEMResults{
    int numExponentials;
    int iterstaken;
    vector<double> means_init;
    vector<double> probs_init;
    vector<double> means_final;
    vector<double> probs_final;
};

struct comppatcounts{
    std::vector<string> compatibilityPattern;
    std::vector<int> count;
};

vector<double> expectationmaximization(struct comppatcounts cpc, struct emsettings ems);
// Function headers 
struct gaussianemresults unmixgaussians(vector<double> values, int numGaussians, int maxiter = 1000, bool verb = false, double rtole=0.000001);

struct gaussianemresults randominitgem(std::vector<double> values, int numGaussians); 

//TODO: Implement K-Means initialization procedure 
struct gaussianemresults kmeansinitgem(std::vector<double> values, int numGaussians);

// Function headers 

vector<double> readvaluefile(string ifilename); 
struct exponentialEMResults unmixexponentials(vector<double> values, int numExponentials, int maxiter, bool verb, double rtole=0.000001);

struct exponentialEMResults randominiteem(vector<double> values, int numExponentials); 

//TODO: Implement K-Means initialization procedure 
struct exponentialEMResults kmeansiniteem(vector<double> values, int numExponentials);


// Some simple utility functions I need.
double min(std::vector<double> values);
double max(std::vector<double> values);
double mean(std::vector<double> values);
double var(std::vector<double> values);
double getNormLH(double value, double mean, double var);
double getExpoLH(double value, double mn);

#endif /* __EM_H__ */
