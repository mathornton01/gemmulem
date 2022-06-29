#include <iostream> 
#include <fstream>
#include <sstream> 
#include <string> 
#include <string.h>
#include <vector> 
#include <algorithm>
#include <assert.h>
#include <math.h>

#include "EM.h"

using namespace std; 

vector<double> expectationmaximization(struct comppatcounts cpc, struct emsettings ems){
    double numtrans = cpc.compatibilityPattern[0].size();
    double numpats = cpc.compatibilityPattern.size();
    vector<double> abdninit; // Initialization vector for the abundance values 
    vector<double> abdn_new; // For storing the newly computed abundance values at a given stage 
    vector<double> abdn_old; // For containing the older abundance values.
    vector<double> expected_counts; // For storing the intermediate expected counts as they are recomputed at each iteration of the EM algorithm.
    double totalreads = 0; // For storing the total number of reads in the comppatcounts struct.
    double totalcomp = 0; // for storing the total number of compatible transcripts for each read in the same structure. 
    double abdn_total = 0; // for storing the total abundance on each row of the compatibility matrix. 
    double relerr = 1000;
    for (int i = 0; i < cpc.count.size(); i++){totalreads=totalreads+cpc.count[i];}   
    //cout << " Total Reads : " << to_string(totalreads);  
    for (int i = 0; i < numtrans; i++){
        abdninit.push_back(1/numtrans);
        expected_counts.push_back(totalreads/numtrans);
    }
    abdn_new = abdninit; // initialize both the previous and current iteration abundances to the initialization values.
    abdn_old = abdninit; 
    int num_iter = 0;
    
    while(true){
    expected_counts.clear();
    for(int j = 0; j < numtrans; j++){
        expected_counts.push_back(0);
    }
    for (int i = 0; i < numpats; i++){
        abdn_total = 0;
        totalcomp = 0;
        for (int j = 0; j < numtrans; j++){
            totalcomp += int(cpc.compatibilityPattern[i][j] == '1');
            if (cpc.compatibilityPattern[i][j] == '1'){
                abdn_total += abdn_old[j];
            }
        }
        for (int j = 0; j < numtrans; j++){
            if (cpc.compatibilityPattern[i][j] == '1'){
                expected_counts[j] += ((abdn_old[j]/abdn_total)*cpc.count[i]);
            }
        }
    }
    for (int j = 0; j < numtrans; j++){
        abdn_new[j] = expected_counts[j]/totalreads;
    }
    relerr = 0; 
    for(int j = 0; j < numtrans; j++){
        //cout << "abdn_new[" << to_string(j) << "] = " << to_string(abdn_new[j]) << "            " << "abdn_old[" << to_string(j) << "] = " << to_string(abdn_old[j]) << endl;
        relerr += ((abdn_new[j])-(abdn_old[j]))*((abdn_new[j])-(abdn_old[j]));
        //cout << to_string(j) << ": alpha_" << to_string(j) << " = " << abdn_old[j] << ", alpha*_" << to_string(j) << "=" << abdn_new[j] << "Relative Error = " << to_string(relerr) << endl; 
    }
    relerr /= numtrans; 
    relerr = sqrt(relerr);
    if (ems.verbose){
        cout << "Info: EM-Algorithm, Iteration " << to_string(num_iter) << " Relative Error: " << to_string(relerr) << endl;
    }
    if (relerr <= ems.rtole){
        if (ems.verbose){
            cout << "Info: EM-Algorith Ran for " << to_string(num_iter) << " iterations" << endl;
        }
        return(abdn_new);
    }
    abdn_old = abdn_new;
    num_iter++;
    }
}

