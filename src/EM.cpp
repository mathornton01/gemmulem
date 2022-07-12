#include <iostream> 
#include <fstream>
#include <sstream> 
#include <string> 
#include <string.h>
#include <vector> 
#include <algorithm>
#include <assert.h>
#include <math.h>
#include <chrono> 

#include "EM.h"

using namespace std; 
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock; 


struct gaussianemresults unmixgaussians(vector<double> values, int numGaussians, int maxiter, bool verb, double rtole){
    // First we must initialize the parameters. 
    //  I will provide a function for this.  The first function will just initialize them
    //  randomly depending on the value vector.
    struct gaussianemresults retgem = randominitgem(values,numGaussians); 
    bool mconv = false;
    bool vconv = false;
    bool pconv = false; 
    vector<double> mcur = retgem.means_init;
    vector<double> mprev = mcur; 
    vector<double> vcur = retgem.vars_init;
    vector<double> vprev = vcur; 
    vector<double> pcur = retgem.probs_init;
    vector<double> pprev = pcur; 
    double merr = NULL; 
    double verr = NULL; 
    double perr = NULL;
    int iter = 1; 

    if (verb){
        cout << "INFO: Gaussian Mixture Deconvolution by Expectation Maximization" << endl; 
        cout << "INFO:  Number of Gaussians involved in mixture - " << to_string(numGaussians) << endl; 
        cout << "INFO:  Number of Values observed total - " << to_string(values.size()) << endl; 
    }

    while((!mconv || !vconv || !pconv) && iter <= maxiter){
       if (verb){
            cout << "INFO:  EM Iteration - " << to_string(iter) << " | Rel. Error Mean : " << merr << " | Rel. Error Var : " << verr << " | Rel. Error Prop : " << perr << endl;
       }
            // << "  MEAN - " << endl << 
        //            to_string(mcur[0]) << endl << 
         //           to_string(mcur[1]) << endl << 
         //           to_string(mcur[2]) << endl;
        //}
        // Get the Likelihoods 
        vector<double> lhval;
        vector<vector<double>> lhall;
        double rowtotal = 0;
        for (int i = 0; i < values.size(); i++){
            lhval.clear();
            rowtotal = 0;
            for (int j = 0; j < numGaussians; j++){
             //   cout << "Geting Normal LH for: " << to_string(values[i]) << " from: N(" << to_string(mcur[j]) << "," << to_string(vcur[j]) << ") " << endl;
             //   cout << "  IT is " << to_string(getNormLH(values[i],mcur[j],vcur[j])*pcur[j]) << endl;
                if (getNormLH(values[i],mcur[j],vcur[j])*pcur[j] != 0) {
                    lhval.push_back(getNormLH(values[i],mcur[j],vcur[j])*pcur[j]);
                    rowtotal+=getNormLH(values[i],mcur[j],vcur[j])*pcur[j];
                }
                else if (getNormLH(values[i],mcur[j],vcur[j])*pcur[j] == 0){
                    lhval.push_back(0.000001);
                    rowtotal+=0.000001;
                }
                
         //       cout << "   Row Total : " << to_string(rowtotal) << endl;
          //      cout << "   LH over Row Total" << to_string(getNormLH(values[i],mcur[j],vcur[j])*pcur[j]/rowtotal);
            }
            for (int j = 0; j < numGaussians; j++){
                lhval[j] = lhval[j]/rowtotal;
            }
            lhall.push_back(lhval);
        }
        
     //   for (int i = 0; i < values.size(); i++){
      //      for (int j = 0; j < numGaussians; j++){
       //         cout << lhall[i][j] << " "; 
        //    } 
         //   cout << endl;
       // }
        // Generate the new proportion estimates 
        for(int j = 0; j < numGaussians; j++){
            pcur[j] = 0;
            for(int i = 0; i < values.size(); i++){
                pcur[j] += lhall[i][j];
            }
            pcur[j] /= values.size();
        }
        // Get the error estimate 
        perr = 0;
        for (int j = 0; j < numGaussians; j++){
            perr += sqrt(pow(pcur[j]-pprev[j],2));
        }
        perr /= numGaussians;

        pprev = pcur;

        pconv = (perr < rtole);


        // Generate the mean estimates 
        double colsum;
        for (int j = 0; j < numGaussians; j++){
            mcur[j] = 0; 
            colsum = 0;
            for (int i = 0; i < values.size(); i++){
                mcur[j] += lhall[i][j]*values[i];
                colsum += lhall[i][j];
            }
            mcur[j] /= colsum;
        }

        // Get the error estimate 
        merr = 0; 
        for (int j = 0; j < numGaussians; j++){
            merr += sqrt(pow(mcur[j]-mprev[j],2));
        }
        merr /= numGaussians;

        mprev = mcur;

        mconv = (merr < rtole);


        // Generate the Variance estimates
        for (int j = 0; j < numGaussians; j++){
            vcur[j] = 0;
            colsum = 0;
            for (int i = 0; i < values.size(); i++){
                vcur[j] += lhall[i][j]*(pow(values[i]-mcur[j],2));
                colsum += lhall[i][j];
            }
            vcur[j] /= colsum;
        }

        verr = 0;
        for (int j = 0; j < numGaussians; j++){
            verr += sqrt(pow(vcur[j]-vprev[j],2));
        }
        verr /= numGaussians;

        vprev = vcur; 

        vconv = (verr < rtole);

        iter++;
    }
    retgem.means_final = mcur;
    retgem.probs_final = pcur;
    retgem.vars_final = vcur;
    retgem.iterstaken = iter;
    return(retgem);
}
struct gaussianemresults randominitgem(vector<double> values, int numGaussians){
    // A function for randomly intializing the starting parameters for the Gaussian mixture problem.
    struct gaussianemresults retgem;
    retgem.numGaussians = numGaussians; 
    double minv = min(values);
    double maxv = max(values); 
    double meanv = mean(values); 
    double varv = var(values); 
    for (int i = 0; i < numGaussians; i++){
        // Initialize the means randomly as starting values in the range. 
        int ri = rand() % 10000;
        double rd = double(ri)/double(10000);
        double rnmn = (maxv-minv)*rd + minv;
        retgem.means_init.push_back(rnmn);

        // Initialize the variances randomly as between 0.8 and 1.2 * the total variance. 
        ri = rand() % 10000; 
        rd = double(ri)/double(10000);
        double rnvr = (varv*0.8-varv*0.4)*rd+varv*0.4;
        retgem.vars_init.push_back(rnvr);

        // Initialize the probabilities as uniform
        retgem.probs_init.push_back(double(1.0L/numGaussians));
    }
    return(retgem);
}


double min(std::vector<double> values){
    double min = values[0];
    for (int i = 0; i < values.size(); i++){
        if (min > values[i])
            {   min = values[i];    }
    }
    return(min);
}

double max(std::vector<double> values){
    double max = values[0];
    for (int i = 0; i < values.size(); i++){
        if (max < values[i]){
            max = values[i];
        }
    }
    return(max);
}

double mean(std::vector<double> values){
    double mean = values[0];
    for (int i = 1; i < values.size(); i++){
        mean = mean+values[i];
    }
    mean = mean/values.size(); 
    return(mean);
}

double var(std::vector<double> values){
    double mn = mean(values); 
    double vr = (values[0]-mn)*(values[0]-mn);
    for (int i = 1; i < values.size(); i++){
        vr = vr + (values[i]-mn)*(values[i]-mn);
    }
    vr = vr/values.size();
    return(vr);
}

double getNormLH(double value, double mean, double var){
    double sd = sqrt(var);
    double LH = (1/(sd*sqrt(2*3.14159265)))*pow(2.71828182,(-0.5)*pow((value-mean)/sd,2));
    return(LH);
}
struct exponentialEMResults unmixexponentials(vector<double> values, int numExponentials, int maxiter = 1000, bool verb = false, double rtole){
    // First we must initialize the parameters. 
    //  I will provide a function for this.  The first function will just initialize them
    //  randomly depending on the value vector.
    struct exponentialEMResults reteem = randominiteem(values,numExponentials); 
    bool mconv = false;
    bool pconv = false; 
    vector<double> mcur = reteem.means_init;
    vector<double> mprev = mcur; 
    vector<double> pcur = reteem.probs_init;
    vector<double> pprev = pcur; 
    double merr = NULL; 
    double perr = NULL;
    int iter = 1; 

    if (verb){
        cout << "INFO: Exponential Mixture Deconvolution by Expectation Maximization" << endl; 
        cout << "INFO:  Number of Exponentials involved in mixture - " << to_string(numExponentials) << endl; 
        cout << "INFO:  Number of Values observed total - " << to_string(values.size()) << endl; 
    }

    while((!mconv || !pconv) && iter <= maxiter){
        if (verb){
            cout << "INFO:  EM Iteration - " << to_string(iter) << " | Rel. Error Mean = " << to_string(merr) << " | Rel. Error Props = " << to_string(perr) << endl;// << "  MEAN - " << endl << 
        }//            to_string(mcur[0]) << endl << 
         //           to_string(mcur[1]) << endl << 
         //           to_string(mcur[2]) << endl;
        //}
        // Get the Likelihoods 
        vector<double> lhval;
        vector<vector<double>> lhall;
        double rowtotal = 0;
        for (int i = 0; i < values.size(); i++){
            lhval.clear();
            rowtotal = 0;
            for (int j = 0; j < numExponentials; j++){
             //   cout << "Geting Exponential LH for: " << to_string(values[i]) << " from: Exp(" << to_string(mcur[j]) << ") " << endl;
             //   cout << "  IT is " << to_string(getExpoLH(values[i],mcur[j],vcur[j])*pcur[j]) << endl;
                if (getExpoLH(values[i],mcur[j])*pcur[j] != 0) {
                    lhval.push_back(getExpoLH(values[i],mcur[j])*pcur[j]);
                    rowtotal+=getExpoLH(values[i],mcur[j])*pcur[j];
                }
                else if (getExpoLH(values[i],mcur[j])*pcur[j] == 0){
                    lhval.push_back(0.000001);
                    rowtotal+=0.000001;
                }
                
         //       cout << "   Row Total : " << to_string(rowtotal) << endl;
          //      cout << "   LH over Row Total" << to_string(getNormLH(values[i],mcur[j],vcur[j])*pcur[j]/rowtotal);
            }
            for (int j = 0; j < numExponentials; j++){
                lhval[j] = lhval[j]/rowtotal;
            }
            lhall.push_back(lhval);
        }
        
     //   for (int i = 0; i < values.size(); i++){
      //      for (int j = 0; j < numGaussians; j++){
       //         cout << lhall[i][j] << " "; 
        //    } 
         //   cout << endl;
       // }
        // Generate the new proportion estimates 
        for(int j = 0; j < numExponentials; j++){
            pcur[j] = 0;
            for(int i = 0; i < values.size(); i++){
                pcur[j] += lhall[i][j];
            }
            pcur[j] /= values.size();
        }
        // Get the error estimate 
        perr = 0;
        for (int j = 0; j < numExponentials; j++){
            perr += sqrt(pow(pcur[j]-pprev[j],2));
        }
        perr /= numExponentials;

        pprev = pcur;

        pconv = (perr < rtole);


        // Generate the mean estimates 
        double colsum;
        for (int j = 0; j < numExponentials; j++){
            mcur[j] = 0; 
            colsum = 0;
            for (int i = 0; i < values.size(); i++){
                mcur[j] += lhall[i][j]*values[i];
                colsum += lhall[i][j];
            }
            mcur[j] /= colsum;
        }

        // Get the error estimate 
        merr = 0; 
        for (int j = 0; j < numExponentials; j++){
            merr += sqrt(pow(mcur[j]-mprev[j],2));
        }
        merr /= numExponentials;

        mprev = mcur;

        mconv = (merr < rtole);

        iter++;
    }
    reteem.means_final = mcur;
    reteem.probs_final = pcur;
    reteem.iterstaken = iter;
    return(reteem);
}

struct exponentialEMResults randominiteem(vector<double> values, int numExponentials){
    // A function for randomly intializing the starting parameters for the Gaussian mixture problem.
    struct exponentialEMResults reteem;
    reteem.numExponentials = numExponentials; 
    double minv = min(values);
    double maxv = max(values); 
    double meanv = mean(values); 
    double varv = var(values); 
    for (int i = 0; i < numExponentials; i++){
        // Initialize the means randomly as starting values in the range. 
        int ri = rand() % 10000;
        double rd = double(ri)/double(10000);
        double rnmn = (maxv-minv)*rd + minv;
        reteem.means_init.push_back(rnmn);

        // Initialize the probabilities as uniform
        reteem.probs_init.push_back(double(1.0L/numExponentials));
    }
    return(reteem);
}
double getExpoLH(double value, double mn){
    if (value < 0){
        return(0.0);
    } else {
        double LH = (1.0/mn)*pow(2.718281,-(1.0/mn)*value);
        return(LH);
    }
}
vector<double> expectationmaximization(struct comppatcounts cpc, struct emsettings ems){
    int start = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
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
        cout << "INFO: EM - Iteration " << to_string(num_iter) << " Relative Error: " << to_string(relerr) << endl;
    }
    if (relerr <= ems.rtole){
        if (ems.verbose){
            cout << "INFO: EM - Ran for " << to_string(num_iter) << " iterations" << endl;
            int end = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
            cout << "INFO: EM - Took " << to_string(end-start) << " nanoseconds to run" << endl;
        }
        return(abdn_new);
    }
    abdn_old = abdn_new;
    num_iter++;
    }
}

