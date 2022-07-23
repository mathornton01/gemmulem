/*
 * Copyright 2022, Micah Thornton and Chanhee Park <parkchanhee@gmail.com>
 *
 * This file is part of GEMMULEM
 *
 * GEMMULEM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GEMMULEM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GEMMULEM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream> 
#include <string> 
#include <vector>
#include <math.h>
#include <fstream>
#include <cstdlib>

using namespace std;

// Function headers 
vector<double> expectationmaximization(struct comppatcounts, struct emsettings ems);
vector<double> readvaluefile(string ifilename); 
struct gaussianemresults unmixgaussians(vector<double> values, int numGaussians, int maxiter, bool verb);

struct gaussianemresults randominitgem(vector<double> values, int numGaussians); 

//TODO: Implement K-Means initialization procedure 
struct gaussianemresults kmeansinitgem(vector<double> values, int numGaussians);

// Some simple utility functions I need.
double min(vector<double> values);
double max(vector<double> values);
double mean(vector<double> values);
double var(vector<double> values);
double getNormLH(double value, double mean, double var);

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

int main(int argc, char ** argv){
    cout << to_string(getNormLH(0,0,1)) << endl;
    srand(0xFEED);

    bool tst1 = false; 
    bool tst2 = false; 
    bool tst3 = true;
    vector<double> tstvals;
    struct gaussianemresults emres;
    int ng = 3;
    if(tst1){
    // gmm_tst_vals_3.txt - means (1.56, 12.43, -22.51), vars (4.10, 18.02, 20.549), probs (0.476, 0.345, 0.179)
    tstvals = readvaluefile("gmm_tst_vals_3.txt");
    ng = 3;
    emres = unmixgaussians(tstvals,ng,1000,true);
    for (int i = 0; i < ng; i++){
        cout << "Gaussian " << to_string(i) << " Mean Estimate: " << to_string(emres.means_final[i]) << " Var Estimate: " << to_string(emres.vars_final[i]) << " Prob Estimate: " << to_string(emres.probs_final[i]) << endl;
    }
    }

    if(tst2){
    // gmm_tst_vals_4.txt - means ([1]  -0.03978567  -5.79104219  10.82317081 -22.42147825  23.97086660), 
    //                       vars ([1]   3.985878     2.169658     8.301868     8.199631    15.147265  ), 
    //                      probs ([1]   0.1815995    0.2347567    0.2322382    0.1924087    0.1589970)
    tstvals = readvaluefile("gmm_tst_vals_4.txt");
    ng = 5;
    emres = unmixgaussians(tstvals,ng,1000,true);
    for (int i = 0; i < ng; i++){
        cout << "Gaussian " << to_string(i) << " Mean Estimate: " << to_string(emres.means_final[i]) << " Var Estimate: " << to_string(emres.vars_final[i]) << " Prob Estimate: " << to_string(emres.probs_final[i]) << endl;
    }
    }

    if(tst3){
    // gmm_tst_vals_5.txt - means ([1]  -7.602595  23.644680  -6.180415 -14.520220   8.800419 -19.425961  17.958428).
    //                       vars ([1]  14.154774   1.380286  16.421753   1.024074  22.598802   1.243414  17.771734).
    //                      probs ([1] 0.20974384 0.08569973 0.21116514 0.05203155 0.20831706 0.08606674 0.14697593)
    tstvals = readvaluefile("gmm_tst_vals_6.txt");
    ng = 7;
    srand(0XBEEF);
    emres = unmixgaussians(tstvals,ng,400,true);
    for (int i = 0; i < ng; i++){
        cout << "Gaussian " << to_string(i) << " Mean Estimate: " << to_string(emres.means_final[i]) << " Var Estimate: " << to_string(emres.vars_final[i]) << " Prob Estimate: " << to_string(emres.probs_final[i]) << endl;
    }
    return(0);
    }
}

vector<double> readvaluefile(string ifilename){
    vector<double> inputvalues; 
    ifstream ifile(ifilename,std::ios::in);
    double num = 0.0;
    while(ifile >> num){
        inputvalues.push_back(num);
    }
    return(inputvalues);
}
struct gaussianemresults unmixgaussians(vector<double> values, int numGaussians, int maxiter = 1000, bool verb = false){
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
       // if (verb){
        //    cout << "INFO:  EM Iteration - " << to_string(iter) << "  MEAN - " << endl << 
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

        pconv = (perr < 0.000001);


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

        mconv = (merr < 0.000001);


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

        vconv = (verr < 0.000001);

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

double min(vector<double> values){
    double min = values[0];
    for (int i = 0; i < values.size(); i++){
        if (min > values[i])
            {   min = values[i];    }
    }
    return(min);
}

double max(vector<double> values){
    double max = values[0];
    for (int i = 0; i < values.size(); i++){
        if (max < values[i]){
            max = values[i];
        }
    }
    return(max);
}

double mean(vector<double> values){
    double mean = values[0];
    for (int i = 1; i < values.size(); i++){
        mean = mean+values[i];
    }
    mean = mean/values.size(); 
    return(mean);
}

double var(vector<double> values){
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
