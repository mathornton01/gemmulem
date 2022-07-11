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
    vector<double> tstvals = readvaluefile("gmm_tst_vals_1.txt");
    int ng = 3;
    struct gaussianemresults emres = unmixgaussians(tstvals,ng,1000,true);
    for (int i = 0; i < ng; i++){
        cout << "Gaussian " << to_string(i) << " Mean Estimate: " << to_string(emres.means_final[i]) << " Var Estimate: " << to_string(emres.vars_final[i]) << " Prob Estimate: " << to_string(emres.probs_final[i]) << endl;
    }
    return(0);
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