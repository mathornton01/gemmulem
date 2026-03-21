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
#include <fstream>
#include <sstream> 
#include <string> 
#include <cstring>
#include <vector> 
#include <algorithm>
#include <cassert>
#include <cmath>

#include "EM.h"
#include "distributions.h"
#include "multivariate.h"
#include "complex_em.h"
#include "streaming.h"

using namespace std;

struct emsettings{
    string ifilename; 
    string ofilename;
    string samfilename;
    string valfilename;
    string type;
    string distname;
    int kmixt;
    int kmax = 0;
    int maxitr;
    bool verbose = false; 
    bool termcat = false;
    bool kmeans_init = false;
    bool autoselect = false;
    bool adaptive = false;
    bool online = false;
    int batch_size = 0;
    bool multivariate = false;
    bool mv_studentt = false;
    bool mv_autok = false;
    bool streaming = false;
    int stream_chunk = 10000;
    int stream_passes = 10;
    int mv_dim = 2;
    CovType cov_type = COV_FULL;
    KMethod kmethod = KMETHOD_BIC;
    bool complex_circular = false;
    bool complex_noncircular = false;
    bool complex_autok = false;
    bool complex_mv = false;
    int complex_mv_dim = 2;
    bool complex_streaming = false;
    double rtole;
};
struct genesamfile{
    vector<string> seqname;
    vector<int> seqlen;
    vector<string> flags;
    vector<string> readnames; 
    vector<int> readflags;
    vector<string> primaryalignmentchr;
    vector<int> primaryalignmentpos;
    vector<int> MAPQ; 
    vector<string> CIGAR;
    vector<string> pairrefname; 
    vector<int> pairpos;
    vector<int> templatelength;
    vector<string> sequences;
    vector<string> qualphred;
    vector<vector<string>> tagname;
    vector<vector<string>> tagval;
};
struct comppatcounts{
    std::vector<string> compatibilityPattern;
    std::vector<int> count;
};

void writeemres(string ofilename, vector<double>& abdn);
void writescreenres(vector<double>& abdn);
void printusage(); 
struct emsettings parseargs(int argc, char ** argv); 
struct comppatcounts parseinputfile(string ifilename);
struct comppatcounts parseinputgenesam(string samfilename);
vector<double> parsegmvinputfile(string valfilename);
struct genesamfile readgenesamintostruct(string samfilename);

void MakeEMConfig(EMConfig_t* ConfigPtr, struct emsettings* ems);

void printusage(){
    cout << "                                                                            " << endl;
    cout << "============================================================================" << endl;
    cout << "|               Gemmule (Version 2.0) Thornton & Park                     |" << endl;
    cout << "|   General Mixed-Family Expectation Maximization  --  (HELP)             |" << endl;
    cout << "|                                                                          |" << endl;
    cout << "| INPUT / OUTPUT                                                           |" << endl;
    cout << "|  -g/-G/--GFILE  <file>   Input: one value per line (mixture data)       |" << endl;
    cout << "|  -i/-I/--IFILE  <file>   Input: CSV compatibility patterns (MN mode)    |" << endl;
    cout << "|  -e/-E/--EFILE  <file>   Input: exponential mixture values              |" << endl;
    cout << "|  -o/-O/--OFILE  <file>   Output file (default: <timestamp>.abdn.txt)    |" << endl;
    cout << "|                                                                          |" << endl;
    cout << "| DISTRIBUTION / COMPONENTS                                                |" << endl;
    cout << "|  -d/-D/--DIST   <family> Distribution family (default: Gaussian)        |" << endl;
    cout << "|     Families: Gaussian, StudentT, Laplace, Cauchy, Logistic, Gumbel,    |" << endl;
    cout << "|       SkewNormal, GenGaussian, Exponential, Gamma, LogNormal, Weibull,  |" << endl;
    cout << "|       InvGaussian, Rayleigh, Pareto, ChiSquared, F, LogLogistic,        |" << endl;
    cout << "|       Nakagami, Levy, Gompertz, BurrXII, HalfNormal, MaxwellBoltzmann,  |" << endl;
    cout << "|       Beta, Kumaraswamy, Triangular, Poisson, Binomial, NegBinomial,    |" << endl;
    cout << "|       Geometric, Zipf, Pearson, Uniform, KDE  (35 families total)       |" << endl;
    cout << "|  -k/-K/--KMIXT  <n>      Number of mixture components (default: 3)      |" << endl;
    cout << "|  --kmax         <n>      Max components for auto/adaptive mode (def: 8) |" << endl;
    cout << "|                                                                          |" << endl;
    cout << "| CONVERGENCE                                                              |" << endl;
    cout << "|  -r/-R/--RTOLE  <tol>    Relative tolerance for EM stopping (def: 1e-5) |" << endl;
    cout << "|  -m/-M/--MAXIT  <n>      Maximum EM iterations (default: 1000)          |" << endl;
    cout << "|  -c/-C/--CSEED  <seed>   Integer seed for random number generator       |" << endl;
    cout << "|                                                                          |" << endl;
    cout << "| MODEL SELECTION MODES                                                    |" << endl;
    cout << "|  --adaptive              Adaptive EM: auto-select k + family per comp.  |" << endl;
    cout << "|  --auto                  Exhaustive search: all families × k values     |" << endl;
    cout << "|  --kmethod  <method>     k-selection criterion (default: bic)           |" << endl;
    cout << "|     Methods: bic  aic  icl  vbem  mml                                   |" << endl;
    cout << "|                                                                          |" << endl;
    cout << "| ONLINE / STREAMING EM  (large datasets)                                 |" << endl;
    cout << "|  --online                Online (mini-batch) EM with step-size schedule |" << endl;
    cout << "|  --batch-size   <n>      Mini-batch size for online EM (default: n/4)   |" << endl;
    cout << "|  --stream                Streaming EM: process file in chunks           |" << endl;
    cout << "|  --chunk-size   <n>      Rows per chunk for streaming EM (def: 10000)   |" << endl;
    cout << "|  --passes       <n>      Number of streaming passes over data (def: 10) |" << endl;
    cout << "|                                                                          |" << endl;
    cout << "| MULTIVARIATE MODES  (requires row-major space/comma-separated -g file)  |" << endl;
    cout << "|  --mv / --multivariate   Multivariate Gaussian mixture                  |" << endl;
    cout << "|  --mvt / --mv-studentt   Multivariate Student-t mixture                 |" << endl;
    cout << "|  --mv-autok              Auto-select k for multivariate mixture         |" << endl;
    cout << "|  --dim          <d>      Dimensionality (auto-detected from data)       |" << endl;
    cout << "|  --cov          <type>   Covariance type: full  diagonal  spherical      |" << endl;
    cout << "|                                                                          |" << endl;
    cout << "| COMPLEX-VALUED MODES  (interleaved re,im pairs in -g file)              |" << endl;
    cout << "|  --complex               Circular symmetric complex Gaussian mixture    |" << endl;
    cout << "|  --complex-nc            Non-circular complex Gaussian (pseudo-cov)     |" << endl;
    cout << "|  --complex-autok         Auto-select k for complex circular mixture     |" << endl;
    cout << "|                                                                          |" << endl;
    cout << "| OUTPUT / DISPLAY                                                         |" << endl;
    cout << "|  -v/-V/--VERBO           Verbose: show iteration-by-iteration status    |" << endl;
    cout << "|  -t/-T/--TERMI           Print results to terminal instead of file      |" << endl;
    cout << "|                                                                          |" << endl;
    cout << "| EXAMPLES                                                                 |" << endl;
    cout << "|  gemmulem -g data.txt -d Gaussian -k 2                                  |" << endl;
    cout << "|  gemmulem -g data.txt -d Gamma -k 3 -v                                  |" << endl;
    cout << "|  gemmulem -g data.txt --adaptive --kmax 8 --kmethod bic                 |" << endl;
    cout << "|  gemmulem -g data.txt --stream --chunk-size 5000 --passes 5 -k 3        |" << endl;
    cout << "|  gemmulem -g mvdata.txt --mvt -k 2 --dim 3 --cov full                   |" << endl;
    cout << "|  gemmulem -i patterns.tsv -o output.txt                                 |" << endl;
    cout << "============================================================================" << endl;
    cout << "                                                                            " << endl;
    exit(1);
}

struct emsettings parseargs(int argc, char ** argv){
    string infile = ""; 
    string ofile = "";
    string sfile = "";
    string gfile = "";
    string efile = "";
    string runtype = "";
    int seed; 
    double rtole = -1; 
    int kmixt = 3;
    int kmax = 0;
    int maxitr = 1000;
    struct emsettings ems; 
    bool verbose = false; 
    bool termcat = false;
    bool kmixtwarn = true;
    for (int i = 0; i < argc; i++){
        if (string(argv[i]) == "-i" | string(argv[i]) == "-I" | string(argv[i]) == "--INFILE"){
            infile = string(argv[i+1]);
            runtype = "MN";
        } else if (string(argv[i]) == "-o" | string(argv[i]) == "-O" | string(argv[i]) == "--OFILE"){
            ofile = string(argv[i+1]);
        } else if (string(argv[i]) == "-r" | string(argv[i]) == "-R" | string(argv[i]) == "--RTOLE"){
            rtole = stod(argv[i+1]);
        } else if (string(argv[i]) == "-s" | string(argv[i]) == "-S" | string(argv[i]) == "--SFILE"){
            sfile = string(argv[i+1]);
            runtype = "MN";
        } else if (string(argv[i]) == "-g" | string(argv[i]) == "-G" | string(argv[i]) == "--GFILE"){
            gfile = string(argv[i+1]);
            runtype = "GM";
        } else if (string(argv[i]) == "-e" | string(argv[i]) == "-E" | string(argv[i]) == "--EFILE"){
            efile = string(argv[i+1]);
            runtype = "EM";
        } else if (string(argv[i]) == "-k" | string(argv[i]) == "-K" | string(argv[i]) == "--KMIXT"){
            kmixt = stoi(string(argv[i+1]));
            kmixtwarn = false;
        }
        else if (string(argv[i]) == "-h" | string(argv[i]) == "-H" | string(argv[i]) == "--HELP"){
            printusage();
        } else if (string(argv[i]) == "-v" | string(argv[i]) == "-V" | string(argv[i]) == "--VERBO"){
            verbose = true;
        } else if (string(argv[i]) == "-t" | string(argv[i]) == "-T" | string(argv[i]) == "--TERMI"){
            termcat = true;
        } else if (string(argv[i]) == "-c" | string(argv[i]) == "-C" | string(argv[i]) == "--CSEED"){
            srand(stoi(string(argv[i+1])));
        } else if (string(argv[i]) == "-m" | string(argv[i]) == "-M" | string(argv[i]) == "--MAXIT"){
            maxitr = stoi(string(argv[i+1]));
        } else if (string(argv[i]) == "--kmeans" | string(argv[i]) == "--KMEANS"){
            ems.kmeans_init = true;
        } else if (string(argv[i]) == "--auto" | string(argv[i]) == "--AUTO"){
            ems.autoselect = true;
        } else if (string(argv[i]) == "--adaptive" | string(argv[i]) == "--ADAPTIVE"){
            ems.adaptive = true;
        } else if (string(argv[i]) == "--kmethod" | string(argv[i]) == "--KMETHOD"){
            string m = string(argv[i+1]);
            if (m == "bic" || m == "BIC") ems.kmethod = KMETHOD_BIC;
            else if (m == "aic" || m == "AIC") ems.kmethod = KMETHOD_AIC;
            else if (m == "icl" || m == "ICL") ems.kmethod = KMETHOD_ICL;
            else if (m == "vbem" || m == "VBEM") ems.kmethod = KMETHOD_VBEM;
            else if (m == "mml" || m == "MML") ems.kmethod = KMETHOD_MML;
            else { cout << "ERROR: Unknown kmethod '" << m << "'. Use bic/aic/icl/vbem/mml" << endl; exit(1); }
        } else if (string(argv[i]) == "--online" | string(argv[i]) == "--ONLINE"){
            ems.online = true;
        } else if (string(argv[i]) == "--batch-size" | string(argv[i]) == "--BATCH-SIZE"){
            ems.batch_size = stoi(string(argv[i+1]));
        } else if (string(argv[i]) == "-d" | string(argv[i]) == "-D" | string(argv[i]) == "--DIST"){
            ems.distname = string(argv[i+1]);
        } else if (string(argv[i]) == "--kmax" | string(argv[i]) == "--KMAX"){
            kmax = stoi(string(argv[i+1]));
        } else if (string(argv[i]) == "--mv" || string(argv[i]) == "--multivariate"){
            ems.multivariate = true;
        } else if (string(argv[i]) == "--mv-studentt" || string(argv[i]) == "--mvt"){
            ems.multivariate = true;
            ems.mv_studentt = true;
        } else if (string(argv[i]) == "--mv-autok"){
            ems.multivariate = true;
            ems.mv_autok = true;
        } else if (string(argv[i]) == "--stream" || string(argv[i]) == "--streaming"){
            ems.streaming = true;
        } else if (string(argv[i]) == "--chunk-size"){
            ems.stream_chunk = stoi(string(argv[i+1]));
        } else if (string(argv[i]) == "--passes"){
            ems.stream_passes = stoi(string(argv[i+1]));
        } else if (string(argv[i]) == "--dim"){
            ems.mv_dim = stoi(string(argv[i+1]));
        } else if (string(argv[i]) == "--cov"){
            string ct = string(argv[i+1]);
            if (ct == "full") ems.cov_type = COV_FULL;
            else if (ct == "diagonal" || ct == "diag") ems.cov_type = COV_DIAGONAL;
            else if (ct == "spherical" || ct == "sph") ems.cov_type = COV_SPHERICAL;
        } else if (string(argv[i]) == "--complex"){
            ems.complex_circular = true;
        } else if (string(argv[i]) == "--complex-nc" || string(argv[i]) == "--complex-noncircular"){
            ems.complex_noncircular = true;
        } else if (string(argv[i]) == "--complex-autok"){
            ems.complex_circular = true;
            ems.complex_autok = true;
        } else if (string(argv[i]) == "--complex-mv"){
            ems.complex_mv = true;
        } else if (string(argv[i]) == "--complex-mv-dim" && i+1 < argc){
            ems.complex_mv_dim = stoi(string(argv[++i]));
        } else if (string(argv[i]) == "--complex-stream"){
            ems.complex_streaming = true;
            ems.complex_circular = true;
        }
    }
    if (infile == "" && sfile == "" && gfile == "" && efile == ""){
        cout << "ERROR:  Please specify a valid input file. " << endl << endl << endl; 
        printusage();
    }
    if (ofile == ""){
        ofile = to_string(int(time(0)))+".abdn.txt";
        if (verbose){
            cout << "WARN: No Output File Specified, writing to " << ofile << endl;
        }
    }
    if (rtole < 0){
        cout << "WARN: Relative tolerance either not specified, or incorrect value, default (0.00001) is being used." << endl;
        rtole = 0.00001;
    }
    if (infile != "" && sfile != "" && gfile != "" && efile != ""){
        cerr << "ERROR: Please specify one only of -s, -i, -g, -e flags depending on the input type. " << endl << endl << endl; 
        exit(1);
    }
    if ((runtype == "GM" || runtype == "EM") && kmixtwarn && verbose){
        cout << "WARN: Number of Mixtures to deconvolve in sample not specified, or incorrect value, default (3) is being used." << endl;
    }
    ems.ifilename = infile; 
    ems.type = runtype;
    if (ems.type == "GM"){
        ems.valfilename = gfile;
    } else if (ems.type == "EM"){
        ems.valfilename = efile;
    }
    ems.ofilename = ofile;
    ems.samfilename = sfile;
    ems.kmixt = kmixt;
    if (ems.kmax == 0) ems.kmax = kmax;
    ems.rtole = rtole; 
    ems.maxitr = maxitr;
    ems.verbose = verbose;
    ems.termcat =termcat; 
    return(ems); 
}

int main(int argc, char** argv)
{

    cout <<
         "                         (Gemmule)                          " << endl <<
         "      General Mixed Multinomial Expectation Maximization    " << endl <<
         "              Micah Thornton & Chanhee Park (2022-2026)     " << endl <<
         "                        [Version 2.0]                       " << endl << endl;

    // Store the user settings for the EM algorithm in the ems structure. 
    struct emsettings ems = parseargs(argc, argv);

    if (ems.verbose){
        cout << "INFO: User Settings - Running Gemmule in Verbose Mode (-v)" << endl;
        if (ems.ifilename != ""){
            if (ifstream(ems.ifilename).is_open()){
                cout << "INFO: User Settings - Running Gemmule in Multinomial De-Coarsening Mode, reading compatibility count input. " << endl;
                cout << "INFO: File IO - (Pattern File -i), Parsing Input File " << ems.ifilename << "." << endl;
            } else {
                cout << "ERROR: File IO - (" << ems.ifilename << ") Not Found Please specify a different file location." << endl << endl << endl;
                exit(1);
            }
        } else if (ems.samfilename != ""){
            cout << "INFO: File IO - File Input (GENE SAM File -s), Parsing Input File " << ems.samfilename << "." << endl;
        }
    }

    // Parse the compatibility pattern input file and store the compatibility patterns and counts. 
    struct comppatcounts cpc;
    vector<double> umv; // univariate mixture values
    if (ems.ifilename != ""){
        if (ems.verbose){cout << "INFO: File IO - Standard Compatibility Patterns Input " << endl;}
        cpc = parseinputfile(ems.ifilename);
    } else if (ems.samfilename != ""){
        if (ems.verbose){cout << "INFO: File IO - Standard Compatibility Patterns Input " << endl;}
        cpc = parseinputgenesam(ems.samfilename);
    } else if (ems.valfilename != ""){
        if (ems.verbose){cout << "INFO: File IO - Standard Mixture Values Input " << endl;}
        umv = parsegmvinputfile(ems.valfilename);
    }

    if (ems.type=="MN"){
        if (ems.verbose){
            cout << "INFO: File IO - Found Compatilbility Patterns " << endl;
            for (int i = 0; i < cpc.compatibilityPattern.size(); i++){
                cout << "INFO:      " << to_string(i) << " - " << cpc.compatibilityPattern[i] << " count - " << to_string(cpc.count[i]) << endl;
            }
        }

        EMConfig_t EMConfig;
        EMResult_t Result;

        MakeEMConfig(&EMConfig, &ems);

        std::string CompatMatrix;

        for(int i = 0; i < cpc.compatibilityPattern.size(); i++) {
            CompatMatrix.append(cpc.compatibilityPattern[i]);
        }

        // Perform expectation maximization on the compatibility patterns and counts.
        //vector<double> emabundances = expectationmaximization(cpc,ems);
        ExpectationMaximization(
                CompatMatrix.data(), /* Compatiblity Matrix */
                cpc.compatibilityPattern.size(), /* NumRows */
                cpc.compatibilityPattern[0].size(), /* NumCols */
                cpc.count.data(), /* CountPtr */
                cpc.count.size(), /* NumCount */
                &Result, &EMConfig);
        vector<double> emabundances(Result.size, 0.0);
        memcpy(emabundances.data(), Result.values, Result.size * sizeof(double));
        ReleaseEMResult(&Result);

        if (ems.verbose){
            if (ems.termcat){
                cout << "INFO: Results - MLE of Proportions" << endl;
                for (int j = 0; j < cpc.compatibilityPattern[0].size(); j++){
                    cout << "INFO:      " << " Transcript - " << to_string(j) << " EM count: " << to_string(emabundances[j]) << endl;
                }
            }
            cout << "INFO: File IO - Writing Results to Output File " << ems.ofilename << "." << endl;
        }

        // Write the determined proportions to an output file, or the screen

        if (ems.termcat){
            writescreenres(emabundances);
            return(0);
        }
        writeemres(ems.ofilename,emabundances);

        cout << endl << endl;
        return(0);
    }

    /* Type "GM" (Gaussian via -g flag): route to new SIMD-accelerated engine.
     * Old legacy path (UnmixGaussians) was 10-20× slower at k≥6. */
    if (ems.type == "GM") {
        if (!ems.distname.empty() && ems.distname != "Gaussian") {
            cerr << "WARNING: -g flag implies Gaussian but -d " << ems.distname
                 << " specified. Using " << ems.distname << endl;
        } else {
            ems.distname = "Gaussian";
        }
        /* Fall through to UnmixGeneric dispatch below (line ~611) */
    } else if (ems.type == "EM") {
        if (ems.verbose){
            cout << "INFO: User Settings - Running Gemmule in Univariate Exponential Deconvolution Mode, reading univariate normal values. " << endl;
            cout << "INFO: File IO - " << to_string(umv.size()) << " values read from file. " << endl;
        }

        EMConfig_t EMConfig;
        EMResultExponential_t Result;

        MakeEMConfig(&EMConfig, &ems);
        UnmixExponentials(umv.data(), umv.size(), ems.kmixt, &Result, &EMConfig);
        //struct exponentialEMResults eer = unmixexponentials(umv, ems.kmixt, ems.maxitr, ems.verbose, ems.rtole);
        if (ems.verbose){
            cout << "INFO: EM Algorithm - Exponentials Unmixed in " << to_string(Result.iterstaken) << " Iterations of EM. " << endl;
        }
        ofstream ofile(ems.ofilename);
        string oline;
        for (int i = 0; i < Result.numExponentials; i++){
            oline = to_string(Result.means_final[i]) + ","  + to_string(Result.probs_final[i]);
            ofile << oline << endl;
        }
        if (ems.termcat){

            for (int i = 0; i < Result.numExponentials; i++) {
                oline = to_string(Result.means_final[i])  + "," + to_string(Result.probs_final[i]);
                cout << "INFO:  Results - " << oline << endl;
            }
        }
        ofile.close();
        cout << "INFO: File IO - Output written on " << ems.ofilename << endl;
        ReleaseEMResultExponential(&Result);
    }

    /* ================================================================
     * Generic distribution mode (-d DIST or --auto)
     *
     * -d gaussian/exponential/gamma/lognormal/weibull/beta/poisson/uniform
     *    Uses the generic EM engine with that family
     *
     * --auto
     *    Tries all valid families x k_min..k_max, picks best by BIC
     * ================================================================ */
    /* ════════════════════════════════════════════════════════════════
     * COMPLEX-VALUED GAUSSIAN MIXTURE
     * ════════════════════════════════════════════════════════════════ */
    if (ems.complex_circular || ems.complex_noncircular || ems.complex_mv || ems.complex_streaming) {
        string datafile = ems.valfilename;
        if (datafile.empty()) {
            cerr << "ERROR: Complex mode requires -g <file> (interleaved re,im pairs)" << endl;
            return 1;
        }
        /* Read interleaved re,im pairs: each line = "re im" or "re,im" */
        vector<double> flat;
        ifstream cf(datafile);
        string line;
        size_t n_complex = 0;
        while (getline(cf, line)) {
            if (line.empty() || line[0] == '#') continue;
            istringstream iss(line);
            string tok;
            vector<double> row;
            while (getline(iss, tok, line.find(',') != string::npos ? ',' : ' ')) {
                if (!tok.empty()) {
                    try { row.push_back(stod(tok)); } catch (...) {}
                }
            }
            if (row.size() >= 2) {
                flat.push_back(row[0]);
                flat.push_back(row[1]);
                n_complex++;
            }
        }
        if (n_complex == 0) {
            cerr << "ERROR: No valid complex data found in " << datafile << endl;
            return 1;
        }

        int rc;

        if (ems.complex_streaming) {
            /* Streaming complex EM — write data to temp file if needed */
            cout << "INFO: Streaming complex circular EM — file=" << datafile
                 << " k=" << ems.kmixt << endl;
            ComplexStreamConfig sconf;
            memset(&sconf, 0, sizeof(sconf));
            sconf.num_components = ems.kmixt;
            sconf.chunk_size = ems.stream_chunk;
            sconf.max_passes = ems.stream_passes;
            sconf.rtole = ems.rtole;
            sconf.verbose = ems.verbose ? 1 : 0;
            sconf.eta_decay = 0.6;
            sconf.type = CGAUSS_CIRCULAR;
            CCircMixtureResult cr;
            memset(&cr, 0, sizeof(cr));
            rc = UnmixComplexStreaming(datafile.c_str(), &sconf, &cr);
            if (rc == 0) {
                cout << "INFO: Streaming EM finished (" << cr.iterations << " steps)" << endl;
                cout << "INFO: LL=" << cr.loglikelihood
                     << "  BIC=" << cr.bic << "  AIC=" << cr.aic << endl << endl;
                for (int j = 0; j < cr.num_components; j++) {
                    cout << "Component " << j << ": weight=" << cr.mixing_weights[j] << endl;
                    cout << "  mean: " << cr.components[j].mu_re
                         << " + " << cr.components[j].mu_im << "i" << endl;
                    cout << "  var:  " << cr.components[j].var << endl;
                }
            }
            ReleaseCCircResult(&cr);
        } else if (ems.complex_mv) {
            /* Multivariate complex Gaussian — data has d complex dims per line */
            int d = ems.complex_mv_dim;
            /* Recount: flat has n_complex × 2 doubles, but for MV we need n × 2d */
            /* Re-read the file to handle multi-dim data */
            vector<double> mv_flat;
            size_t n_mv = 0;
            {
                ifstream mvf(datafile);
                string line;
                while (getline(mvf, line)) {
                    if (line.empty() || line[0] == '#') continue;
                    istringstream iss(line);
                    string tok;
                    vector<double> row;
                    while (getline(iss, tok, line.find(',') != string::npos ? ',' : ' ')) {
                        if (!tok.empty()) {
                            try { row.push_back(stod(tok)); } catch (...) {}
                        }
                    }
                    if ((int)row.size() >= 2*d) {
                        for (int dd = 0; dd < 2*d; dd++) mv_flat.push_back(row[dd]);
                        n_mv++;
                    }
                }
            }
            if (n_mv == 0) {
                cerr << "ERROR: No valid MV complex data (need " << 2*d << " values per line)" << endl;
                return 1;
            }
            cout << "INFO: Multivariate complex Gaussian — n=" << n_mv
                 << " d=" << d << " k=" << ems.kmixt << endl;
            MVComplexMixtureResult mvr;
            memset(&mvr, 0, sizeof(mvr));
            rc = UnmixMVComplex(mv_flat.data(), n_mv, d, ems.kmixt,
                                ems.maxitr, ems.rtole,
                                ems.verbose ? 1 : 0, &mvr);
            if (rc == 0) {
                cout << "INFO: Converged in " << mvr.iterations << " iterations" << endl;
                cout << "INFO: LL=" << mvr.loglikelihood
                     << "  BIC=" << mvr.bic << "  AIC=" << mvr.aic << endl << endl;
                for (int j = 0; j < mvr.num_components; j++) {
                    cout << "Component " << j << ": weight=" << mvr.mixing_weights[j] << endl;
                    cout << "  mean: [";
                    for (int dd = 0; dd < d; dd++) {
                        if (dd > 0) cout << ", ";
                        cout << mvr.components[j].mean[2*dd]
                             << "+" << mvr.components[j].mean[2*dd+1] << "i";
                    }
                    cout << "]" << endl;
                }
            }
            ReleaseMVComplexResult(&mvr);
        } else if (ems.complex_noncircular) {
            cout << "INFO: Non-circular complex Gaussian mixture — n=" << n_complex
                 << " k=" << ems.kmixt << endl;
            CNonCircMixtureResult ncr = {0};
            rc = UnmixComplexNonCircular(flat.data(), n_complex, ems.kmixt,
                                         ems.maxitr, ems.rtole,
                                         ems.verbose ? 1 : 0, &ncr);
            if (rc == 0) {
                cout << "INFO: Converged in " << ncr.iterations << " iterations" << endl;
                cout << "INFO: LL=" << ncr.loglikelihood
                     << "  BIC=" << ncr.bic << "  AIC=" << ncr.aic << endl << endl;
                for (int j = 0; j < ncr.num_components; j++) {
                    cout << "Component " << j << ": weight=" << ncr.mixing_weights[j] << endl;
                    cout << "  mean:    " << ncr.components[j].mu_re
                         << " + " << ncr.components[j].mu_im << "i" << endl;
                    cout << "  var:     " << ncr.components[j].cov_re << endl;
                    cout << "  pseudo:  " << ncr.components[j].pcov_re
                         << " + " << ncr.components[j].pcov_im << "i" << endl;
                }
            }
            ReleaseCNonCircResult(&ncr);
        } else if (ems.complex_autok) {
            int kmax = ems.kmax > 0 ? ems.kmax : 10;
            cout << "INFO: Complex circular auto-k — n=" << n_complex
                 << " k_max=" << kmax << endl;
            CCircMixtureResult cr = {0};
            rc = UnmixComplexCircularAutoK(flat.data(), n_complex, kmax,
                                            ems.maxitr, ems.rtole,
                                            ems.verbose ? 1 : 0, &cr);
            if (rc == 0) {
                cout << "INFO: Auto-k selected k=" << cr.num_components
                     << "  BIC=" << cr.bic << endl;
                cout << "INFO: LL=" << cr.loglikelihood
                     << "  AIC=" << cr.aic << endl << endl;
                for (int j = 0; j < cr.num_components; j++) {
                    cout << "Component " << j << ": weight=" << cr.mixing_weights[j] << endl;
                    cout << "  mean: " << cr.components[j].mu_re
                         << " + " << cr.components[j].mu_im << "i" << endl;
                    cout << "  var:  " << cr.components[j].var << endl;
                }
            }
            ReleaseCCircResult(&cr);
        } else {
            cout << "INFO: Circular complex Gaussian mixture — n=" << n_complex
                 << " k=" << ems.kmixt << endl;
            CCircMixtureResult cr = {0};
            rc = UnmixComplexCircular(flat.data(), n_complex, ems.kmixt,
                                      ems.maxitr, ems.rtole,
                                      ems.verbose ? 1 : 0, &cr);
            if (rc == 0) {
                cout << "INFO: Converged in " << cr.iterations << " iterations" << endl;
                cout << "INFO: LL=" << cr.loglikelihood
                     << "  BIC=" << cr.bic << "  AIC=" << cr.aic << endl << endl;
                for (int j = 0; j < cr.num_components; j++) {
                    cout << "Component " << j << ": weight=" << cr.mixing_weights[j] << endl;
                    cout << "  mean: " << cr.components[j].mu_re
                         << " + " << cr.components[j].mu_im << "i" << endl;
                    cout << "  var:  " << cr.components[j].var << endl;
                }
            }
            ReleaseCCircResult(&cr);
        }
        if (rc != 0) cerr << "ERROR: Complex EM failed (rc=" << rc << ")" << endl;
        return rc;
    }

    /* ════════════════════════════════════════════════════════════════
     * MULTIVARIATE GAUSSIAN MIXTURE
     * ════════════════════════════════════════════════════════════════ */
    if (ems.multivariate) {
        string datafile = ems.valfilename;
        if (datafile.empty()) {
            cerr << "ERROR: Multivariate mode requires -g <file> (row-major, space/comma-separated)" << endl;
            return 1;
        }
        /* Read n×d matrix: each line is one observation with d space/comma-separated values */
        vector<double> flat;
        int d_detected = -1;
        ifstream mvf(datafile);
        string line;
        size_t n_mv = 0;
        while (getline(mvf, line)) {
            if (line.empty() || line[0] == '#') continue;
            vector<double> row;
            istringstream iss(line);
            string tok;
            while (getline(iss, tok, line.find(',') != string::npos ? ',' : ' ')) {
                if (!tok.empty()) {
                    try { row.push_back(stod(tok)); } catch (...) {}
                }
            }
            if (row.empty()) continue;
            if (d_detected < 0) d_detected = (int)row.size();
            if ((int)row.size() != d_detected) continue;
            for (double v : row) flat.push_back(v);
            n_mv++;
        }
        if (ems.mv_dim > 1) d_detected = ems.mv_dim;
        if (n_mv == 0 || d_detected <= 0) {
            cerr << "ERROR: No valid multivariate data found in " << datafile << endl;
            return 1;
        }
        cout << "INFO: Multivariate Gaussian mixture — n=" << n_mv
             << " d=" << d_detected << " k=" << ems.kmixt
             << " cov=" << (ems.cov_type == COV_FULL ? "full" :
                            ems.cov_type == COV_DIAGONAL ? "diagonal" : "spherical") << endl;

        int rc;

        if (ems.mv_autok) {
            /* Auto-k selection */
            int kmax = ems.kmax > 0 ? ems.kmax : 8;
            MVAutoKResult akr;
            rc = UnmixMVAutoK(flat.data(), n_mv, d_detected, kmax,
                              ems.cov_type, ems.maxitr, ems.rtole,
                              ems.verbose ? 1 : 0, &akr);
            if (rc == 0) {
                cout << "INFO: Auto-k selected k=" << akr.best_k
                     << "  BIC=" << akr.best_bic << endl;
                MVMixtureResult& mr = akr.best_model;
                cout << "INFO: LL=" << mr.loglikelihood
                     << "  BIC=" << mr.bic
                     << "  AIC=" << mr.aic << endl << endl;
                for (int j = 0; j < mr.num_components; j++) {
                    cout << "Component " << j << ": weight=" << mr.mixing_weights[j] << endl;
                    cout << "  mean: [";
                    for (int dd = 0; dd < d_detected; dd++)
                        cout << (dd ? ", " : "") << mr.components[j].mean[dd];
                    cout << "]" << endl;
                }
            }
            ReleaseMVAutoKResult(&akr);
        } else if (ems.mv_studentt) {
            /* Student-t mixture */
            MVStudentTResult tr;
            rc = UnmixMVStudentT(flat.data(), n_mv, d_detected, ems.kmixt,
                                 ems.cov_type, ems.maxitr, ems.rtole,
                                 ems.verbose ? 1 : 0, &tr);
            if (rc == 0) {
                cout << "INFO: Converged in " << tr.iterations << " iterations" << endl;
                cout << "INFO: LL=" << tr.loglikelihood
                     << "  BIC=" << tr.bic << "  AIC=" << tr.aic << endl << endl;
                for (int j = 0; j < tr.num_components; j++) {
                    cout << "Component " << j << ": weight=" << tr.mixing_weights[j]
                         << "  nu=" << tr.components[j].nu << endl;
                    cout << "  mean: [";
                    for (int dd = 0; dd < d_detected; dd++)
                        cout << (dd ? ", " : "") << tr.components[j].mean[dd];
                    cout << "]" << endl;
                }
            }
            ReleaseMVStudentTResult(&tr);
        } else {
            /* Standard Gaussian */
            MVMixtureResult mv_result;
            rc = UnmixMVGaussian(flat.data(), n_mv, d_detected, ems.kmixt,
                                 ems.cov_type, ems.maxitr, ems.rtole,
                                 ems.verbose ? 1 : 0, &mv_result);
            if (rc == 0) {
                cout << "INFO: Converged in " << mv_result.iterations << " iterations" << endl;
                cout << "INFO: LL=" << mv_result.loglikelihood
                     << "  BIC=" << mv_result.bic
                     << "  AIC=" << mv_result.aic << endl << endl;
                for (int j = 0; j < mv_result.num_components; j++) {
                    cout << "Component " << j << ": weight=" << mv_result.mixing_weights[j] << endl;
                    cout << "  mean: [";
                    for (int dd = 0; dd < d_detected; dd++)
                        cout << (dd ? ", " : "") << mv_result.components[j].mean[dd];
                    cout << "]" << endl;
                    if (ems.verbose) {
                        cout << "  cov:" << endl;
                        for (int a = 0; a < d_detected; a++) {
                            cout << "    [";
                            for (int b = 0; b < d_detected; b++)
                                cout << (b ? ", " : "") << mv_result.components[j].cov[a*d_detected+b];
                            cout << "]" << endl;
                        }
                    }
                }
            }
            ReleaseMVMixtureResult(&mv_result);
        }
        if (rc != 0) cerr << "ERROR: Multivariate EM failed (rc=" << rc << ")" << endl;
        return rc;
    }

    /* ════════════════════════════════════════════════════════════════
     * STREAMING EM — file-based chunked processing
     * ════════════════════════════════════════════════════════════════ */
    if (ems.streaming) {
        string datafile = ems.valfilename;
        if (datafile.empty()) {
            cerr << "ERROR: Streaming mode requires -g <file>" << endl;
            return 1;
        }

        DistFamily fam = DIST_GAUSSIAN;
        if (!ems.distname.empty()) {
            string dn = ems.distname;
            for (auto& c : dn) c = tolower(c);
            if (dn == "gaussian" || dn == "normal") fam = DIST_GAUSSIAN;
            else if (dn == "exponential" || dn == "exp") fam = DIST_EXPONENTIAL;
            else if (dn == "gamma") fam = DIST_GAMMA;
            else if (dn == "laplace") fam = DIST_LAPLACE;
            else if (dn == "studentt" || dn == "t") fam = DIST_STUDENT_T;
        }

        StreamConfig scfg;
        scfg.num_components = ems.kmixt;
        scfg.chunk_size = ems.stream_chunk;
        scfg.max_passes = ems.stream_passes;
        scfg.rtole = ems.rtole;
        scfg.verbose = ems.verbose ? 1 : 0;
        scfg.family = fam;
        scfg.eta_decay = 0.6;

        cout << "INFO: Streaming EM — " << GetDistName(fam)
             << " k=" << ems.kmixt
             << " chunk=" << ems.stream_chunk
             << " passes=" << ems.stream_passes << endl;

        MixtureResult result;
        int rc = UnmixStreaming(datafile.c_str(), &scfg, &result);
        if (rc == 0) {
            cout << "INFO: LL=" << result.loglikelihood
                 << "  BIC=" << result.bic
                 << "  AIC=" << result.aic << endl << endl;
            const DistFunctions* df = GetDistFunctions(fam);
            for (int j = 0; j < result.num_components; j++) {
                cout << "Component " << j << ": weight=" << result.mixing_weights[j];
                for (int p = 0; p < df->num_params; p++)
                    cout << "  p" << p << "=" << result.params[j].p[p];
                cout << endl;
            }
            /* Write CSV output */
            if (!ems.ofilename.empty()) {
                ofstream ofile(ems.ofilename);
                if (ofile.is_open()) {
                    ofile << "# Gemmule streaming EM output" << endl;
                    ofile << "# Family: " << df->name << ", k=" << result.num_components << endl;
                    ofile << "# LL=" << result.loglikelihood << "  BIC=" << result.bic << endl;
                    for (int j = 0; j < result.num_components; j++) {
                        ofile << result.mixing_weights[j];
                        for (int p = 0; p < df->num_params; p++)
                            ofile << "," << result.params[j].p[p];
                        ofile << endl;
                    }
                    ofile.close();
                }
            }
        } else {
            cerr << "ERROR: Streaming EM failed (rc=" << rc << ")" << endl;
        }
        ReleaseMixtureResult(&result);
        return rc;
    }

    if (!ems.distname.empty() || ems.autoselect || ems.adaptive) {
        if (umv.empty() && ems.valfilename.empty()) {
            /* Need a data file — try reading from -g/-e filename or reparse */
            cerr << "ERROR: Generic/auto mode requires data values (-g or -e input)." << endl;
            return 1;
        }
        /* If umv wasn't populated by prior modes, load it now */
        if (umv.empty() && !ems.valfilename.empty()) {
            umv = parsegmvinputfile(ems.valfilename);
        }

        int k_min = 1;
        int k_max = ems.kmax > 0 ? ems.kmax : ems.kmixt;
        if (k_max < k_min) k_max = k_min;

        if (ems.adaptive) {
            cout << "INFO: Adaptive mode — k-selection: " << GetKMethodName(ems.kmethod)
                 << ", discovering distribution families from data" << endl;

            AdaptiveResult aResult;
            int rc = UnmixAdaptiveEx(umv.data(), umv.size(),
                                     k_max, ems.maxitr, ems.rtole,
                                     ems.verbose ? 1 : 0, ems.kmethod, &aResult);
            if (rc == 0) {
                cout << endl;
                cout << "========================================" << endl;
                cout << "  Adaptive Result: k=" << aResult.num_components
                     << "  (method: " << GetKMethodName(ems.kmethod) << ")" << endl;
                cout << "  LL  = " << aResult.loglikelihood << endl;
                cout << "  BIC = " << aResult.bic << endl;
                cout << "  AIC = " << aResult.aic << endl;
                if (ems.kmethod == KMETHOD_ICL)
                    cout << "  ICL = " << aResult.icl << endl;
                cout << "========================================" << endl;
                cout << endl;

                for (int j = 0; j < aResult.num_components; j++) {
                    cout << "  Component " << j << ": "
                         << GetDistName(aResult.families[j])
                         << "  weight=" << aResult.mixing_weights[j];
                    const DistFunctions* df = GetDistFunctions(aResult.families[j]);
                    for (int p = 0; p < df->num_params; p++)
                        cout << "  p" << p << "=" << aResult.params[j].p[p];
                    cout << endl;
                }

                /* Write output file */
                ofstream of(ems.ofilename);
                of << "# Gemmule Adaptive Result" << endl;
                of << "# k=" << aResult.num_components
                   << " LL=" << aResult.loglikelihood
                   << " BIC=" << aResult.bic << endl;
                for (int j = 0; j < aResult.num_components; j++) {
                    of << GetDistName(aResult.families[j])
                       << "," << aResult.mixing_weights[j];
                    const DistFunctions* df = GetDistFunctions(aResult.families[j]);
                    for (int p = 0; p < df->num_params; p++)
                        of << "," << aResult.params[j].p[p];
                    of << endl;
                }
                of.close();
                cout << "INFO: Output written to " << ems.ofilename << endl;
            } else {
                cerr << "ERROR: Adaptive EM failed with code " << rc << endl;
            }
            ReleaseAdaptiveResult(&aResult);

        } else if (ems.autoselect) {
            cout << "INFO: Auto-selecting best distribution family and number of components" << endl;
            cout << "INFO: Testing k=" << k_min << " to k=" << k_max << endl;

            ModelSelectResult msResult;
            int rc = SelectBestMixture(umv.data(), umv.size(),
                                       NULL, 0,  /* try all valid families */
                                       k_min, k_max,
                                       ems.maxitr, ems.rtole, ems.verbose ? 1 : 0,
                                       &msResult);
            if (rc == 0) {
                cout << endl;
                cout << "========================================" << endl;
                cout << "  Best Model: " << GetDistName(msResult.best_family)
                     << " with k=" << msResult.best_k << endl;
                cout << "  BIC = " << msResult.best_bic << endl;
                cout << "========================================" << endl;
                cout << endl;

                /* Print all candidates sorted by BIC */
                cout << "All candidates:" << endl;
                for (int c = 0; c < msResult.num_candidates; c++) {
                    MixtureResult& mr = msResult.candidates[c];
                    cout << "  " << GetDistName(mr.family)
                         << " k=" << mr.num_components
                         << "  LL=" << mr.loglikelihood
                         << "  BIC=" << mr.bic
                         << "  AIC=" << mr.aic;
                    if (mr.family == msResult.best_family && mr.num_components == msResult.best_k)
                        cout << "  <-- BEST";
                    cout << endl;
                }

                /* Print best model parameters */
                for (int c = 0; c < msResult.num_candidates; c++) {
                    MixtureResult& mr = msResult.candidates[c];
                    if (mr.family == msResult.best_family && mr.num_components == msResult.best_k) {
                        cout << endl << "Best model parameters:" << endl;
                        const DistFunctions* df = GetDistFunctions(mr.family);
                        for (int j = 0; j < mr.num_components; j++) {
                            cout << "  Component " << j << ": weight=" << mr.mixing_weights[j];
                            for (int p = 0; p < df->num_params; p++)
                                cout << "  p" << p << "=" << mr.params[j].p[p];
                            cout << endl;
                        }
                        break;
                    }
                }
            }
            ReleaseModelSelectResult(&msResult);

        } else {
            /* Specific distribution */
            DistFamily fam = DIST_GAUSSIAN;
            string dn = ems.distname;
            /* Lowercase it */
            for (auto& c : dn) c = tolower(c);

            if (dn == "gaussian" || dn == "normal") fam = DIST_GAUSSIAN;
            else if (dn == "exponential" || dn == "exp") fam = DIST_EXPONENTIAL;
            else if (dn == "poisson") fam = DIST_POISSON;
            else if (dn == "gamma") fam = DIST_GAMMA;
            else if (dn == "lognormal" || dn == "lognorm") fam = DIST_LOGNORMAL;
            else if (dn == "weibull") fam = DIST_WEIBULL;
            else if (dn == "beta") fam = DIST_BETA;
            else if (dn == "uniform") fam = DIST_UNIFORM;
            else if (dn == "studentt" || dn == "t" || dn == "student-t") fam = DIST_STUDENT_T;
            else if (dn == "laplace") fam = DIST_LAPLACE;
            else if (dn == "cauchy") fam = DIST_CAUCHY;
            else if (dn == "invgaussian" || dn == "invgauss" || dn == "inversegaussian") fam = DIST_INVGAUSS;
            else if (dn == "rayleigh") fam = DIST_RAYLEIGH;
            else if (dn == "pareto") fam = DIST_PARETO;
            else if (dn == "logistic") fam = DIST_LOGISTIC;
            else if (dn == "gumbel") fam = DIST_GUMBEL;
            else if (dn == "skewnormal" || dn == "skewnorm") fam = DIST_SKEWNORMAL;
            else if (dn == "gengaussian" || dn == "gengauss" || dn == "generalizedgaussian") fam = DIST_GENGAUSS;
            else if (dn == "chisquared" || dn == "chisq" || dn == "chi2") fam = DIST_CHISQ;
            else if (dn == "f" || dn == "fdist") fam = DIST_F;
            else if (dn == "loglogistic" || dn == "loglogist") fam = DIST_LOGLOGISTIC;
            else if (dn == "nakagami") fam = DIST_NAKAGAMI;
            else if (dn == "levy") fam = DIST_LEVY;
            else if (dn == "gompertz") fam = DIST_GOMPERTZ;
            else if (dn == "burr") fam = DIST_BURR;
            else if (dn == "halfnormal" || dn == "halfnorm") fam = DIST_HALFNORMAL;
            else if (dn == "maxwell") fam = DIST_MAXWELL;
            else if (dn == "kumaraswamy" || dn == "kumar") fam = DIST_KUMARASWAMY;
            else if (dn == "triangular" || dn == "triangle") fam = DIST_TRIANGULAR;
            else if (dn == "binomial" || dn == "binom") fam = DIST_BINOMIAL;
            else if (dn == "negbinomial" || dn == "negbinom" || dn == "negativebinomial") fam = DIST_NEGBINOM;
            else if (dn == "geometric" || dn == "geom") fam = DIST_GEOMETRIC;
            else if (dn == "zipf") fam = DIST_ZIPF;
            else if (dn == "kde") fam = DIST_KDE;
            else {
                cerr << "ERROR: Unknown distribution '" << ems.distname << "'" << endl;
                cerr << "  Valid: gaussian, exponential, poisson, gamma, lognormal, weibull, beta, uniform," << endl;
                cerr << "         studentt, laplace, cauchy, invgaussian, rayleigh, pareto, logistic, gumbel," << endl;
                cerr << "         skewnormal, gengaussian, chisquared, f, loglogistic, nakagami, levy," << endl;
                cerr << "         gompertz, burr, halfnormal, maxwell, kumaraswamy, triangular," << endl;
                cerr << "         binomial, negbinomial, geometric, zipf, kde" << endl;
                return 1;
            }

            if (ems.online) {
                cout << "INFO: Online EM — " << GetDistName(fam) << " mixture with k=" << ems.kmixt
                     << ", batch_size=" << (ems.batch_size > 0 ? ems.batch_size : (int)fmin(256, umv.size()/4)) << endl;
            } else {
                cout << "INFO: Fitting " << GetDistName(fam) << " mixture with k=" << ems.kmixt << endl;
            }

            MixtureResult result;
            int rc;
            if (ems.online) {
                rc = UnmixOnline(umv.data(), umv.size(), fam, ems.kmixt,
                                 ems.maxitr, ems.rtole, ems.batch_size,
                                 ems.verbose ? 1 : 0, &result);
            } else {
                rc = UnmixGeneric(umv.data(), umv.size(), fam, ems.kmixt,
                                  ems.maxitr, ems.rtole, ems.verbose ? 1 : 0, &result);
            }
            if (rc == 0) {
                cout << "INFO: Converged in " << result.iterations << " iterations" << endl;
                cout << "INFO: LL=" << result.loglikelihood
                     << "  BIC=" << result.bic
                     << "  AIC=" << result.aic << endl;
                cout << endl;

                const DistFunctions* df = GetDistFunctions(fam);
                for (int j = 0; j < result.num_components; j++) {
                    cout << "Component " << j << ": weight=" << result.mixing_weights[j];
                    for (int p = 0; p < df->num_params; p++)
                        cout << "  p" << p << "=" << result.params[j].p[p];
                    cout << endl;
                }

                /* Write to file */
                ofstream ofile(ems.ofilename);
                ofile << "# " << GetDistName(fam) << " mixture, k=" << result.num_components << endl;
                ofile << "# LL=" << result.loglikelihood << " BIC=" << result.bic << endl;
                for (int j = 0; j < result.num_components; j++) {
                    ofile << result.mixing_weights[j];
                    for (int p = 0; p < df->num_params; p++)
                        ofile << "," << result.params[j].p[p];
                    ofile << endl;
                }
                ofile.close();
                cout << "INFO: Output written on " << ems.ofilename << endl;
            }
            ReleaseMixtureResult(&result);
        }
    }
}

void writescreenres(vector<double>& abdn){
    for (int i = 0; i < abdn.size(); i++){
        cout << to_string(abdn[i]) << endl;
    }
}

void writeemres(string ofilename, vector<double>& abdn){
    ofstream ofile(ofilename);
    for (int i = 0; i < abdn.size(); i++){
        ofile << to_string(abdn[i]) << endl;
    }
    // Note, this output should not be masked by verbosity
    cout << "INFO: Output written on " << ofilename << endl;
    ofile.close();
}
struct comppatcounts parseinputfile(string ifilename){
    ifstream ifile(ifilename);
    string linebuf; 
    string curpatt; 
    string curcounts;
    struct comppatcounts cts; 
    int curcount;
    if (ifile.is_open()){
        while (ifile.good()){
            getline(ifile,curpatt,',');
            if(ifile.eof()){
                break;
            }
            getline(ifile,curcounts);
            if(ifile.eof()){
                break;
            }
            curcount = stoi(curcounts);
            cts.compatibilityPattern.push_back(curpatt);
            cts.count.push_back(curcount);
            if(ifile.eof()){
                break;
            }
        }
    }
    return(cts);
}

vector<double> parsegmvinputfile(string valfilename){
    vector<double> gmv;
    ifstream ifs(valfilename);
    double d;
    while(!ifs.eof()){
        ifs >> d; 
        if (ifs.eof()){
            return(gmv);
        }
        gmv.push_back(d);
    }
    return(gmv);
}

struct genesamfile readgenesamintostruct(string samfilename){
    ifstream gsam(samfilename);
    string linebuf;
    stringstream linebufstream;
    struct genesamfile gsf; 
    
    // for parsing the '@' lines
    string curseqname;
    string discard; 
    string curseqlen; 
    string curseqtags;

    // for parsing the non-'@' lines
    string curreadname; 
    string curreadflag;
    string curreadchralgn;
    string curreadchrpos; 
    string curreadmapq;
    string curreadcigar;
    string curreadpairchralgn;
    string curreadpairchrpos;
    string curreadobstemplatelen;
    string curreadseq; 
    string curreadqphred; 
    // For parsing the tags at the end of the SAM line.
    string curreadtags;
    string curreadtg;
    vector<string> curreadtaghead; 
    vector<string> curreadtagfield;

    if (gsam.is_open()){
        while(!gsam.eof()){
            curreadtags = "";
            curreadtaghead.clear();
            curreadtagfield.clear();
            getline(gsam,linebuf);
            linebufstream = stringstream(linebuf);
            if (linebuf[0] == '@'){
                if (linebuf[1] == 'H'){
                    continue;
                } else {
                    assert(linebuf[1] == 'S');
                    linebufstream >> discard; 
                    linebufstream >> curseqname; 
                    linebufstream >> curseqlen;
                    linebufstream >> curseqtags;
                }
                gsf.seqname.push_back(curseqname.substr(3,curseqname.size()-3));
                gsf.seqlen.push_back(stoi(curseqlen.substr(3,curseqlen.size()-3)));
                gsf.flags.push_back(curseqtags);//.substr(3,curseqtags.size()-3));
            } else {
                linebufstream >> curreadname;
                linebufstream >> curreadflag;
                linebufstream >> curreadchralgn;
                linebufstream >> curreadchrpos;
                linebufstream >> curreadmapq;
                linebufstream >> curreadcigar;
                linebufstream >> curreadpairchralgn;
                linebufstream >> curreadpairchrpos;
                linebufstream >> curreadobstemplatelen;
                linebufstream >> curreadseq; 
                linebufstream >> curreadqphred; 
                while(!linebufstream.eof()){
                    linebufstream >> curreadtg;
                    curreadtags = curreadtags+" "+curreadtg;
                }
                gsf.readnames.push_back(curreadname);
                gsf.readflags.push_back(stoi(curreadflag));
                gsf.primaryalignmentchr.push_back(curreadchralgn);
                gsf.primaryalignmentpos.push_back(stoi(curreadchrpos));
                gsf.MAPQ.push_back(stoi(curreadmapq)); 
                gsf.CIGAR.push_back(curreadcigar);
                gsf.pairrefname.push_back(curreadpairchralgn);
                gsf.pairpos.push_back(stoi(curreadpairchrpos));
                gsf.templatelength.push_back(stoi(curreadobstemplatelen));
                gsf.sequences.push_back(curreadseq);
                gsf.qualphred.push_back(curreadqphred);
                linebufstream = stringstream(curreadtags);
                while(!linebufstream.eof()) {
                    linebufstream >> curreadtg; 
                    curreadtaghead.push_back(curreadtg.substr(0,5));
                    curreadtagfield.push_back(curreadtg.substr(5,curreadtg.size()-6));
                }
                gsf.tagname.push_back(curreadtaghead); 
                gsf.tagval.push_back(curreadtagfield);
            }
        }
    }
    return(gsf);
}

struct comppatcounts parseinputgenesam(string samfilename){
    struct genesamfile gsf = readgenesamintostruct(samfilename);
    struct comppatcounts cpc; 
    string curpat;
    vector<string> pats;
    vector<int> counts;
    vector<string> alltransnames;
    stringstream curtagval;
    string curtagstring;
    stringstream curtagstream;
    string curtransname;
    for (int i = 0; i < gsf.seqname.size(); i++) {
        cout << gsf.flags[i][0] << endl;
        if (gsf.flags[i][0] == 'G'){
        alltransnames.push_back(gsf.seqname[i]);
        }
    }
    vector<string> curcompattransnames;
    for (size_t j = 0; j < gsf.readnames.size(); j++){
        curpat = "";
        curcompattransnames.clear();
        for (size_t k = 0; k < gsf.tagname[j].size(); k++){
            if (gsf.tagname[j][k] == "TI:Z:"){
                curtagval = stringstream(gsf.tagval[j][k]);
                getline(curtagval,curtransname,'-');
                curcompattransnames.push_back(curtransname);
            } else if (gsf.tagname[j][k] == "TO:Z:"){
                curtagval = stringstream(gsf.tagval[j][k]);
                while(!curtagval.eof()){
                    getline(curtagval,curtagstring,'|');
                    curtagstream = stringstream(curtagstring);
                    getline(curtagstream,curtransname,'-');
                    curcompattransnames.push_back(curtransname);
                }
            }
        }
        for (size_t i = 0; i < alltransnames.size(); i++){
            curpat += '0';
        }
        for (size_t m = 0; m < curcompattransnames.size(); m++){
            for (size_t i = 0; i < alltransnames.size(); i++){
                if (curcompattransnames[m] == alltransnames[i]){
                    curpat[i] = '1';
                }
           }
        }
        pats.push_back(curpat);
     }

    /* Aggregate patterns into unique patterns with counts */
    for (size_t i = 0; i < pats.size(); i++){
        bool found = false;
        for (size_t j = 0; j < cpc.compatibilityPattern.size(); j++){
            if (cpc.compatibilityPattern[j] == pats[i]){
                cpc.count[j]++;
                found = true;
                break;
            }
        }
        if (!found){
            cpc.compatibilityPattern.push_back(pats[i]);
            cpc.count.push_back(1);
        }
    }

    return(cpc);
}

void MakeEMConfig(EMConfig_t* ConfigPtr, struct emsettings* ems)
{
    if (ConfigPtr == NULL || ems == NULL) {
        return;
    }
    ConfigPtr->verbose = ems->verbose;
    ConfigPtr->maxiter = ems->maxitr;
    ConfigPtr->rtole = ems->rtole;
    ConfigPtr->init_method = ems->kmeans_init ? EM_INIT_KMEANS : EM_INIT_RANDOM;
    ConfigPtr->seed = 0;
}
