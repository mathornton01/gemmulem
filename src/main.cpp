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

using namespace std;

struct emsettings{
    string ifilename; 
    string ofilename;
    string samfilename;
    string valfilename;
    string type;
    int kmixt;
    int maxitr;
    bool verbose = false; 
    bool termcat = false;
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
    cout << "                                                                           " << endl;
    cout << "---------------------------------------------------------------------------" << endl; 
    cout << "|               GEMMULEM (Version 1.1) Thornton & Park                    |" << endl;
    cout << "|                              (HELP)                                     |" << endl;
    cout << "|                                                                         |" << endl;  
    cout << "|[-i/-I/--IFILE] <input filename>: Specify Input (CSV: \"pattern,count\")   |" << endl; 
    cout << "|[-o/-O/--OFILE] <output filename>: Specify output (\"Proportions\")        |" << endl;
    cout << "|[-r/-R/--RTOLE] <relative tolerance>: Specify EM Stopping Criteria       |" << endl;
    // cout << "|[-s/-S/--SFILE] <input samfilename>: Specify either one of -i, -s, -g, -e|" << endl; // For now we will hide this option, it may return in Version 3.0.
    cout << "|[-g/-G/--GFILE] <input valfilename>: Specify either one of -i, -g, -e    |" << endl;
    cout << "|[-e/-E/--EFILE] <input valfilename>: Specify either one of -i, -g, -e    |" << endl;
    cout << "|[-v/-V/--VERBO] Specify display of status and info messages (verbosity)  |" << endl;
    cout << "|[-k/-K/--KMIXT] <number of mixture distributions>: with -g/e (default=3) |" << endl;
    cout << "|[-t/-T/--TERMI] Specify display of results in the terminal (not written) |" << endl;
    cout << "|[-c/-C/--CSEED] integer Seed for random number generator (used in init.) |" << endl;
    cout << "|[-m/-M/--MAXIT] <maximum number of EM iterations> a convergence criteria |" << endl;
    cout << "|                                                                         |" << endl;  
    cout << "|Note: Both -g/e and -i may not be simultaneously specified               |" << endl;  
    cout << "|                                                                         |" << endl; 
    cout << "|File Types and Modes of Running:                                         |" << endl; 
    cout << "|                                                                         |" << endl; 
    cout << "|   * '-i':  Input appears as a csv file with compatibility patterns and  |" << endl;
    cout << "|      counts.  Ex. 1011,40<ret>1100,45<ret>...                           |" << endl; 
    cout << "|           (runs in coarse multinomial mode)                             |" << endl;
    cout << "|                                                                         |" << endl; 
    cout << "|   * '-g':  Input appears as a list of values comming from a mixture of  |" << endl; 
    cout << "|      univariate gaussians. Ex. 23.51<ret>25.62<ret>40.06<ret>...        |" << endl; 
    cout << "|            (runs in Gaussian Mixture Deconvolution Mode)                |" << endl;
    cout << "|                                                                         |" << endl;     
    cout << "|   * '-e':  Input appears as a list of values comming from a mixture of  |" << endl; 
    cout << "|      univariate exponentials. Ex. 2.2<ret>1.3<ret>4.2<ret>...           |" << endl; 
    cout << "|            (runs in Exponential Mixture Deconvolution Mode)             |" << endl;
    cout << "---------------------------------------------------------------------------" << endl;
    cout << "                                                                           " << endl;
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
    ems.rtole = rtole; 
    ems.maxitr = maxitr;
    ems.verbose = verbose;
    ems.termcat =termcat; 
    return(ems); 
}

int main(int argc, char** argv)
{

    cout <<
         "                         (GEMMULEM)                         " << endl <<
         "      General Mixed Multinomial Expectation Maximization    " << endl <<
         "              Micah Thornton & Chanhee Park (2022)          " << endl <<
         "                        [Version 1.1]                       " << endl << endl;

    // Store the user settings for the EM algorithm in the ems structure. 
    struct emsettings ems = parseargs(argc, argv);

    if (ems.verbose){
        cout << "INFO: User Settings - Running GEMMULEM in Verbose Mode (-v)" << endl;
        if (ems.ifilename != ""){
            if (ifstream(ems.ifilename).is_open()){
                cout << "INFO: User Settings - Running GEMMULEM in Multinomial De-Coarsening Mode, reading compatibility count input. " << endl;
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
    } else if (ems.type == "GM") {
        if (ems.verbose){
            cout << "INFO: User Settings - Running GEMMULEM in Univariate Gaussian Deconvolution Mode, reading univariate normal values. " << endl;
            cout << "INFO: File IO - " << to_string(umv.size()) << " values read from file. " << endl;
        }

        EMConfig_t EMConfig;
        EMResultGaussian_t Result;

        MakeEMConfig(&EMConfig, &ems);
        UnmixGaussians(umv.data(), umv.size(), ems.kmixt, &Result, &EMConfig);

        //struct gaussianemresults ger = unmixgaussians(umv, ems.kmixt, ems.maxitr, ems.verbose,ems.rtole);
        if (ems.verbose){
            cout << "INFO: EM Algorithm - Gaussians Unmixed in " << to_string(Result.iterstaken) << " Iterations of EM. " << endl;
        }
        ofstream ofile(ems.ofilename);
        string oline;
        for (int i = 0; i < Result.numGaussians; i++){
            oline = to_string(Result.means_final[i]) + "," + to_string(Result.vars_final[i]) + "," + to_string(Result.probs_final[i]);
            ofile << oline << endl;
        }
        if (ems.termcat){

            for (int i = 0; i < Result.numGaussians; i++){
                oline = to_string(Result.means_final[i]) + "," + to_string(Result.vars_final[i]) + "," + to_string(Result.probs_final[i]);
                cout << "INFO:  Results - " << oline << endl;
            }
        }
        ofile.close();
        cout << "INFO: File IO - Output written on " << ems.ofilename << endl;
        ReleaseEMResultGaussian(&Result);

    } else if (ems.type == "EM") {
        if (ems.verbose){
            cout << "INFO: User Settings - Running GEMMULEM in Univariate Exponential Deconvolution Mode, reading univariate normal values. " << endl;
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
    for (int j = 0; j < gsf.readnames.size(); j++){
        curpat = "";
        curcompattransnames.clear();
        for (int k = 0; k < gsf.tagname[j].size(); k++){
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
        for (int i = 0; i < alltransnames.size(); i++){
            curpat += '0';
        }
        for (int j = 0; j < curcompattransnames.size(); j++){
            for (int i = 0; i < alltransnames.size(); i++){
                if (curcompattransnames[j] == alltransnames[i]){
                    curpat[i] = '1';
                }
           }
        }
        cout << curpat << endl;
        pats.push_back(curpat);
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
}
