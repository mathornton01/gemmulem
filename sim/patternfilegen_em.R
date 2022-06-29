

# These are some utility functions for generating random composition patterns.
gen_random_comp_pattern_data <- function(numalloc,
  numcat, probcat=rep(1 / numcat, numcat), maxcompat=200,abdfile="",
  abdinfile="") {
    if (abdinfile != "") {
      probcat <- scan(file = abdinfile);
      if (length(probcat) != numcat){
        cat("ERROR:  Number of specified probabilities in input
             abundance file does not match number of categories");
        quit('n');
      }
    }
  if (abdfile != "") {
    write.table(probcat, file = abdfile);
  }
  return(table(sample(apply(Vectorize(rbinom,
    vectorize.args = c("prob"))(n = maxcompat, size = 1, prob = probcat), 1,
                            function(x) {
                              paste(x, collapse = "")
                              }),
                      size = numalloc,
                      replace = TRUE)))
}

write_random_comp_pattern_file <- function(numalloc,
  numcat, filename, probcat = NA, maxcompat=200,
  quote = FALSE, rnames=FALSE, cnames=FALSE, sp = ",",
  verbose = FALSE, abdfile = "", abdinfile = "") {
    if (is.na(probcat)) {
      probcat <- runif(numcat);
    }
    d <- gen_random_comp_pattern_data(numalloc, numcat, probcat, maxcompat,
                                      abdfile, abdinfile);
    d_df <- data.frame(comppat = names(d), patcount = as.vector(d));
  write.table(d_df, file = filename, quote = quote,
    row.names = rnames, col.names = cnames, sep = sp);
}

# A Utility Function for printing the usage for this R-Script.
print_usage <- function() {
  cat("                                               \n");
  cat("-------------------------------------------------\n");
  cat("|   Simulate EM Compatibility Pattern Files     |\n");
  cat("|                (HELP)                         |\n");
  cat("|                                               |\n");
  cat("| [-n/-N/--NREAD] - Specify how many objects    |\n");
  cat("| [-c/-C/--NCAT] - Specify how many categories  |\n");
  cat("| [-o/-O/--OFILE] - Name Output File            |\n");
  cat("| [-to/-TO/--TOFILE] - True Abundances Out File |\n");
  cat("| [-ti/-TI/--TIFILE] - True Abundances Out File |\n");
  cat("| [-m/-M/--MCOM] - Specify Max compat. pattern  |\n");
  cat("| [-h/-H/--HELP] - Display this help message    |\n");
  cat("-------------------------------------------------\n");
  cat("                                               \n");
  q("no");
}

# Below is the main part of the script which produces the simulated file.
args <- commandArgs(trailingOnly = TRUE);
nalloc <- 10e6;
ncat <- 20;
fns <- "ts.csv";
mcompat <- 10;
trueabdfile <- "";
trueabdinfile <- "";

if (length(args) > 0){
for (a in 1:length(args)) {
  ca <- args[a];
  if(ca == "-n" || ca == "-N" || ca == "--NREAD") {
    nalloc <- args[a + 1];
  } else if (ca == "-c" || ca == "-C" || ca == "--NCAT") {
    ncat <- args[a + 1];
  } else if (ca == "-o" || ca == "-O" || ca == "--OFILE") {
    fns <- args[a + 1];
  } else if (ca == "-m" || ca == "-M" || ca == "--MCOM") {
    mcompat <- args[a + 1];
  } else if (ca == "-h" || ca == "-H" || ca == "--HELP") {
    print_usage();
  } else if (ca == "-to" || ca == "-TO" || ca == "--TOFILE"){
    trueabdfile <- args[a + 1];
  } else if (ca == "-ti" || ca == "-TI" || ca == "--TIFILE"){
    trueabdinfile <- args[a + 1];
  }
}}

write_random_comp_pattern_file(numalloc = nalloc,
                           numcat = ncat,
                           filename = fns,
                           maxcompat = mcompat,
                           abdfile = trueabdfile,
                           abdinfile = trueabdinfile);
