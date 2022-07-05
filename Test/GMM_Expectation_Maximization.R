# An R Code implementing the Expectation Maximization Procedure for 
#  Gaussian Mixture Models (GMM). 

## Function Definitions
# For performing the actual expectation maximization, we can usually start
#  in the case of the Gaussian Mixture Model by just randomly generating the initial
#  parameter values (mean and variance for each of k models) separately. 

gmm_expectation_maximization <- function(values,k,rtol=1e-5){
  require(rlist)
  mean_init <- sample(values,k); 
  var_init <- runif(k,gaussian.var.min,gaussian.var.max);
  
  mean_old <- mean_init; 
  mean_new <- mean_init; 
  
  var_old <- var_init; 
  var_new <- var_init; 
  
  err <- 1;
  iter <- 1; 
  while (err > rtol){
    cat(paste("Iteration ",iter, ": "));
    cat("means: ", mean_new, "\n"); 
    cat("Vars: ", var_new, "\n"); 
    cat("\n\n");
    prob.matrix <- list.rbind(lapply(1:k, 
                                     function(x){dnorm(values,
                                                       mean_old[x],
                                                       var_old[x])}));
    unlist(apply(prob.matrix,2,function(x){which.max(x)})) -> guessed.gaussian.id;
    unlist(lapply(1:k,function(x){ifelse(is.na(mean(values[guessed.gaussian.id == x])),0,mean(values[guessed.gaussian.id == x]))})) -> mean_new;
    unlist(lapply(1:k,function(x){ifelse(is.na(var(values[guessed.gaussian.id == x])),0,var(values[guessed.gaussian.id == x]))})) -> var_new; 
    err <- sum((mean_old-mean_new)^2);
    err <- err+sum((var_old-var_new)^2); 
    mean_old <- mean_new; 
    var_old <- var_new;
    iter <- iter+1; 
  }
  return(list(parameters=cbind(mean_new,var_new),labels=guessed.gaussian.id));
}

# Some settings that I will use to attempt to keep the rest of this code 
#  more general for potential simulation result generations. 

num.gaussians <- 3;
num.values.total <- 100000;

gaussian.mean.mean <- 0;
gaussian.mean.var <- 25;

gaussian.var.min <- 1;
gaussian.var.max <- 1.5;

# First fix probabilities for generating from that many normals. 

prob.gaussians <- runif(num.gaussians);
prob.gaussians <- prob.gaussians/sum(prob.gaussians);

prob.gaussians <- rep(1/num.gaussians,num.gaussians);

# Fix the Means and Variances for the gaussians


sampled.gaussian.ids <- sample(1:num.gaussians, size=num.values.total, 
                               prob=prob.gaussians, replace=TRUE);
gaussian.means <- rnorm(num.gaussians,gaussian.mean.mean,gaussian.mean.var);
gaussian.vars <- runif(num.gaussians, gaussian.var.min, gaussian.var.max);

as.numeric(names(table(sampled.gaussian.ids))) -> gid.indexes;
as.vector(table(sampled.gaussian.ids)) -> gid.counts;

unlist(lapply(gid.indexes, function(x) {
  rnorm(gid.counts[x],gaussian.means[x],gaussian.vars[x])
})) -> sampled.gaussian.values;

cbind(sampled.gaussian.ids,sampled.gaussian.values,gaussian.means[sampled.gaussian.ids],gaussian.vars[sampled.gaussian.ids]);

tst <- gmm_expectation_maximization(sampled.gaussian.values,3);
tst
gaussian.means
gaussian.vars
