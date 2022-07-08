# Exponential mixture modeling deconvolution (two parameter version)

ne <- 2; 
ml <- 1;
mu <- 25;
no <- 100000;
mean.m.dist <- 10;

m.tr <- runif(ne,ml,mu);
while(mean(dist(m.tr)) <= mean.m.dist){
  m.tr <- runif(ne,ml,mu);
}
p.tr <- runif(ne); p.tr/sum(p.tr) -> p.tr;

Z <- sample(1:ne,no,replace=TRUE,prob=p.tr);
X <- rexp(no,rate=1/m.tr[Z]);

# The first step of the EM is to initialize the random gaussians based on k.

k <- 2;


em <- function(X,k,v=FALSE,max.iter = 1000){
  require(rlist);
  
  # Initialization values for random guesses:
  m.i <- runif(k,min(X),max(X)); m.p <- m.i; m.c <- m.p;
  p.i <- rep(1/k,k); p.p <- p.i; p.c <- p.p;
  p.v <- FALSE; m.v <- FALSE;
  m.e <- NA; p.e <- NA;
  iter <- 1;
  # If anything has not converged yet.
  while(any(!p.v,!m.v) & iter <= max.iter)
  {
    if(v){
      cat(paste("Iteration ", iter, ": \n", "| mean = ", m.c, " (error = ", m.e,  " ) \n| prob = ", p.c," (error = ", p.e, "\n\n"));
    }
    lh <- list.cbind(lapply(1:k, function(x) {dexp(X,rate=1/m.c[x])}))
    t(t(lh)*p.c) -> lh;
    lh <- lh/rowSums(lh);
    
    p.c <- colSums(lh)/sum(lh);
    p.e <- sum((p.c-p.p)^2);
    p.p <- p.c;
    p.v <- (p.e <= 1e-6);
    
    m.c <- colSums(lh*X)/colSums(lh);
    m.e <- sum((m.c-m.p)^2);
    m.p <- m.c;
    m.v <- (m.e <= 1e-6);
    
    iter <- iter+1;
  }
  classlh <- list.cbind(lapply(1:k, function(x){dexp(X,rate=1/m.c[x])}));
  classlh <- t(t(classlh)*p.c);
  classlh <- classlh/rowSums(classlh);
  unlist(apply(classlh,1,which.max)) -> classmem;
  return(list(cbind(p.c,m.c),
              classmem,
              iter));
}

tst <- em(X,2,max.iter=1e4)
tst[[3]]
tst[[1]];
cbind(p.tr,m.tr);

table(tst[[2]],Z) -> cmattab;

cmattab

ctmp <- cmattab;
tidx <- c();
for (i in 1:ncol(cmattab)){
  which.max(ctmp[,i]) -> max;
  ctmp[max,] <- 0;

  cat(paste(max, " <-> ", i,"\n"));
  tidx <- as.vector(c(tidx,max));
}

tidx
tst[[1]][tidx,]
cbind(p.tr,m.tr)
tidx
tst[[1]]
cmattab
