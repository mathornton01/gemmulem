# Gaussian mixture modeling deconvolution (three parameter version)

ng <- 5; 
ml <- -25;
mu <- 25;
sl <- 1;
su <- 5;
no <- 100000;
min.avg.mean.dist <- 15;

m.tr <- runif(ng,ml,mu);
while (mean(dist(m.tr)) <= min.avg.mean.dist){
  m.tr <- runif(ng,ml,mu);
}
s.tr <- runif(ng,sl,su);
p.tr <- runif(ng); p.tr/sum(p.tr) -> p.tr;
p.tr <- (rep(1/ng,ng))

Z <- sample(1:ng,no,replace=TRUE,prob=p.tr);
X <- rnorm(no,mean=m.tr[Z],sd=s.tr[Z]);

# The first step of the EM is to initialize the random gaussians based on k.

em <- function(X,k,v=FALSE,max.iter = 1000){
  require(rlist);
  
  # Initialization values for random guesses:
  #m.i <- runif(k,min(X),max(X)); m.p <- m.i; m.c <- m.p;
  #s.i <- runif(k,0.9*var(X),1.1*var(X)); s.p <- s.i; s.c <- s.p;
  #p.i <- rep(1/k,k); p.p <- p.i; p.c <- p.p;
  K <- kmeans(X,k,nstart=10)[[1]];
  list.rbind(lapply(1:k, function(x) {X[K==x] -> classobs; 
                                         c(length(classobs)/length(X),mean(classobs),sd(classobs))})) -> kmparm;
  m.i <- kmparm[,2];m.p <- m.i; m.c <- m.p;
  s.i <- kmparm[,3];s.p <- s.i; s.c <- s.p;
  p.i <- kmparm[,1];p.p <- p.i; p.c <- p.p;
  p.v <- FALSE; m.v <- FALSE; s.v <- FALSE;
  m.e <- NA; s.e <- NA; p.e <- NA;
  iter <- 1;
  # If anything has not converged yet.
  while(any(!p.v,!m.v,!s.v) & iter <= max.iter)
  {
    if(v){
      cat(paste("Iteration ", iter, ": \n", "| mean = ", m.c, " (error = ", m.e, " ) \n| var = ", s.c^2, " (error = ", s.e, " ) \n| prob = ", p.c," (error = ", p.e, "\n\n"));
    }
    lh <- list.cbind(lapply(1:k, function(x) {dnorm(X,m.c[x],s.c[x])}))
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
    
    s.c <- sqrt(colSums(lh*list.cbind(lapply(m.c, function(x){(X-x)^2})))/colSums(lh));
    s.e <- sum((s.c-s.p)^2);
    s.p <- s.c;
    s.v <- (s.e <= 1e-6);
    iter <- iter+1;
  }
  classlh <- list.cbind(lapply(1:k, function(x){dnorm(X,m.c[x],s.c[x])}));
  classlh <- t(t(classlh)*p.c);
  classlh <- classlh/rowSums(classlh);
  unlist(apply(classlh,1,which.max)) -> classmem;
  return(list(cbind(p.c,m.c,s.c),
              classmem,
              iter));
}

M <- 5; 
lapply(1:M, function(x){
tst <- em(X,5)
list(tst[[3]],
tst[[1]])
}) -> res;
cbind(p.tr,m.tr,s.tr)
table(tst[[2]],Z) -> cmattab;

ctmp <- cmattab;

tidx <- c();

for (i in 1:ncol(cmattab)){
  which.max(ctmp[,i]) -> max;
  ctmp[max,] <- 0;
  tidx <- as.vector(c(tidx,max));
}

tst[[1]][tidx,]
cbind(p.tr,m.tr,s.tr);

em_error <- colSums((tst[[1]][tidx,]-cbind(p.tr,m.tr,s.tr))^2);

em_error
