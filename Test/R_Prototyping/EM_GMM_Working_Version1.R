# Gaussian mixture modeling deconvolution (three parameter version)

ng <- 3; 
ml <- -25;
mu <- 25;
sl <- 1;
su <- 5;
no <- 10000;

m.tr <- runif(ng,ml,mu);
s.tr <- runif(ng,sl,su);
p.tr <- runif(ng); p.tr/sum(p.tr) -> p.tr;

Z <- sample(1:ng,no,replace=TRUE,prob=p.tr);
X <- rnorm(no,mean=m.tr[Z],sd=s.tr[Z]);

# The first step of the EM is to initialize the random gaussians based on k.

k <- 3;


em <- function(X,k,v=FALSE,max.iter = 1000){
  require(rlist);
  
  # Initialization values for random guesses:
  m.i <- runif(k,min(X),max(X)); m.p <- m.i; m.c <- m.p;
  s.i <- runif(k,0.9*var(X),1.1*var(X)); s.p <- s.i; s.c <- s.p;
  p.i <- rep(1/k,k); p.p <- p.i; p.c <- p.p;
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

tst <- em(X,4)
tst[[1]];
cbind(p.tr,m.tr,s.tr);

table(tst[[2]],Z) -> cmattab;

cmattab
