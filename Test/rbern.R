# This R-Code implements a version of the 
# Multivariate Bernoulli Distribution. 
#   Either Fixed Joint and Marginal Probabilities can be supplied 
#   Or dimensions can be provided, and the fixed joint and marginals
#   also simulated at random. It also just runs the test of the EM code.

library(rlist);
M <- 10; 
sss <- c(5,10,25,50,100,500,1000,5000,10000,20000,50000,100000,200000,500000,1000000);
css <- c(3,4,5,6,7,8,9,10,11,12,13,14,15);
Vectorize(rbinom,vectorize.args=c('prob')) -> vrbinom;


list.rbind(lapply(1:M, function(z) {
  list.rbind(lapply(sss,function(ss){
  list.rbind(lapply(css, function(cs){
num <- ss; 
size <- cs;
mp <- runif(size);
mp <- mp/sum(mp);
pp <- matrix(runif(size**2,0.7,0.75),size,size);
diag(pp) <- 1;
sample(1:size,num,prob=mp,replace = TRUE) -> sel;
unlist(lapply(sel, function(x) {paste(vrbinom(1,size=1,prob=pp[x,]),collapse="")})) -> pats;

mptst <- colSums(list.rbind(lapply(1:length(pats), function(x) {as.numeric(unlist(strsplit(pats[x],"")))})))/sum(colSums(list.rbind(lapply(1:length(pats), function(x) {as.numeric(unlist(strsplit(pats[x],"")))}))))
table(pats) -> tabdat; 
write.table(cbind(names(tabdat),as.vector(tabdat)),row.names=FALSE,col.names=FALSE,
            quote=FALSE,sep=",",file="1m_tst.tab")

system(paste("gemmulem -i","1m_tst.tab","-o","out.txt","-r","0.000001"),
       ignore.stdout = TRUE);
res <- scan("out.txt",quiet=TRUE);
c(z,cs,ss,sum(((mp-res)^2)))
#cat(paste(cs,",",ss,",",sum(((mp-res)^2)),"\n",sep=""));
}))}))})) -> simdata;
list.rbind(lapply(sss, function(ss) { list.rbind(lapply(css, function(cs) {c(cs,ss,mean(simdata[simdata[,2]==cs & simdata[,3]==ss,4]),sd(simdata[simdata[,2]==cs & simdata[,3]==ss,4]))}))})) -> simres;

png("simres_error.png",height=600,width=1000)
matplot(x=log(sss),
        t(list.rbind(lapply(css,function(cs){c(unlist(simres[simres[,1]==cs,c(3)]))}))),
        xlab="Sample Size [Log - Scale]",xaxt='n',
        ylab = "Mean Squared Error",
        main="GEMMULEM Quantification Error In  10 Replicates of \n Simulated Multivariate Bernoulli Trials with Fixed Marginal and Pair probabilities",
        col = rainbow(length(css)),
        lty=1:length(css),pch=1:length(css),type='b');
axis(1,at=log(sss),labels=sss,tick=TRUE)
legend(title="Number of Categories",'topright',legend=css,lty=1:length(css),pch=1:length(css),col=rainbow(length(css)))        
dev.off()

write.table(simres,"simres.csv");

#cbind(mp,mptst,res,abs(mp-res),abs(mptst-res));


