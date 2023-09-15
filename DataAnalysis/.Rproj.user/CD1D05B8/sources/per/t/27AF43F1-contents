rm(list=ls())
library(here)
#Read Vanilla Files ----
E1 = read.csv(here("VanillaRuns", "E1TTest.csv"), header = T)
E2 = read.csv(here("VanillaRuns", "E2TTest.csv"), header = T)
E3 = read.csv(here("VanillaRuns", "E3TTest.csv"), header = T)
E4 = read.csv(here("VanillaRuns", "E4TTest.csv"), header = T)
E5 = read.csv(here("VanillaRuns", "E5TTest.csv"), header = T)
#Read WGAN files ----
E1 = read.csv(here("WGANRuns", "E1TTest.csv"), header = T)
E2 = read.csv(here("WGANRuns", "E2TTest.csv"), header = T)
E3 = read.csv(here("WGANRuns", "E3TTest.csv"), header = T)
E4 = read.csv(here("WGANRuns", "E4TTest.csv"), header = T)
E5 = read.csv(here("WGANRuns", "E5TTest.csv"), header = T)

summaryStats <- function(data)
{
  mean1 = mean(data$FID)
  stdev1 = sd(data$FID)
  cat("mean1:", mean1, "stdev1:", stdev1)
}

#If we have raw data:
do_t_test <- function(dataset1, dataset2)
{
  result = t.test(dataset1$FID, dataset2$FID, conf.level = 0.95, paired=FALSE) #Welch's t.test
  print(result)
  return(result)
}

do_t_test(E1, E2)
do_t_test(E1, E3)
do_t_test(E1, E4)
do_t_test(E1, E5)
do_t_test(E2, E3)
do_t_test(E2, E4)
do_t_test(E2, E5)
do_t_test(E3, E4)
do_t_test(E3, E5)

plotWithError <- function(mean, sdev)
{
  x = 1:5
  plot(x, mean,
       ylim=range(c(mean-sdev, mean+sdev)),
       pch=19, xlab="Experiment", ylab="Average FID Score",
       main="MedFID Scores for WGAN-GP experiments"
  )
  # hack: we draw arrows but with very special "arrowheads"
  arrows(x, mean-sdev, x, mean+sdev, length=0.05, angle=90, code=3)
}

mean = c(mean(E1$FID),mean(E2$FID), mean(E3$FID), mean(E4$FID),mean(E5$FID))
sdev = c(sd(E1$FID),sd(E2$FID), sd(E3$FID), sd(E4$FID),sd(E5$FID))
plotWithError(mean, sdev)

#StyleGAN ----
Style = read.csv(here("StyleRuns", "StyleForT.csv"), header = T)
do_t_test(Style, E1)
do_t_test(Style, E2)
do_t_test(Style, E3)
do_t_test(Style, E4)
do_t_test(Style, E5)
#Trash ----

#If we only have summary statistics
t.test2 <- function(m1,m2,s1,s2,n1,n2,m0=0,equal.variance=FALSE)
{
  if( equal.variance==FALSE ) 
  {
    se <- sqrt( (s1^2/n1) + (s2^2/n2) )
    # welch-satterthwaite df
    df <- ( (s1^2/n1 + s2^2/n2)^2 )/( (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) )
  } else
  {
    # pooled standard deviation, scaled by the sample sizes
    se <- sqrt( (1/n1 + 1/n2) * ((n1-1)*s1^2 + (n2-1)*s2^2)/(n1+n2-2) ) 
    df <- n1+n2-2
  }      
  t <- (m1-m2-m0)/se 
  dat <- c(m1-m2, se, t, 2*pt(-abs(t),df))    
  names(dat) <- c("Difference of means", "Std Error", "t", "p-value")
  return(dat) 
}
tt2 <- t.test2(mean1, mean2, stdev1, stdev2, 10, 10)
tt2



