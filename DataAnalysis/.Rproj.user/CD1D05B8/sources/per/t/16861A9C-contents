rm(list=ls())
library(here)
###       CelebA
  #Blur ----
  blur = read.csv(here("CelebA", "blurR.csv"), header = T)
  plot(blur$FID~blur$Iteration, xlab="Kernal", ylab="FID")
  cor(blur$Iteration,blur$FID)
  cor.test(blur$Iteration, blur$FID, method = "pearson")
  
  #noise ----
  noise = read.csv(here("CelebA", "noiseR.csv"), header = T)
  plot(noise$FID~noise$Iteration, xlab="Mean (stdev=0.1)", ylab="FID")
  cor(noise$Iteration,noise$FID)
  cor.test(noise$Iteration, noise$FID, method = "pearson")
  
  #saltandpepper ----
  saltandpepper = read.csv(here("CelebA", "saltandpepperR.csv"), header = T)
  plot(saltandpepper$FID~saltandpepper$Iteration, xlab="nNise", ylab="FID")
  cor(saltandpepper$Iteration,saltandpepper$FID)
  cor.test(saltandpepper$Iteration, saltandpepper$FID, method = "pearson")
  
  #blocks ----
  blocks = read.csv(here("CelebA", "blocksR.csv"), header = T)
  plot(blocks$FID~blocks$Iteration, xlab="nNise", ylab="FID")
  cor(blocks$Iteration,blocks$FID)
  cor.test(blocks$Iteration, blocks$FID, method = "pearson")

###       MED
  #Blur ----
  blur = read.csv(here("Med", "blurR.csv"), header = T)
  plot(blur$FID~blur$Iteration, xlab="Kernal", ylab="FID")
  cor(blur$Iteration,blur$FID)
  cor.test(blur$Iteration, blur$FID, method = "pearson")
  
  #noise ----
  noise = read.csv(here("Med", "noiseR.csv"), header = T)
  plot(noise$FID~noise$Iteration, xlab="Mean (stdev=0.1)", ylab="FID")
  cor(noise$Iteration,noise$FID)
  cor.test(noise$Iteration, noise$FID, method = "pearson")
  
  #saltandpepper ----
  saltandpepper = read.csv(here("Med", "saltandpepperR.csv"), header = T)
  plot(saltandpepper$FID~saltandpepper$Iteration, xlab="Noise", ylab="FID")
  cor(saltandpepper$Iteration,saltandpepper$FID)
  cor.test(saltandpepper$Iteration, saltandpepper$FID, method = "pearson")
  
  #blocks ----
  blocks = read.csv(here("Med", "blocksR.csv"), header = T)
  blocks <- blocks[order(blocks$Iteration), ]
  blocks
  blocks = head(blocks, -2)
  blocks
  plot(blocks$FID~blocks$Iteration, xlab="nNise", ylab="FID")
  cor(blocks$Iteration,blocks$FID)
  cor.test(blocks$Iteration, blocks$FID, method = "pearson")
  