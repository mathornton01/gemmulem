# EM demo

library(rgemmulem)

data <- read.csv("gmm_tst_vals_1.txt", header = F)
values <- unlist(data[1], use.names = F)

result <- rgemmulem_unmixgaussians(values, 3)
