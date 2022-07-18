# EM demo

library(rgemmulem)

data <- read.csv("ts.csv", header=FALSE, colClasses = c("character", "numeric"));


compats <- unlist(data[1], use.names=FALSE)
counts <- as.integer(unlist(data[2], use.names=FALSE))

result <- rgemmulem_expectationmaximization(compats, counts)
