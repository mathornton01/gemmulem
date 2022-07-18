# demo

library(mseed)

mseedFile <- system.file("extdata", "test.mseed", package="mseed")
rawData <- readBin(mseedFile, raw(), n=1e+6)

length(rawData)

rawData[1:10]

test_hello_mseed_mem(rawData)

