library(M3Drop)

b <- read.csv('ya.csv',header = F)
Normalized_data <- M3DropCleanData(b, 
                                   is.counts=TRUE, 
                                   min_detected_genes=70)
#the min_detected_genes in each dataset is different,because of different expression values
c <- Normalized_data$data
c <- t(c)
write.csv(c,'yan_m3.csv',row.names = F, col.names = F)