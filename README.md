# scBKAP: a single-cell RNA-Seq data clustering pipeline

## Overview

scBKAP, the cornerstone of which is a single-cell bisecting K-means clustering method based on an autoencoder network and a dimensionality reduction model MPDR. Specially, scBKAP utilizes an autoencoder network to reconstruct gene expression values from scRNA-Seq data to alleviate the dropout issue, and the MPDR model composed of the M3Drop feature selection algorithm and the PHATE dimensionality reduction algorithm to reduce the dimensions of reconstructed data. The dimension-ality-reduced data are then fed into the bisecting K-means clustering algorithm to identify the clusters of cells.

## Requirement:

- `python = 3.6`
- `numpy = 1.16.3`
- `scipy = 1.4.1`
- `pandas = 0.25.3`
- `scikit-learn = 0.22.1`
- `tensorflow = 1.13.1`
- `matplotlib = 3.0.3`
- `R = 3.6`

## Dataset:

We only provide one single-cell RNA-seq dataset Ting, other datasets can be obtained from https://hemberg-lab.github.io/scRNA.seq.datasets/

## Usage:

### Input
The input of scBKAP should be a csv file (row: genes, col: cells).

### Run

1. Run the `genefilter.py` to filter the data， the `data` is the scRNA-seq data matrix:

```
filte('ting', 'ting_filter')
```

2. Reconstruct the data:

```
X_con = autorunner('ting_fitler', 1000, 800, 200, 'ting_auto')
```

3. Use the M3drop:

```
library(M3drop)
b <- read.csv('ting_auto.csv',header = F)
Normalized_data <- M3DropCleanData(b, 
                                   is.counts=TRUE, 
                                   min_detected_genes=112)
#the min_detected_genes in each dataset is different,because of different expression values
c <- Normalized_data$data
c <- t(c)
write.csv(c,'data_m3.csv',row.names = F, col.names = F)
```

4. clust the data:

```
y_pred = clust('ting_m3', 'ting_label', 20, 5)
```

### Output

- `ting_filter.csv` The filtered data.
- `ting_auto.csv` The reconstructed data.
- `ting_m3.csv` The data selected by M3drop.
- `y_pred` The predicted label by scBKAP.
