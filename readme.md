# Running the experiments for the "On the Usefulness of the Fit-on-the-Test View on Evaluating Calibration of Classifiers"

## 0. Preliminary

To prepare the environment, use "environment.yml" as following:

```
conda env create -f environment.yml
```

## 1. Experiments on the Synthetic, Pseudo-Real and Real Data

Experiments code is divided into folders: "Experiments_Synthetic", "Experiments_Pseudo" and "Experiments_Real". 
To run experiments read the "readme.md" from each folder.


## 2. Preprocessing Data

The preprocessing of data is done separately in each folder.

## 3. Generating Figures and Tables for the article and supplementary

The figure and table generation is done via notebooks, which are in the folder called "Tables_Figures"
The generation of tables and figures of synthetic experiments in the main article and supplementary are done with notebooks that have the name "1. Synthetic - ...".
The generation of tables and figures of pseudo-real experiments in the main article and supplementary are done with notebooks that have the name "2. Pseudo-real - ...".
The generation of tables and figures of real experiments in the main article and supplementary are done with notebooks that have the name "3. Real - ...".
The generation of tables and figures of other stuff in the main article and supplementary are done with notebooks that have the name "4. Other - ...".
