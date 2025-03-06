# Running the experiments for the "On the usefulness of the fit-on-test view on evaluating calibration of classifiers" ([link to article](https://doi.org/10.1007/s10994-024-06652-6))

## 0. Preliminary

To prepare the environment, use "environment.yml" as follows:

```
conda env create -f environment.yml
```

## 1. Experiments on the Synthetic, Pseudo-Real and Real Data

* The experiment code is divided into folders: "Experiments_Synthetic", "Experiments_Pseudo", and "Experiments_Real". 
* Read the "readme.md" in each folder to run experiments.


## 2. Preprocessing Data

The data is preprocessed separately in each folder.

## 3. Generating Figures and Tables for the article and supplementary

* The figure and table generation is done via notebooks, which are in the folder called "Tables_Figures"
* The generation of tables and figures of synthetic experiments in the main article and supplementary are done with notebooks that have the name "1. Synthetic - ...".
* The generation of tables and figures of pseudo-real experiments in the main article and supplementary are done with notebooks that have the name "2. Pseudo-real - ...".
* The generation of tables and figures of real experiments in the main article and supplementary are done with notebooks that have the name "3. Real - ...".
* The generation of tables and figures of other stuff in the main article and supplementary are done with notebooks that have the name "4. Other - ...".

## 4. Citation

```
Kängsepp, M., Valk, K. & Kull, M. On the usefulness of the fit-on-test view on evaluating calibration of classifiers. Mach Learn 114, 105 (2025). https://doi.org/10.1007/s10994-024-06652-6
```
