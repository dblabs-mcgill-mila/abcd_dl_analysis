# abcd_dl_analysis

This repository contains all analyses associated with our paper <>.

Please cite the paper: 

## Resources and Scripts

This study utilizes all non-imaging instruments as outlined in "Release Notes: Adolescent Brain Cognitive Development Study (ABCD) Data Release 4.0" under "Table of shared non-imaging instruments": https://nda.nih.gov/static/docs/NDA4.0ReleaseNotesABCD.pdf

(ABCD) Data Release 4.0 available from https://nda.nih.gov/abcd

After downloading all non-imaging instruments from ABCD release 4.0, analysis scripts can be run in the following order:

1. phase1_data_collection.py
2. phase1_data_cleaning.py
3. phase2_model_training_validation.py
4. phase3_main_interpretation_analysis.py

--

CVAE model in folder /trvaep adapted from https://github.com/theislab/trvaep (as utilized in Lotfollahi et al., 2020)

Geoplotting data in /geo_us_data from https://github.com/joncutrer/geopandas-tutorial.git

Abcd_data_dictionary.csv, choices_coding_nda.3.0.csv, and choices_coding_nda.4.0.csv created following process outlined by https://github.com/ABCD-STUDY/analysis-nda




