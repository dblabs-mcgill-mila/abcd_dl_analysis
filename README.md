# abcd_dl_analysis

This repository contains all analyses associated with our paper "Deep learning reveals that multidimensional social status drives population variation in 11,875 US participant cohort"

Please cite the paper: 

## Resources

- The final model is saved in /phase2_outputs/cvae_model_final/ folder, architecture comparison figures are in the /phase2_outputs folder
- All main and supplementary tables and figures are saved in the /phase3_outputs folder

## Scripts

This study utilizes all non-imaging instruments as outlined in "Release Notes: Adolescent Brain Cognitive Development Study (ABCD) Data Release 4.0" under "Table of shared non-imaging instruments": https://nda.nih.gov/static/docs/NDA4.0ReleaseNotesABCD.pdf

(ABCD) Data Release 4.0 available from https://nda.nih.gov/abcd

To replicate analyses, download all non-imaging instruments from ABCD release 4.0 and create folder abcd_dl_analysis/ABCD_NonImaging to save all .txt files

Run analysis scripts in the following order:

1. phase1_data_collection.py
   - Merges all individual study files from ABCD release 4.0 non-imaging on the combination of 'subjectkey' and 'eventname' columns to create dataframe for subsequent analyses

2. phase1_data_cleaning.py
   - Performs steps outlined in Methods 'Data curation protocol’ to produce preprocessed dataframe for training CVAE deep learning architecture

3. phase2_model_training_validation.py
   - Carries out steps outlined in Methods section 'Design of deep learning model architecture' to evaluate several model architectures and compare performance against PCA
   
4. phase3_main_interpretation_analysis.py
   - Contains all analyses interpreting the trained CVAE model to produce main results

--

CVAE model in folder /trvaep adapted from https://github.com/theislab/trvaep

Abcd_data_dictionary.csv, choices_coding_nda.3.0.csv, and choices_coding_nda.4.0.csv created following process outlined by https://github.com/ABCD-STUDY/analysis-nda




