# config.py

import os

root = os.getcwd()

#output directory for phase 1 data collection and cleaning
SAVE_DIRECTORY_PHASE1 = f'{root}/phase1_outputs/'
#output directory for phase 2 model training and validation
SAVE_DIRECTORY_PHASE2 = f'{root}/phase2_outputs/'
#output directory for phase 3 analysis and interpretation
SAVE_DIRECTORY_PHASE3 = f'{root}/phase3_outputs/'

#location of ABCD release 4.0 non-imaging data instruments (.txt files)
ABCD_DATA_PATH = f'{root}/ABCD_NonImaging/*.txt'

#location of abcd_data_dictionary, choices_coding_nda.3.0.csv, and choices_coding_nda.4.0.csv from https://github.com/ABCD-STUDY/analysis-nda
ANALYSIS_NDA_OUTPUTS = f'{root}/ABCD_RDS_4'

#geoplot data

#location of usa-states-census-2014.shp from https://github.com/joncutrer/geopandas-tutorial.git
GEO_DATA = f'{root}/geo_us_data'
