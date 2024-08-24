This directory contains all the code necessary to recapitulate the analyses
in Mai 2024. 

The data can be downloaded from here: https://doi.org/10.5281/zenodo.13346017
Unzip the download file. Note the full path to the unzipped files. 

Create a text file in the same directory as this README file called "path_to_downloaded_data". This text file should contain one line of text, which is the path to the unzipped downloaded files you noted above. All of the scripts will look for this text file and load the path from it, and that's how they know where to find the data they need. Without completing this step, you will get an error message that "path_to_downloaded_data" doesn't exist or that it doesn't contain a valid path. 

To generate the figures, run the codes in each subdirectory in sorted order
according to their filename. 


Requirements:
numpy
scipy
pandas
matplotlib
my (https://github.com/cxrodgers/my.git)

