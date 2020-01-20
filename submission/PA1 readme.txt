CSE 253: Programming Assignment 1

In this submission folder the following files relating to this assignment are included: 
1) PA1 Complete Logistic Regresion.py - code for generating Log Reg plots
2) PA2 Complete Softmax Regression.py - code for generating Softmax Reg plots
3) dataloader.py - provided code with helper functions for loading data
4) PCA_sanity_check.py - Code to test PCA validity
4) aligned - directory containing aligned images
5) resized - directory containing resized images
6) PA1.yml - environment configuration file (see last lines)

The Logistic and Softmax regression files are standalone scripts, meaning they can be run individually to produce the outputs relevant to each implementation.
However, running all the above scripts requires one to have the following packages installed and loaded into one's environment: 

pillow 7.0.0
NumPy 1.17.3
matplotlib 3.1.2
pandas 0.25.3 
seaborn 1.4.1

All packages were installed using the conda package manager.

Instructions For Running: 
1) PA1 Complete Logistic Regresion.py***:
- `python PA1\ Complete\ Logistic\ Regression.py`
- This file must be run in the same directory as the "aligned" and "resized" directories with the respective images in each
- It must also be run with the dataloader.py file in the same directory
- This file can be run within an IDE (testing completed in Spyder) or the command line***, assuming the correct packages have been installed and loaded into the environment
- We have also attached a Jupyter notebook for this file with the name PA1 Complete Logistic Regression.ipynb for convenience

***NOTE: PA1 Complete Logistic Regresion.py was implemented on a Windows OS. Though it will run on a MacOS, slightly different but similar results might be obtained than shown in report.
	 If need be, we are happy to show how we generated the plots actually used in the report.
	
2) PA2 Complete Softmax Regression.py
- `python PA1\ Complete\ Softmax\ Regression.py`
- This file must be run in the same directory as the "aligned" directory with the aligned images in them
- It must also be run with the dataloader.py file in the same directory
- This file can be run within an IDE (testing completed in PyCharm) or the command line, assuming the correct packages have been installed and loaded into the environment
- We have also attached a Jupyter notebook for this file with the name PA1 Complete Softmax Regression.ipynb for convenience

3) PCA_sanity_check.py
- `python PCA_sanity_check.py`
- This file must be run in the same directory as the "aligned" directory with the aligned images in them
- It must also be run with the dataloader.py file in the same directory
- This file can be run within an IDE (testing completed in PyCharm) or the command line, assuming the correct packages have been installed and loaded into the environment
- We have also attached a Jupyter notebook for this file with the name PCA_sanity_check.ipynb for convenience

NOTE: Easy environment set up using conda:
If you have conda installed you can run the following lines to set up your environment. Again, results may differ slightly for Logistic Regression as explained above
- `conda env create -f PA1.yml`
- `conda activate PA1`