CSE 253: Programming Assignment 1

In this .zip folder the following files relating to this assignment are included: 
1) PA1 Complete Logistic Regresion.py 
2) PA2 Complete Softmax Regression.py
3) dataloader.py
4) aligned
5) resized

The Logistic and Softmax regression files are standalone scripts, meaning they can be run individually to produce the outputs relevant to each implementation.
However, running each regression scripts requires one to have the following packages installed and loaded into one's environment: 
PIL 
NumPy
matplotlib
Pandas
seaboarn

Instructions For Running:
1) PA1 Complete Logistic Regresion.py:
- This file should NOT be run from the command line. Trying will result in the following error: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
- This file should be run inside the PA1 folder
- Instead this file should be run either by 1) Running the code within an IDE (for testing purposed we performed this using SPYDER) or 2) Evaluating inside a jupyter notebook (we have attached the equivalent of the .py file here as a .ipynb) (Recommended) 
- If running in an IDE we suggest the following code modifications for ease of output visualization (due to the splitting of outputs between text and plots making the outputs hard to track):
	- Comment out all but one line (or in the case of the learning rate plots comment out all lines in the main function above oneLearningRate = and after #Perform remainder... ). 
	- Each function call returns the training accuracy, minimum validation loss, normalized test loss and test set accuracy along with plots of validation accuracy and loss and of PC visualization. Our testing within 
	  Spyder resulted in the output returning all the values first and then all the plots making it hard to track which test set result maps to which plot.

- For convenience we have also attached a jupyter notebook as well with the name PA1 Complete Logistic Regression.ipynb

2) PA2 Complete Softmax Regression.py
- This file must be run in the same directory as the "aligned" directory with the aligned images in them
- It must also be run with the dataloader.py file in the same directory
- This file can be run within an IDE (testing completed in PyCharm) or the command line, assuming the correct packages have been installed and loaded into the environment
- We have also attached a Jupyter notebook for this file with the name PA1 Complete Softmax Regression.ipynb for convenience


 
