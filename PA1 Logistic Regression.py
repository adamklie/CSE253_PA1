# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 22:43:07 2020

@author: james 

Utilizes functions from dataloader.py given in the PA1.zip file. Will utilize PCA...
"""
from os import listdir
import os, random, copy
from PIL import Image
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

''' 
list of face expressions (contempt, neutral are excluded) are:
1. anger
2. disgust
3. fear
4. happiness
5. sadness
6. surprise
'''

def load_data(data_dir="./aligned/"):
	""" Load all PNG images stored in your data directory into a list of NumPy
	arrays.

	Args:
		data_dir: The relative directory path to the CK+ image directory.
	Returns:
		images: A dictionary with keys as emotions and a list containing images associated with each key.
		cnt: A dictionary that stores the # of images in each emotion
	"""
	images = defaultdict(list)

	# Get the list of emotional directory:
	for e in listdir(data_dir):
		# excluding any non-directory files
		if not os.path.isdir(os.path.join(data_dir, e)):
			continue
		# Get the list of image file names
		all_files = listdir(os.path.join(data_dir, e))

		for file in all_files:
			# Load only image files as PIL images and convert to NumPy arrays
			if '.png' in file:
				img = Image.open(os.path.join(data_dir, e, file))
				images[e].append(np.array(img))

	print("Emotions: {} \n".format(list(images.keys())))

	cnt = defaultdict(int)
	for e in images.keys():
		print("{}: {} # of images".format(e, len(images[e])))
		cnt[e] = len(images[e])
	return images, cnt

def balanced_sampler(dataset, cnt, emotions):
	# this ensures everyone has the same balanced subset for model training, don't change this seed value
	random.seed(20)
	print("\nBalanced Set:")
	min_cnt = min([cnt[e] for e in emotions])
	balanced_subset = defaultdict(list)
	for e in emotions:
		balanced_subset[e] = copy.deepcopy(dataset[e])
		random.shuffle(balanced_subset[e])
		balanced_subset[e] = balanced_subset[e][:min_cnt]
		print('{}: {} # of images'.format(e, len(balanced_subset[e])))
	return balanced_subset

def display_face(img):
	""" Display the input image and optionally save as a PNG.

	Args:
		img: The NumPy array or image to display

	Returns: None
	"""
	# Convert img to PIL Image object (if it's an ndarray)
	if type(img) == np.ndarray:
		print("Converting from array to PIL Image")
		img = Image.fromarray(img)

	# Display the image
	img.show()


#Crossfold validation splitting: 
#INPUT: 1) A dictionary mapping two emotions to an equal number of classes 2) The number of k mutually exclusive sets
#OUTPUT: a dictionary of k mutually exclusive sets where each set contains roughly the same number of each category in each set 
def CrossFold(toBeSplit, howMany):
    toReturn = defaultdict(list)
    targets = defaultdict(list)
    total = len([val for k,v in toBeSplit.items() for val in v])
    midpoint = int((total/howMany)/2)
    remainder = int(total/howMany) - midpoint
    
    keys = [k for k in toBeSplit]
    first = [el for el in toBeSplit[keys[0]]] #images of class 1
    random.shuffle(first) #randomly shuffle images
    second = [el for el in toBeSplit[keys[1]]] #images of class 2 
    random.shuffle(second)
    
    indexOne = 0
    indexTwo = 0
    difference = total % howMany
    for i in range(howMany):
        if difference != 0:
            toReturn[i] = first[indexOne:indexOne+remainder] + second[indexTwo:indexTwo+remainder]
            targets[i] = [0]*remainder + [1]*remainder
            indexOne += remainder
            indexTwo += remainder
            difference = difference -1 
        else:
            if (i%2 == 0):
                toReturn[i] = first[indexOne:indexOne+midpoint] + second[indexTwo:indexTwo+remainder]
                targets[i] = [0]*midpoint + [1]*remainder
                indexOne += midpoint
                indexTwo += remainder
            else:
                toReturn[i] = first[indexOne:indexOne+remainder] + second[indexTwo:indexTwo+midpoint]
                targets[i] = [0]*remainder + [1]*midpoint
                indexOne += remainder
                indexTwo += midpoint
             
    if sum([len(v) for k,v in toReturn.items()]) != total:
        print("We have problems...")
        
    return toReturn, targets

def CheckMutuallyExclusive(areMutuallyExclusive):
    for k,v in areMutuallyExclusive.items():
        #within a fold:
        for i in range(len(v)):
            toBeEvaluated = v[i]
            for j in range(i+1, len(v)):
                if(np.array_equal(toBeEvaluated,v[j])):
                    print("Well this sucks I have redundancies within a fold...")
        #between folds:
        for key,value in areMutuallyExclusive.items():
            if k == key:
                continue
            else:
                for i in range(len(v)):
                    for j in range(len(value)):
                        if np.array_equal(v[i], value[j]):
                            print("Son of a.... there are redundancies between folds")
                    
    return None 

#PlaceHolder PCA function:
def PCAPlaceholder():
    return None 

#Function that takes folds and returns 2 lists of np arrays - one for each class --> Needed???
def SplitIntoClasses():
    return None

#LOGISTIC REGRESSION FUNCTIONS:

#INPUT: List of numpy arrays
#OUTPUT: list of flattened numpy arrays with a 1 to account for bias at the front 
def Transform(aSeriesOfUnfortunateEvents): #Flatten the  numpy arrays and add a bias term to the front
    countOlaf = list()
    for baudelaire in aSeriesOfUnfortunateEvents:
        countOlaf.append(np.concatenate((np.array([1]),baudelaire.flatten())))
    return countOlaf 


#INPUT: 1) Flattened images  2) targets 3) The current weights 
#OUPUT: A float representing the current loss
def CrossEntropy(battleMages, targetDog, w):
    if len(battleMages) != len(targetDog):
        print("Ugh what is going on?")
    theOneSumToRuleThemAll = 0
    for i in range(len(battleMages)):
        if targetDog[i] == 1:
            theOneSumToRuleThemAll += np.log(ComputeSigmoidFunction(np.dot(battleMages[i], w)))
        else:
            #print(w)
            #print((w * [-1]))
            theOneSumToRuleThemAll += np.log(ComputeSigmoidFunction(np.dot(battleMages[i], (w * -1))))
        
    return (-1* theOneSumToRuleThemAll)

#Batch gradient computation for each pixel
#INPUT: 1) Flattened images  2) targets 3) The current weights 
#OUPUT: gradient for each pixel as a list
def Gradient(faces, whereDoIBelong, weigh):
    if len (weigh) != len(faces[0]):
        print("just... why?")
        return None
    gradient = [0]*len(weigh)
    
    for i in range(len(faces)): #over all samples --> batch
        intermediate = ComputeSigmoidFunction(np.dot(faces[i],weigh))
        for j in range(len(faces[i])): 
            gradient[j] += (whereDoIBelong[i] - intermediate)*faces[i][j]
                                
    return np.array(gradient).reshape(len(gradient), 1)

def ComputeSigmoidFunction(power):
    if type(power) is np.ndarray:
        if len(power) > 1:
            print("um what?")
        power = power[0]
    return 1/(1 + np.exp(-1 * power))

    
if __name__ == "__main__":
    # The relative path to your image directory
    data_dir = "./aligned/"
    dataset, cnt = load_data(data_dir)
    # test with happiness and anger
    images = balanced_sampler(dataset, cnt, emotions=['happiness', 'fear'])
    
    splitEmUp,labels = CrossFold(images, 10)
    #print(labels)
    print([len(v) for k,v in splitEmUp.items()])
    CheckMutuallyExclusive(splitEmUp)
    #initialize the weights to 0
    temporaryWeights = [0]*(1 + splitEmUp[0][0].shape[0]*splitEmUp[0][0].shape[1])
    weights = np.array(temporaryWeights).reshape(len(temporaryWeights), 1) #column vector 
    
    inListForm = []
    targetsInListForm = []
    whichClassAmI = 0
    for k,v in images.items():
        temp = Transform(v)
        for vals in v:
            targetsInListForm.append(whichClassAmI)  
        for t in temp:
            inListForm.append(t)
        whichClassAmI += 1
        
    print("The loss is: " + str(CrossEntropy(inListForm, targetsInListForm, weights)))
    learningRate = 10**-10
    for i in range(50):
        weights = weights + learningRate*Gradient(inListForm, targetsInListForm, weights)
        #print("After one step the loss is: " + str(CrossEntropy(inListForm, targetsInListForm, weights)))
        print("new loss is: " + str(CrossEntropy(inListForm, targetsInListForm, weights)))
    #display_index = 0
    #display_face(images['anger'][display_index])
    #dimensions = images['anger'][0].shape
    #print((dimensions))
    #for k,v in images.items():
    #    for vals in v:
    #        if vals.shape != dimensions:
    #            print("error!")
                
    #print(images['anger'][1].shape)
    