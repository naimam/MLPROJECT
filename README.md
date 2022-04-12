# MLPROJECT

## HOW TO RUN THE CODE

1. Classification.py
      - Prerequisites: 
         -  Libraries to install:
          - pip install pandas
          - pip install numpy
          - pip install scikit-learn 
          - pip install -U imbalanced-learn
      - or if using anaconda:
           - conda install pandas
           - conda install numpy
           - conda install -c anaconda scikit-learn
           - conda install -c conda-forge imbalanced-learn
           
      1b. ANNFromScratch.py:
      
               - pip install pandas
               - pip install numpy
               - pip install scikit-learn 
               - pip install -U imbalanced-learn
               - python -m pip install -U matplotlib
     
     You will need to edit lines 16 to 34 in Classification.py with the respective the path of the files in your system or environment in order to run the file.
     
     You will also need to edit lines 14 to 16 in ANNFromScracth.py file with the respective path of the files in your system or environment to run the file without issue.

2. MissingValueEstimation.py:
   - Prerequisites: 
     -  Libraries to install:
          - pip install pandas
          - pip install numpy
          - pip install scikit-learn 
      - or if using anaconda:
          - conda install pandas
          - conda install numpy
          - conda install -c anaconda scikit-learn
    - On lines 7,8,9, change the path from 'input/MissingData1.txt' to the path of the MissingData files.
    - On line 37, you can change the path from 'MissingData1.txt' to a folder. e.g: 'results/MohamedMissingR...'
    - Then simply run the python file.

---
## C1. Classification

We implemented 2 algorithms learned in class which are Ann and decision tree, and we had better result with our decision tree from scratch implementation. While our Ann with SKlearn performed much better than  any other algorithm we mentioned here, so all are output files come from this algorithm.

### Testing And Improving performance

Techniques we used to improve perfomance and do our testing. 

ANN: Cross validation, mutiples hidden layers, trying out different hiddeen layers, Overfitting, balancing data(oversample/undersample), standarize data, Knn for missing values.

Decision Tree: cross validation, overfitting, knn for missing values, feature selection with low variance and correlation, use entropy and gini method to find information gain.



* Ann from Scratch
* Decision Tree from Scratch
* Ann with SKLearn
* Decision tree with SKLearn

### Dataset 1

<h4>Results:</h4>

* Ann from Scratch: 50% ~ 55% testing accuracy and 60% ~ 70% training accuracy
* Decision Tree from Scratch: 84% ~ 86% testing accruacy and 100% training accuracy
* Ann with SKLearn: ~98% testing accuracy and 100% training accuracy
* Decision tree with SkLearn: 98% testing accuracy and 100% training accuracy


### Dataset 2:

<h4>Results:</h4>

* Ann form Scratch: 10% testing accuracy, 20% training accuracy
* Decision tree from Scratch: 50% testing accuracy, 100% (Overfitting)
* Ann with Sklearn: 90% testing accuracy, 100% training accuracy
* Decision tree with sklearn: 67% testing accuracy with 100% trainning accuracy (Overfitting).

### Dataset 3:

<h4>Results:</h4>

* Ann from Scratch: DNP
* Decision tree from Scratch: ~34% testing accuracy, 36% testing accuracy
* Ann with SKlearn: 60% ~ 67% testing accuracy, 91% training accuracy(overfitting)
* Decision tree with Sklearn: 64% ~ 66% testing accuracy, 94% training accuracy (overfitting)

### Dataset 4:

<h4>Results:</h4>

* Ann from Scratch: DNP
* Decision tree from Scratch: ~58% testing accuracy, 65% training accuracy
* Ann with Sklearn: 91% testing accuracy with 100% training accuracy
* Decision tree with Sklearn: 81% testing accuracy with 100 training accuracy

### Dataset 5
<h4>Results:</h4>

* Ann from Scratch: DNP
* Decision tree from Scratch: 54% testing accuracy and 68% training accuracy
* Ann with sklearn: ~82% testing accuracy and ~96% training
* Decision tree w Sklearn: 81% testing accuracy and 100 % training accuracy

