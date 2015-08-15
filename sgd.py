#!/usr/bin/python2.7
import numpy as np
import matplotlib.pyplot as pt
import random
import pylab

# get data from csv file and store in matrix 'data'
data = np.loadtxt('cancer.csv', delimiter=',')
data = data[:369,:]	# the problem only wants the 1st 369 rows

np.random.shuffle(data)
lmbdas = np.array([1e-3, 1e-2, 1e-1, 1])

a = np.zeros((9,))  # choose random a and b to start with
b = 0

validation_set = data[:100]
test_set = data[100:200]
training_set = data[200:]
N_e = 50
N_s = 100
errors_list = []
# steplength() calculates the steplength.
# ^ formula on the bottom of p.338
def steplength(e):
    return 1/(0.01*e+50) 

# new_a_b() calculates a_(n+1) and b_(n+1)
# @params x is feature vector, y is label, 
#   a is current a, b is current b,
#   lmda is the regularization weight, n is steplength
def new_ab(x,y,a,b,lmbda,n):
    ''' 
    SCALARS: y,lmda,n,b
    VECTORS: x,a
    formulas are in the middle of p.338
    '''
    if y*(np.dot(a,x)+b) >= 1:
        a_new = a - n*(lmbda*a)   # n = steplength
        b_new = b - 0
    else:
        a_new = a - n*(lmbda*a - y*x)
        b_new = b - n*(-y)
        
    return a_new, b_new

def classify(a,b,x):
    if int(np.dot(a,x)+b) > 0:
        return 1
    else:
        return -1
    
def determine_accuracy(a,b,validation_set):
    errors = np.zeros((100,))
    for i in range(100):
        x = validation_set[i][:9]
        guessed_class = classify(a,b,x)
        actual_class = validation_set[i][9]

        if actual_class == guessed_class:
            errors[i] = 1

    return np.mean(errors)
  
for l in range(len(lmbdas)):   # for each regularization weight
    lmbda = lmbdas[l]
    
    z = 0
    errors = np.zeros((N_e*N_s/10,))   # 50(100/10)=500 total

    for e in range(N_e):   # for each epoch (50 total)        
        n = steplength(e)   # compute steplength        
        np.random.shuffle(training_set)  # choose subset of the training set of size 50
        subset = training_set[:50]

        for s in range(N_s):  # for each step (100 total)
            # select a random data item
            i = random.randint(0,49)
            data_item = subset[i]
            
            x = data_item[:9]
            y = data_item[9]
            
            # update a and b
            a,b = new_ab(x,y,a,b,lmbda,n)
            
            # plot the accuracy every 10 steps:
            if (s+1)%10 == 0:
                errors[z] = determine_accuracy(a,b,validation_set)
                z += 1  
    errors_list.append(errors) 
    #print(errors)

pt.figure()
pt.title('Cancer Classification Errors Using SGD Method')  
for i in range(4):
    pt.plot(errors_list[i], label=lmbdas[i])
    pylab.ylim([0,2])   
     
pt.xlabel('Steps')
pt.ylabel('Error')
pt.legend()
pt.show()

print("Average errors:")
for i in range(4):
    print(lmbdas[i], np.mean(errors_list[i]))