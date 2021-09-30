# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 22:26:45 2021

@author: tajr
"""

import os
import random

# Target dirrectory to split
target_dir = 'ESL4'
print('looking in:')
print(target_dir + "\n")

# path on other machine
Path = 'data/ESL_DataV4/ESL4/'

class_num = 24

# Percentages for data split
train = 0.85
val = 0.1
test = 0.05
print('Splitting:')
print(str(train) + ',' + str(val) + ',' + str(test) + "\n")

# Find out how many image there are in dataset
files = [f for f in os.listdir(target_dir) if f.endswith('.jpg')]
print('number of images:')
print(str(len(files)) + "\n")

# lists collecting file names
noClassList = []
A = []
B = []
C = []
D = []
E = []
F = []
G = []
I = []
K = []
L = []
M = []
N = []
O = []
P = []
Q = []
R = []
S = []
T = []
U = []
V = []
W = []
X = []
Y = []
Z = []

# Splitting into classes
for image in files:
    
    # Finding annotation for image
    fileName, ext = os.path.splitext(image)
    txtName = target_dir + '/' + fileName + '.txt'
    
    # Opening input .txt file to read old annotations
    txt = open(txtName) 
    
    # If txt is empty skip (no class)
    if os.path.getsize(txtName) > 0: 
        
        # clear values
        params = []
        
        # Reading annotation content
        content = txt.read()
                          
        # split out params
        params = content.split(" ")
        
        # Close file
        txt.close
        
        # Annotations
        group = params[0]
        
        if group == '0':
            A.append(image)
        
        elif group == '1':
            B.append(image)
            
        elif group == '2':
            C.append(image)
        
        elif group == '3':
            D.append(image)

        elif group == '4':
            E.append(image)
        
        elif group == '5':
            F.append(image)

        elif group == '6':
            G.append(image)
        
        elif group == '7':
            I.append(image)

        elif group == '8':
            K.append(image)
        
        elif group == '9':
            L.append(image)

        elif group == '10':
            M.append(image)
        
        elif group == '11':
            N.append(image)

        elif group == '12':
            O.append(image)
        
        elif group == '13':
            P.append(image)            
        
        elif group == '14':
            Q.append(image)

        elif group == '15':
            R.append(image)
        
        elif group == '16':
            S.append(image)

        elif group == '17':
            T.append(image)
        
        elif group == '18':
            U.append(image)

        elif group == '19':
            V.append(image)
        
        elif group == '20':
            W.append(image)

        elif group == '21':
            X.append(image)
        
        elif group == '22':
            Y.append(image)        
        
        elif group == '23':
            Z.append(image)

        else:
            print('error')
            
    # No class in txt
    else:
        noClassList.append(image)
        
        # Close file
        txt.close

print(str(len(F)))

train_split = int(round(len(files)*train))
val_split = int(round(len(files)*val))
test_split = int(round(len(files)*test))

print('Data split:')
print(str(train_split) + ',' + str(val_split) + ',' + str(test_split) + "\n")

######################################################################
# Test
######################################################################

test_list = []

print('test')

# number of iamges to get
rand = 100

# get rand from lists
testA = random.sample(A, rand)
testB = random.sample(B, rand)
testC = random.sample(C, rand)
testD = random.sample(D, rand)
testE = random.sample(E, rand)
testF = random.sample(F, rand)
testG = random.sample(G, rand)
testI = random.sample(I, rand)
testK = random.sample(K, rand)
testL = random.sample(L, rand)
testM = random.sample(M, rand)
testN = random.sample(N, rand)
testO = random.sample(O, rand)
testP = random.sample(P, rand)
testQ = random.sample(Q, rand)
testR = random.sample(R, rand)
testS = random.sample(S, rand)
testT = random.sample(T, rand)
testU = random.sample(U, rand)
testV = random.sample(V, rand)
testW = random.sample(W, rand)
testX = random.sample(X, rand)
testY = random.sample(Y, rand)
testZ = random.sample(Z, rand)
test_noClass = random.sample(noClassList, rand)

# Add to list
test_list.extend(testA + testB + testC + testD + testE + testF + testG + testI + testK + testL + testM +  testN + testO + testP + testQ+ testR+ testS+ testT+ testU+ testV+ testW+ testX+ testY+ testZ+ test_noClass)

# Remove all the files randomly selected
files2 = [x for x in files if x not in test_list]

file_test = open('test.txt', 'w')
        
for image_test in test_list:
    file_test.write(Path + image_test + "\n")
    
file_test.close()

######################################################################
# Validation
######################################################################

val_list = []

print('val')

# number of iamges to get
rand2 = round((len(files2)*val) / class_num + 1)

# get rand2 from lists
valA = random.sample(A, rand2)
valB = random.sample(B, rand2)
valC = random.sample(C, rand2)
valD = random.sample(D, rand2)
valE = random.sample(E, rand2)
valF = random.sample(F, rand2)
valG = random.sample(G, rand2)
valI = random.sample(I, rand2)
valK = random.sample(K, rand2)
valL = random.sample(L, rand2)
valM = random.sample(M, rand2)
valN = random.sample(N, rand2)
valO = random.sample(O, rand2)
valP = random.sample(P, rand2)
valQ = random.sample(Q, rand2)
valR = random.sample(R, rand2)
valS = random.sample(S, rand2)
valT = random.sample(T, rand2)
valU = random.sample(U, rand2)
valV = random.sample(V, rand2)
valW = random.sample(W, rand2)
valX = random.sample(X, rand2)
valY = random.sample(Y, rand2)
valZ = random.sample(Z, rand2)
val_noClass = random.sample(noClassList, rand2)

# Add to list
val_list.extend(valA + valB + valC + valD + valE + valF + valG + valI + valK + valL + valM +  valN + valO + valP + valQ + valR + valS + valT + valU + valV + valW + valX + valY + valZ + val_noClass)

# Remove all the files randomly selected
files3 = [x for x in files if x not in val_list]

file_val = open('val.txt', 'w')
        
for image_val in val_list:
    file_val.write(Path + image_val + "\n")
    
file_val.close()

######################################################################
# Train
######################################################################

print('train')

file_train = open('train.txt', 'w')
        
for image_train in files3:
    file_train.write(Path + image_train + "\n")
    
file_train.close()

print('Done!')
