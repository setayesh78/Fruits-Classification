# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:34:58 2021

@author: Setayesh
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from dataSets import Loading_Datasets
dataset = Loading_Datasets.Loadings()[0]

class Backpropagation():
    
    def act_fun(x):
        return 1/(1 + np.exp(-x))
    
    def der_act_fun(x):        
        return (np.exp(-x)/np.square(1 + np.exp(-x))) 
    
    
    
    
    def StochasticGradientDescent(self, input_, output_, hidden1_, hidden2_):
        
        W1 = np.random.randn(hidden1_, input_)
        b1 = np.zeros((hidden1_,1))
        
        W2 = np.random.randn(hidden2_,hidden1_)
        b2 = np.zeros((hidden2_,1))
        
        W3 = np.random.randn(output_, hidden2_)
        b3 = np.zeros((output_,1))
        
        learning_rate = 1
        number_of_epochs = 5
        batch_size = 10
        total_costs = []
        train_set = [dataset[x:x+batch_size] for x in range(0, 200, batch_size)]
        
        for i in range(0,number_of_epochs):
            random.shuffle(train_set)
            
            for batch in train_set:
                grad_W1 = np.zeros((hidden1_, input_))
                grad_b1 = np.zeros((hidden1_,1))
                grad_W2 = np.zeros((hidden2_,hidden1_))
                grad_b2 = np.zeros((hidden2_,1))
                grad_W3 = np.zeros((output_, hidden2_))
                grad_b3 = np.zeros((output_,1))
                
                for image, label in batch:
                    # compute the output (image is equal to a0)
                    a1 = self.act_fun(W1.dot(image) + b1)
                    z1 = W1.dot(image) + b1
                    a2 = self.act_fun(W2.dot(a1) + b2)
                    z2 = W2.dot(self.act_fun(a1)) + b2
                    a3 = self.act_fun(W3.dot(a2) + b3)
                    z3 = W3.dot(self.act_fun(a2)) + b3
                    
                    
                    #======= backpropagation ========
                    
                    #weight for Last layer
                    for j in range(output_):
                        for k in range(hidden2_):
                            grad_W3[j, k] += 2 * (a3[j, 0] - label[j, 0]) * self.der_act_fun(z3[j, 0]) * a2[k, 0]
                    
                    # bias for Last layer
                    for j in range(output_):
                            grad_b3[j, 0] += 2 * (a3[j, 0] - label[j, 0]) * self.der_act_fun(z3[j, 0])
                    
                    # activation for 3rd layer
                    delta_3 = np.zeros((hidden2_, 1))
                    for k in range(hidden2_):
                        for j in range(output_):
                            delta_3[k, 0] += 2 * (a3[j, 0] - label[j, 0]) * self.der_act_fun(z3[j, 0]) * W3[j, k]
                    
                    # weight for 2rd layer
                    for k in range(hidden2_):
                        for m in range(hidden1_):
                            grad_W2[k, m] += delta_3[k, 0] * self.der_act_fun(z2[k,0]) * a1[m, 0]
                    
                    # bias for 2rd layer
                    for k in range(hidden2_):
                            grad_b2[k, 0] += delta_3[k, 0] * self.der_act_fun(z2[k,0])
                            
                    # activation for 2nd layer
                    delta_2 = np.zeros((hidden1_, 1))
                    for m in range(hidden1_):
                        for k in range(hidden2_):
                            delta_2[m, 0] += delta_3[k, 0] * self.der_act_fun(z2[k,0]) * W2[k, m]
                    
                    # weight for first layer
                    for m in range(hidden1_):
                        for v in range(input_):
                            grad_W1[m, v] += delta_2[m, 0] * self.der_act_fun(z1[m,0]) * image[v, 0]
                            
                    # bias for first layer
                    for m in range(hidden1_):
                            grad_b1[m, 0] += delta_2[m, 0] * a1[m, 0] * (1 - z1[m, 0])
                
                W3 = W3 - (learning_rate * (grad_W3 / batch_size))
                W2 = W2 - (learning_rate * (grad_W2 / batch_size))
                W1 = W1 - (learning_rate * (grad_W1 / batch_size))
                
                b3 = b3 - (learning_rate * (grad_b3 / batch_size))
                b2 = b2 - (learning_rate * (grad_b2 / batch_size))
                b1 = b1 - (learning_rate * (grad_b1 / batch_size)) 
                
            # calculate cost average per epoch
            cost = 0
            for train_data in dataset[:200]:
                a0 = train_data[0]
                a1 = self.act_fun(W1.dot(a0) + b1)
                a2 = self.act_fun(W2.dot(a1) + b2)
                a3 = self.act_fun(W3.dot(a2) + b3)
        
                for j in range(output_):
                    cost += np.power((a3[j, 0] - train_data[1][j,  0]), 2)
                    
            cost = cost/ 200
            total_costs.append(cost)  
        
        number_of_correct_estimations = 0
        
        for train_data in dataset[:200]:
            a0 = train_data[0]
            a1 = self.act_fun(W1.dot(a0) + b1)
            a2 = self.act_fun(W2.dot(a1) + b2)
            a3 = self.act_fun(W3.dot(a2) + b3) 
    
            if a3.argmax() == train_data[1].argmax():
                number_of_correct_estimations += 1 
            
        print(f"Accuracy: {number_of_correct_estimations / 200}")
        
        return total_costs
                          
                
    
        
if __name__ == "__main__":  
    
    start_time = time.time()

    total_costs = Backpropagation.StochasticGradientDescent(Backpropagation, 102, 4, 150, 60)
    
    end_time = time.time()
    
    print(end_time - start_time)
    
    epoch_size = [x for x in range(5)]
    plt.plot(epoch_size, total_costs)

    

    