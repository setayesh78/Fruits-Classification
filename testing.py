# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 09:31:53 2021

@author: Setayesh
"""
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
        
        learning_rate = 0.2
        number_of_epochs = 30
        batch_size = 15
        total_costs = []
        total_accuracy = []
        
        train_set = [dataset[x:x+batch_size] for x in range(0, len(dataset), batch_size)]
        
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
                    grad_W3 += 2 * (a3 - label) * self.der_act_fun(z3) @ np.transpose(a2)
                    
                    # bias for Last layer                
                    grad_b3 += 2 * (a3 - label) * self.der_act_fun(z3)
                    
                    # activation for 3rd layer
                    delta_3 = np.zeros((hidden2_, 1))
                    delta_3 += np.transpose(W3) @ (2 *(a3 - label) * self.der_act_fun(z3))                    
                    
                    # weight for 2rd layer
                    grad_W2 += delta_3 * self.der_act_fun(z2) @ np.transpose(a1)
                    
                    # bias for 2rd layer
                    grad_b2 += delta_3 * self.der_act_fun(z2)
                            
                    # activation for 2nd layer
                    delta_2 = np.zeros((hidden1_, 1))
                    delta_2 += np.transpose(W2) @ (delta_3 * self.der_act_fun(z2))
                    
                    # weight for first layer
                    grad_W1 += delta_2 * self.der_act_fun(z1) @ np.transpose(image)
                            
                    # bias for first layer
                    grad_b1 += delta_2 * self.der_act_fun(z1)
                            
                
                W3 = W3 - (learning_rate * (grad_W3 / batch_size))
                W2 = W2 - (learning_rate * (grad_W2 / batch_size))
                W1 = W1 - (learning_rate * (grad_W1 / batch_size))
                
                b3 = b3 - (learning_rate * (grad_b3 / batch_size))
                b2 = b2 - (learning_rate * (grad_b2 / batch_size))
                b1 = b1 - (learning_rate * (grad_b1 / batch_size)) 
                
            # calculate cost average per epoch
            cost = 0
            for train_data in dataset:
                a0 = train_data[0]
                a1 = self.act_fun(W1.dot(a0) + b1)
                a2 = self.act_fun(W2.dot(a1) + b2)
                a3 = self.act_fun(W3.dot(a2) + b3)
        
                for j in range(output_):
                    cost += np.power((a3[j, 0] - train_data[1][j,  0]), 2)
                    
            cost = cost/ len(dataset)
            total_costs.append(cost)  
        
        number_of_correct_estimations = 0
        
        for train_data in dataset:
            a0 = train_data[0]
            a1 = self.act_fun(W1.dot(a0) + b1)
            a2 = self.act_fun(W2.dot(a1) + b2)
            a3 = self.act_fun(W3.dot(a2) + b3) 
    
            if a3.argmax() == train_data[1].argmax():
                number_of_correct_estimations += 1 
                
        Accuracy = number_of_correct_estimations / len(dataset)
            
        total_accuracy.append(Accuracy)
        
        return total_costs,total_accuracy
                          
                
    
        
if __name__ == "__main__":  
    
    start_time = time.time()
    
    total_costs,Accuracy = Backpropagation.StochasticGradientDescent(Backpropagation, 102, 4, 150, 60)
    
    temp = []
    
    for i in range(0,8):

        temp.append(Backpropagation.StochasticGradientDescent(Backpropagation, 102, 4, 150, 60))
        
        total_costs =  np.array(total_costs) + np.array(temp[i][0])          
               
        Accuracy = np.array(Accuracy) + np.array(temp[i][1]) 
        
    
    end_time = time.time()
    
    print(end_time - start_time)        
    
    avg_total_cost = np.array(total_costs)/10
    
    avg_accuracy = np.array(Accuracy)/10
    
    print("avg_accuracy : ")
    
    print(avg_accuracy[0])
        
    
    epoch_size = [x for x in range(30)]
    plt.plot(epoch_size, avg_total_cost)

    

    
