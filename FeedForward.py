# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:33:31 2021

@author: Setayesh
"""

import numpy as np
from dataSets import Loading_Datasets
dataset = Loading_Datasets.Loadings()[0]

class FeedForward():
    
    def act_fun(x):
        
    	#bias = -.5 
    	
        #return 1/(1 + np.exp(-x + bias))
        
        return 1/(1 + np.exp(-x))
    
    
    
    
    def layers(self, input_, output_, hidden1_, hidden2_, input_data):
        
        W1 = np.random.randn(hidden1_, input_)
        
        b1 = np.zeros((hidden1_,1))
        
        mul1 = W1.dot(input_data) + b1
        
        out1 = self.act_fun(mul1)
        
        
        

        W2 = np.random.randn(hidden2_,hidden1_)
        
        b2 = np.zeros((hidden2_,1))
        
        mul2 = W2.dot(out1) + b2
        
        out2 = self.act_fun(mul2)
        
        
        

        W3 = np.random.randn(output_, hidden2_)
        
        b3 = np.zeros((output_,1))
        
        mul3 = W3.dot(out2) + b3
        
        out3 = self.act_fun(mul3)
        
        return out3
        
        
        

#3*3 matrix with numbers between 0 , 1
#A = np.random.normal(0, 1, (3, 3))




if __name__ == "__main__":  
    
    all_count = 0
    
    success_count = 0
    
    for i in range(0,199):
        
        all_count = all_count +1
        
        input_data = dataset[i][0] 
    
    
        temp = FeedForward.layers(FeedForward, 102, 4, 150, 60, input_data)

    
        if temp.argmax() == input_data.argmax():
            
            success_count = success_count + 1
            
    
    accuracy = success_count / all_count
    
    print(success_count)
    
    print(all_count)
    
    print(accuracy)        
        

    