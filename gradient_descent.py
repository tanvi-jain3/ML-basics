#A gradient descent is a fundamental concept in developing neural networks. 
#A gradient function minimises the value of the cost function in an n-dimensional space (for a neural network with n points as input)
#This following piece of code outlines how to represent a gradient function in python to find the best fit line for a given set of data.
#this is how the Linear Regression function in scikit-learn works. There is a separate code file explaining that as well.

import numpy as np

def gradient_descent(x,y) :
    m_curr=b_curr=0
    iterations=10000
    n=len(x)
    learning_rate=0.08 #optimal learning rate which gives us the closest expected result of m=2 and b=3
    
    for i in range(iterations):
        y_predicted=m_curr*x+b_curr
        cost=(1/n)*sum([val**2 for val in (y-y_predicted)]) #MSE between the predicted vaues and the actual output values y
        md = -(2/n)*sum(x*(y-y_predicted)) #derivative of slope
        bd = -(2/n)*sum(y-y_predicted) #derivative of y-intercept
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))


x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_descent(x,y)