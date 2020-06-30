
"""
    In this example, a linear model Y = Xw is adjusted by using 
    Gradient Descent Method (GDM)  
    
    The lost funcion is quadratic error function
    The result is compared with OLS method
Input: 
     X a matrix of the form [1,X1] con X_1 \in R^{m},  cada elemento es X_1i con i = 1,2,...m,
     Y a vector of size m
Output:
     W a vector of size n+1
       
Note:

     h(x,w0,w1) = f_xw= y_hat = w0 + w1 x 
     
Error:
     J(w0,w1) = (1/2m) \sum(h(x_i,w0,w1)-y_i)^{2}
     
Objetive:

      Min J(w0,w1)
      
@author: MFlores
    Created on Fri Mar  6 17:11:37 2020
    
"""

import numpy as np
import random as rn
import matplotlib.pyplot as plt
import math

#====================================================


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def datosRegresion():
    X1=np.array([[1.0,0.0],
            [1.0,1.0],
            [1.0,2.0],
            [1.0,3.0],
            [1.0,4.0],
            [1.0,5.0],
            [1.0,6.0],
            [1.0,7.0],
            [1.0,8.0],
            [1.0,9.0],
            [1.0,10.0],
            [1.0,11.0]])
    
    Y1=[0.25,1.15,2.1,2.8,4.25,5.2,6.1,6.9,8.13,9.1,10.01,11.001]
    Y1 = np.array(Y1).reshape(1,len(X1))

    Y1 = Y1.T
    Y1 = Y1-130.
    return X1,Y1

def testRegresion(W):
    # Nuevos datos de para test
    Xnew = np.array([[1,3.],[1,4.],[1,0.],[1,3.5],[1,12.],[1,11.],[1,10.3]])
    ynew = np.zeros(len(Xnew))
    
    for i in range(len(Xnew)):
        ynew[i] = np.dot(Xnew[i],W)
        
    plt.plot(Xnew[:,1], ynew,c='blue')
    plt.show()
    
def plotJtheta(Jtheta,cl,ver):
    plt.plot(Jtheta,c=cl)
    plt.xlabel("Iteraciones")
    plt.ylabel("J(Theta)")
    plt.title("J(Theta) vs Iteraciones")
    if ver == True:
        plt.show()
    
#====================================================
def Jtheta(w,x,y):
      y_hat = f_xw(w,x)
      d = float(0.0)
      for i in range(len(Y)):
          d += (y[i] - y_hat[i])**2.
      err = d/10.
      return err[0]    
      
def f_xw(w,x):
    r,c = x.shape
    y_x_w = np.zeros(r)
    for i in range(r):
        y_x_w[i] = np.dot(w,x[i,:])
#        print(y_x_w[i])
    return y_x_w
#----------------------------------------------------
    
def gradientError(w,x,y):

    m = float(len(y))
    gr = np.zeros(len(w))
    
    gr[0] = -np.mean(y) + w[0] + w[1]*np.mean(x)
    gr[1] = -np.dot(x,y)/m + 1.0*w[0]*np.mean(x) + w[1]*np.dot(x,x)/m
    
    return gr

def gradientError_i(n,w,x,y):
    gr = np.zeros(n)
    i = rn.randint(0,3)
    print("Valor i: {}".format(i))
    gr[0] = 2.0*(-y[i] + w[0] + w[1]*x[i])
    gr[1] = gr[0]*x[i]
    return gr
#----------------------------------------------------

def gradientDescent_withConst(X,Y, eta = 0.01):
    ''' 
    https://en.wikipedia.org/wiki/Gradient_descent
    https://machinelearningmastery.com/gradient-descent-for-machine-learning/
    Learning Rate: The learning rate value is a small real 
    value such as 0.1, 0.001 or 0.0001. Try different values 
    for your problem and see which works best.
    '''
    n = X.shape[1] 
    
    w0 = [rn.uniform(-15.0,15.0) for _ in range(n)] 


    iter_k =[]
    iter_J = []
    k = 0

    eps = 1.0e-6
    while True:
      g = gradientError(w0,X[:,1],Y)
      w = w0 - eta*g
      #dif = np.linalg.norm(w-w0)
      dif = np.linalg.norm(g)

      if dif < eps or k>40000:
          break
      w0 = w          
      err = Jtheta(w0,X,Y)
      if math.isnan(err) or math.isinf(err):
          break
      iter_J.append(err)

      k=k+1      
    return w0, k, iter_J
     
#----------------------------------------------------
    
X,Y = datosRegresion()

plt.plot(X[:,1], Y,c='red')
plt.title('Y = wo + w1*X')
plt.xlabel('X')
plt.ylabel('Y')
#plt.show()

#====================================================
# Gradient Descent with \eta = const

W10,k0,J_theta0 = gradientDescent_withConst(X,Y,0.04)
W11,k1,J_theta1 = gradientDescent_withConst(X,Y,0.01)
W12,k2,J_theta2 = gradientDescent_withConst(X,Y,0.001)
W13,k3,J_theta3 = gradientDescent_withConst(X,Y,0.0001)

##print("Solucion por GD con eta = cte: %s" % W1)
print("\nSolucion por GD con eta = 0.1: {}".format(W10))
print("\nSolucion por GD con eta = 0.01: {}".format(W11))
print("\nSolucion por GD con eta = 0.001: {}".format(W12))
print("\nSolucion por GD con eta = 0.0001: {}".format(W13))

testRegresion(W11)
#testRegresion(W12)

plotJtheta(J_theta0,"red",False)
plotJtheta(J_theta1,"yellow",False)
plotJtheta(J_theta2,"blue",False)
plotJtheta(J_theta3,"green",True)


