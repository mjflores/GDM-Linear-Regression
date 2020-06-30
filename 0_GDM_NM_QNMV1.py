
"""
Find the minimum of the following function:
    
        g(x,y)= (x-2)^2 + (y-3)^2
        
by using the gradient descent methods.
The exact solution is X* = (2,3)
@author: MFlores
Date:    Wed Apr 22 10:21:02 2020

"""


import matplotlib.pyplot as plt
import numpy as np

def contour_plot_gxy(npts = 1000):
    
    xlist = np.linspace(-5.0, 10.0, npts)
    ylist = np.linspace(-5.0, 10.0, npts)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros((npts,npts))
    for i in range(npts):
        for j in range(npts):
            Z[i][j] = gxy(np.array([X[i][j],Y[i][j]]))
            
    
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.scatter([0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0],c='y')
    ax.set_title('Filled Contours Plot of g(x,y)')    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
#    plt.show()


def surface_plot_gxy(npts = 1000):
    xlist = np.linspace(-5.0, 10.0, npts)
    ylist = np.linspace(-5.0, 10.0, npts)
    X, Y = np.meshgrid(xlist, ylist)
    Z = (X-2.)**2 + (Y-3)**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection= '3d')
    ax.plot_surface(X, Y, Z)
    #ax.set_title('Surface')    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('g(x,y)')
#    plt.show()
    
def gxy(x):
    return pow(x[0]-2.0,2.0) + pow(x[1]-3.0,2.0)

def grad_gxy(x):
    a1 = 2.0*(x[0]-2.0)
    a2 = 2.0*(x[1]-3.0)
    df= [a1,a2]
    return np.array(df)

def invHessian_gxy(x):
    # This matriz was obtained manually
    invH = [[0.5, 0.0],[0.0, 0.05]]
    return np.array(invH)
    
def gradientDescent_constant(eps = 0.001, gamma= 0.001):
   
    x0 = np.array([[1.0],[1.0]])    
    k = 0
    while True:
        gradfx = grad_gxy(x0)
        x1 = x0 - gamma*gradfx
        dif = np.linalg.norm(gradfx)
        #print("Iter [%d] = " % k,x1)
        if dif < eps:
            break
        x0 = x1
        k = k+1
    itg=[]
    return  x1, k

    
def stepBackTracking_v2(y0,gradfy):
    t = 1.0
    beta = 0.999999
    fxy = gxy(y0)
    grad2 = 0.5*np.linalg.norm(gradfy)**2.0
    k = 1
    while gxy(y0-t*gradfy)[0] > (fxy - t*grad2)[0]:
        t *= beta
        k+=1
         
    return t
    
def gradientDescent_BackTracking(eps = 0.01):

    x0 = np.array([[1.0],[1.0]])    
    k = 0
    while True:
        gradfx = grad_gxy(x0)              
        gamma  = stepBackTracking_v2(x0,gradfx)
        #print("gamma_k=",gamma)
        x1 = x0 - gamma*gradfx
        dif = np.linalg.norm(gradfx)
        if dif < eps:
            break
        x0 = x1
        k = k+1
    return  x1, k
    
    
def newtonMethod(eps = 0.001):
    x0 = np.array([[1.0],[1.0]])    
    k = 0
    while True:
        grad_fx = grad_gxy(x0)
        x1 = x0 - invHessian_gxy(x0).dot(grad_fx)
        dif = np.linalg.norm(grad_fx)
        #print("Iter: ",k,x1)
        if dif < eps:
            break
        x0 = x1
        k = k+1
    return  x1, k

def Bk1_Hk(Bk,sk,qk):
    dem1 = qk.T.dot(sk)[0][0]
    dem2 = sk.T.dot(Bk.dot(sk))[0][0]
    aux1 = qk*qk.T/dem1
    Bs = Bk.dot(sk)
    sB = sk.T.dot(Bk.T)    
    aux2 = Bs.dot(sB)/dem2    
    B1 = Bk + aux1-aux2
    return np.linalg.inv(B1)
    #return B1

def quasiNewtonMethod_BFGS(eps = 0.001):
    #Ref: https://www.mathworks.com/help/optim/ug/unconstrained-nonlinear-optimization-algorithms.html
    x0 = np.array([[-10.0],[1000.0]])        
    gradfx0 = grad_gxy(x0)
    H0 = np.eye(2)
    k = 0
    while True:            
        x1 = x0 - H0.dot(gradfx0)
        #===========================        
        sk = x1-x0
        gradfx1 = grad_gxy(x1)                
        qk = gradfx1 - gradfx0
        H1 = Bk1_Hk(H0,sk,qk)   
        
        dif = np.linalg.norm(gradfx1)
        #print("Iter: ",k,x1)
        if dif < eps:
            break
        x0 = x1
        H0 = H1
        gradfx0 = gradfx1
        k = k+1
        
    return  x1, k

#================================================
x_gd, k_gd = gradientDescent_constant()
print("1.1 GDM (step=cte) en %d iteraciones x=[%f,%f]" % (k_gd,x_gd[0],x_gd[1]))

x_gd_bt, k_gd_bt = gradientDescent_BackTracking()
print("1.2 GDM Backtracking (step=BT) en %d iteraciones x=[%f,%f]" % (k_gd_bt,x_gd_bt[0],x_gd_bt[1]))


x_nm, k_nm = newtonMethod()
print("2. NM en %d iteraciones x=[%f,%f]" % (k_nm,x_nm[0],x_nm[1]))

x_qnm, k_qnm = quasiNewtonMethod_BFGS()
print("3. QNM en %d iteraciones x=[%f,%f]" % (k_qnm,x_qnm[0],x_qnm[1]))

#================================================
#================================================
    
contour_plot_gxy(10)
surface_plot_gxy(10)

