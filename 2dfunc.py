import math
import numpy as np
from numba import jit

@jit(nopython=True)
def forward_fast(param,z):
    """
    Forward (fast version)
    """
    A,B,b1,b2,t = param[0],param[1],param[2],param[3],param[4]
    z = np.array(z)
    z = A.dot(z) + b1
    z = z * (z > t)
    z = B.dot(z) + b2
    return z

class Model_2D_1hidden:
    """
    A ReLu activated Neural Network from \R^2 \to \R
    with a hidden layer of height self.height.
    - self.parametersA = first weight matrix
    - self.parametersB = second weight matrix
    - self.parametersb1 = first biases
    - self.parametersb2 = second bias
    - self.parameterst = ReLu threshold
    """
    def __init__(self,height=4):
        self.height = height
        self.parametersA = np.random.rand(height,2)
        self.parametersB = np.random.rand(height)
        self.parametersb1 = np.random.rand(height)
        self.parametersb2 = np.random.uniform()
        #self.parameterst = np.random.rand(height)
        self.parameterst = np.zeros(height)
        self.lr = 0.000005

    def sigma(self,x,t):
        """
        Activation function (ReLu)
        """
        return x * (x > t)

    def sigmap(self,x,t):
        """
        Derivative of activation function
        """
        return (x > t)

    def forward(self,z,fast = False):
        """
        Evaluate the model at z. Accelerated by jit.
        """
        if fast:
            params = (self.parametersA,self.parametersB,self.parametersb1,self.parametersb2,self.parameterst)
            return forward_fast(params,z)
        else:
            z = np.array(z)
            z = self.parametersA.dot(z) + self.parametersb1
            z = self.sigma(z,self.parameterst)
            z = self.parametersB.dot(z) + self.parametersb2
            return z

    def grad(self,z):
        """
        Evaluate the gradient of the model at the current paramater
        values at z. Parameters are considered stacked into one
        long array.
        """
        x,y = z[0],z[1]
        gradient = np.zeros(5*self.height+1,np.float64)
        for k in range(self.height):
            sum_ = self.parametersA[k][0]*x + self.parametersA[k][1]*y + self.parametersb1[k]
            gradient[2*k] = self.parametersB[k]*x*self.sigmap(sum_,self.parameterst[k])
            gradient[2*k+1] = self.parametersB[k]*y*self.sigmap(sum_,self.parameterst[k])
            gradient[k + 2*self.height] = self.sigma(sum_,self.parameterst[k])
            gradient[k + 3*self.height] = self.parametersB[k]*self.sigmap(sum_,self.parameterst[k])
            gradient[k + 4*self.height] = self.parametersB[k]*self.sigmap(sum_,self.parameterst[k])
        gradient[-1] = 1
        return (gradient,)
        
    def step(self,dtheta):
        """
        Step the parameters by dtheta (a stacked, flattened array)
        """
        dtheta = dtheta[0]
        s = "Incorrect increment length ({}).".format(len(dtheta))
        assert len(dtheta) == 5*self.height+1, s
        dtheta *= self.lr
        for k in range(self.height):
            self.parametersA[k][0] += dtheta[2*k]
            self.parametersA[k][1] += dtheta[2*k + 1]
            self.parametersB[k] += dtheta[k + 2*self.height]
            self.parametersb1[k] += dtheta[k + 3*self.height]
            #self.parameterst[k] += dtheta[k + 4*self.height]
        self.parametersb2 += dtheta[-1]
   
    def mathematica_print(self,array):
        deep = 0
        try:
            if len(array.shape) > 1:
                deep = 1    
        except:
            return str(array)
        s = "{"
        for item in array:
            if deep:
                s+=self.mathematica_print(item)+","
            else:
                s+=str(item)+","
        s = s[:-1]
        s += "}"
        return s

    def export_params(self):
        print("A = {};".format(self.mathematica_print(self.parametersA)))
        print("B = {};".format(self.mathematica_print(self.parametersB)))
        print("b1 = {};".format(self.mathematica_print(self.parametersb1)))
        print("b2 = {};".format(self.parametersb2))
        print("t = {};".format(self.mathematica_print(self.parameterst)))

    def export_zeros(self):
        for k in range(self.height):
            xcoef = self.parametersA[k][0]*self.parametersB[k]
            ycoef = self.parametersA[k][1]*self.parametersB[k]
            const = self.parametersb1[k]*self.parametersB[k]
            s = "{:.4f}*x + {:.4f}*y + {:.4f} == 0,".format(xcoef,ycoef,const)
            print(s)



scale = 2*math.pi
eps = 0.05
#test_func = lambda x,y: math.sin(scale*(x+y))
test_func = lambda x,y: math.sin(scale*x)+math.cos(scale*y)
model = Model_2D_1hidden(height=70)
model.lr = 0.005
n_points = 40
box_size = 1

#train
def train(n_epochs):
    for epoch in range(n_epochs):
        losses = []
        for x0 in np.linspace(-box_size,box_size,n_points):
            for y0 in np.linspace(-box_size,box_size,n_points):
                diff = model.forward((x0,y0)) - test_func(x0,y0)
                loss = (diff)**2 
                grad = model.grad((x0,y0))
                dtheta = ()
                for g in grad:
                    dtheta += (-1*g*diff,)
                model.step(dtheta)
                losses.append(loss)
        print("Epoch {} average loss: {}".format(epoch,sum(losses)/len(losses)))


#strain(300, 2500)
train(300)
