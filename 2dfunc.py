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


class Model_2D_2hidden:
    """
    A ReLu activated Neural Network from \R^2 to \R
    with two hidden layers of height self.height.
    
    FOR SOME REASON not working -- can't get it to 
    converge. Possible wrong gradient calculation.
    """
    def __init__(self,height=4):
        self.height = height
        self.parametersA = np.random.rand(height,2)
        self.parametersB = np.random.rand(height,height)
        self.parametersC = np.random.rand(height)
        self.parametersb1 = np.random.rand(height)
        self.parametersb2 = np.random.rand(height)
        self.parametersb3 = np.random.uniform()
        self.lr = 0.0005
        self.all_params = (self.parametersA,
                           self.parametersB,
                           self.parametersC,
                           self.parametersb1,
                           self.parametersb2,
                           self.parametersb3)

    def param_size(self):
        s = 0
        s+= np.sum(self.parametersA**2)
        s+= np.sum(self.parametersB**2)
        s+= np.sum(self.parametersC**2)
        s+= np.sum(self.parametersb1**2)
        s+= np.sum(self.parametersb2**2)
        s+= self.parametersb3**2
        return s

    def sigma(self,x):
        """
        Activation function (ReLu)
        """
        return x * (x > 0)

    def sigmap(self,x):
        """
        Derivative of activation function
        """
        return (x > 0)

    def forward(self,z):
        """
        Evaluate the model at z. 
        """
        z = np.array(z)
        z = self.parametersA.dot(z) + self.parametersb1
        z = self.sigma(z)
        z = self.parametersB.dot(z) + self.parametersb2
        z = self.sigma(z)
        z = self.parametersC.dot(z) + self.parametersb3
        return z

    def grad(self,z):
        """
        Evaluate the gradient of the model at the current paramater
        values at z. 
        """
        z = np.array(z)
        x,y = z[0],z[1]
        grad_A = np.zeros((self.height,2))
        grad_B = np.zeros((self.height,self.height))
        grad_C = np.zeros(self.height)
        grad_b1 = np.zeros(self.height)
        grad_b2 = np.zeros(self.height)
        grad_b3 = 0

        for k in range(self.height):
            sum1 = self.parametersA.dot(z) + self.parametersb1
            sum2 = self.parametersB.dot(self.sigma(sum1)) + self.parametersb2 
   
            # A, b1 gradient 
            ind_A_x = np.zeros_like(self.parametersA)
            ind_A_y = np.copy(ind_A_x)
            ind_b1 = np.zeros_like(self.parametersb1)
            ind_A_x[k][0] = 1
            ind_A_y[k][1] = 1
            ind_b1[k] = 1

            gA_vecx = self.sigmap(sum2)*(self.parametersB.dot(self.sigmap(sum1)*ind_A_x.dot(z)))
            gA_vecy = self.sigmap(sum2)*(self.parametersB.dot(self.sigmap(sum1)*ind_A_y.dot(z)))
            gb1_vec = self.sigmap(sum2)*(self.parametersB.dot(self.sigmap(sum1)*ind_b1))
            grad_A[k][0] = self.parametersC.dot(gA_vecx)
            grad_A[k][1] = self.parametersC.dot(gA_vecy)
            grad_b1[k] = self.parametersC.dot(gb1_vec)
        
            # B, b2 gradient
            for j in range(self.height):
                ind_B = np.zeros_like(self.parametersB)
                ind_B[k][j] = 1
                gB_vec = self.sigmap(sum2)*ind_B.dot(sum1)
                grad_B[k][j] = self.parametersC.dot(gB_vec)

            ind_b2 = np.zeros_like(self.parametersb2)
            ind_b2[k] = 1
            gb2_vec = self.sigmap(sum2)*ind_b2
            grad_b2[k] = self.parametersC.dot(gb2_vec)

            # C, b3 gradient
            grad_C[k] = self.sigma(sum2)[k]
            grad_b3 = 1
                

        gradient = (grad_A,grad_B,grad_C,grad_b1,grad_b2,grad_b3)
        return gradient
        
    def step(self,dtheta):
        """
        Step the parameters by dtheta. Assumes dtheta unpacks correctly.
        """
        for inc in dtheta:
            inc *= self.lr
        dA,dB,dC,db1,db2,db3 = dtheta[0],dtheta[1],dtheta[2],dtheta[3],dtheta[4],dtheta[5]

        self.parametersA += dA
        self.parametersB += dB
        self.parametersC += dC
        self.parametersb1 += db1
        self.parametersb2 += db2
        self.parametersb3 += db3
        
   
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
                s+="{}".format(item)+","
        s = s[:-1]
        s += "}"
        return s

    def export_params(self):
        print("A = {};".format(self.mathematica_print(self.parametersA)))
        print("B = {};".format(self.mathematica_print(self.parametersB)))
        print("CC = {};".format(self.mathematica_print(self.parametersC)))
        print("b1 = {};".format(self.mathematica_print(self.parametersb1)))
        print("b2 = {};".format(self.mathematica_print(self.parametersb2)))
        print("b3 = {};".format(self.parametersb3))



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
