import numpy as np

class MultipleLinearRegression:
    def __init__(self,iteration=10000,learning_rate=0.001):
        self.iteration = iteration
        self.learning_rate = learning_rate
    
    #main function
    def RegFunc(self,wi,xi,b):
        return np.dot(wi,xi)+b
    
    #cost function
    def costJwb(self,X,yi):
        m = X.shape[0]
        cost = 0
        for i in range(self.m):
            cost += ((self.RegFunc(self.w,X[i],self.b)-yi)**2)/(2*m)
        return cost
    
    #find gradients    
    def gradient(self,X,y,wi,b):
        m,n = X.shape

        dj_dw = np.zeros(n)
        dj_db = 0

        for i in range(m):
            error = self.RegFunc(wi,X[i],b)-y[i]
            for j in range(n):
                dj_dw[j] += error * X[i][j]
            dj_db +=error
        dj_dw = dj_dw/m
        dj_db = dj_db/m

        return dj_dw,dj_db

    #do gradient descent
    def Gradient_dec(self,X,y):
        n = X.shape[1]
        
        w = np.zeros(n)
        b = 0
        
        for i in range(self.iteration):
            dj_dw , dj_db = self.gradient(X,y,w,b)
            w = w - self.learning_rate * dj_dw
            b = b - self.learning_rate * dj_db
        
        return w,b
    

    #train model
    def Fit(self,X_train,y_train):
        self.w,self.b = self.Gradient_dec(X_train,y_train)


    #/predict model
    def Predict(self,x_test):
        return np.dot(self.w,x_test)+self.b
    


