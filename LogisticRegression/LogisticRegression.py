import numpy as np

class logisticRegression:
    def __init__(self,iterations = 100000, learning_rate = 0.0001) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def comp_h(self,w,x,b):
        return np.dot(w,x)+b
            
    def sigmoid(self,h):
        return (1/(1+np.exp(-1*h)))
    
    def comp_gradient(self,X,y,w,b):
        m,n = X.shape
        dj_dw = np.zeros(n)
        dj_db = 0
        for i in range(m):
            error = self.sigmoid(self.comp_h(w,X[i],b)) - y[i]
            dj_db += error
            for j in range(n):
                dj_dw[j]+=error*X[i,j]
        dj_dw = dj_dw/m
        dj_db = dj_db/m
        return dj_dw,dj_db

    def comp_gradient_desc(self,X,y):
        m,n = X.shape
        w = np.zeros(n)
        b = 0
        for i in range(self.iterations):
            dj_dw, dj_db = self.comp_gradient(X,y,w,b)
            w = w - (self.learning_rate * dj_dw)
            b = b - (self.learning_rate * dj_db)
        return w,b

    def fit(self,X,y):
        self.w,self.b = self.comp_gradient_desc(X,y)

    def predict(self,x):
        return  self.sigmoid(self.comp_h(self.w,x,self.b))


