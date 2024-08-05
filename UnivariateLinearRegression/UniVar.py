
class UniVariantLinearReggressionModel:
    def __init__(self):
        
        self.w = 0
        self.b = 0
    def Model(self,x_train,intercept,slope):
        return ((slope*x_train)+intercept)
    
    def cost_functionJ(self,b,w,x_train,y_train):
        model = self.Model(x_train,intercept=b,slope=w)
        cost_function = ((1/(2*model.size)) * np.sum((model-y_train)**2))
        return (model,cost_function)

    def GradientDesc(self,x_train,y_train,model):
        import numpy as np
        for i in range(1000):
            djdw = ((1/model.size)*np.sum((model-y_train)*x_train))
            djdb = ((1/model.size)*np.sum((model-y_train)))
            temp_w = self.w-(0.00001)*djdw
            temp_b = self.b - (0.00001)*djdb
            self.w = temp_w
            self.b = temp_b
            model = self.Model(x_train,intercept = self.b, slope = self.w)
            #print(w,b)

    def train_model(self,x_train,y_train):
        model = self.Model(x_train,self.b,self.w)
        self.GradientDesc(x_train,y_train,model)
        

    def predict(self,x_test):
        return self.Model(x_test,self.b,self.w)

    def VisualizeFit(self,x_train,y_train):
            plt.scatter(x=x_train,y=y_train)
            plt.xlabel("x data -->")
            plt.ylabel("y data -->")
            plt.axline((0,intercept),slope=slp,color='r')