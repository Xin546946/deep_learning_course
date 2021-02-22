# This file is referenced by Peter Chen 
import numpy as np
from matplotlib import pyplot as plt

class Perceptron:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        self.w = np.array([0, 0]).reshape(2, 1)
        self.b = 0
        self.lr = 1               # learning rate
        self.max_iter = 100       # maximum iteration
        
        print('迭代次数 \t 误分类点 \t w \t b \n')
        print('0 \t N/A \t (%.4f, %4f) \t %.4f \n' % (self.w[0], self.w[1], self.b))
        
    def hypothesis(self, X):
        return (X @ self.w + self.b).ravel()
        
        
    def train(self):
        for iter in range(self.max_iter):
            # search misclassified point
            H = self.hypothesis(self.X)
            misclassified_indices = np.where(self.y * H <= 0)
            if(misclassified_indices[0].size == 0):    # no misclassification
                print('没有误分类点，感知结束\n')
                break
            
            # stochastic gradient descent
            i = np.random.choice(misclassified_indices[0], 1)
            xi = self.X[i, :].reshape(2, 1) 

            # update
            self.w = self.w + self.lr * self.y[i] * xi
            self.b = self.b + self.lr * self.y[i]
            
            print('%d \t (%.4f, %.4f) \t (%.4f, %.4f) \t %.4f \n' % (iter, xi[0], xi[1], self.w[0], self.w[1], self.b))

def main():           
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    percep = Perceptron(X, y)
    percep.train()



    positive_samples = X[np.where(y>0)[0], :]
    negative_samples = X[np.where(y<0)[0], :]

    fig, ax = plt.subplots()
    ax.scatter(positive_samples[:, 0], positive_samples[:, 1], color='r')
    ax.scatter(negative_samples[:, 0], negative_samples[:, 1], color='k')

    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])
    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])
    if percep.w[1] < 1e-6:
        pt1 = [(-percep.w[1]*y_min-percep.b)/percep.w[0], y_min]
        pt2 = [(-percep.w[1]*y_max-percep.b)/percep.w[0], y_max]
    else:
        pt1 = [x_min, (-percep.w[0]*x_min-percep.b)/percep.w[1]]
        pt2 = [x_max, (-percep.w[0]*x_max-percep.b)/percep.w[1]]
        
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])

    plt.show()

if __name__ == '__main__':
    main()