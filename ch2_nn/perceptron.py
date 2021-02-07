#   This code is for test perceptron in the lecture 
import numpy as np
import matplotlib.pyplot as plt
import time

# given input data M1 = [(3,3) ,+1], M2 = [(4,3), -1], M3 = [(1,1),-1]
points = np.array([[3,3],[4,3],[1,1]])
labels = np.array([[+1],[+1],[-1]])

# visualize data
plt.figure(figsize = (6,6), dpi = 80)
plt.xlim((-3,5))
plt.ylim((-3,5))
plt.xlabel('x')#设定x轴注释
plt.ylabel('y')#设定y轴注释
plt.title('perceptron')

for point,label in zip(points,labels):
    if(label == +1):
        plt.scatter(point[0], point[1], c = 'red', s = 30, marker = 'o', alpha = 0.7)
    else:
        plt.scatter(point[0], point[1], c = 'blue', s = 30, marker = 'x', alpha = 0.7)

# perceptron
# set initial paran 
w = np.array([0,0]) 
b = 0
learning_rate = 1.0
success = False
num_right_classified = 0

axes = plt.gca()
line_x = np.array(axes.get_xlim())
line = None 

print("@@Current weight: {}, bias: {}.".format(w,b)) 
while(num_right_classified != len(points)):
    num_right_classified = 0
    for point,label in zip(points,labels):
        if((np.dot(w,point) + b) * label <= 0): 
            if(line):
                line_for_remove = line.pop(0)
                line_for_remove.remove()   
            # print("point: ", point, " is misclassified.")
            w = w + learning_rate * label * point
            b = b + learning_rate * label  
            print("@@Current weight: {}, bias: {}.".format(w,b)) 
            line_y = - w[0] / (w[1] + 1e-6) * line_x - b / (w[1] +1e-6) 
            
            line = plt.plot(line_x, line_y, '-')
            plt.pause(1)
           
        else:
            # print("point; ", point, " is not misclassified.")
            num_right_classified = num_right_classified + 1

line_y = - w[0] / (w[1] + 1e-6) * line_x - b / (w[1] +1e-6) 
plt.plot(line_x, line_y, '-')
plt.draw()           
plt.show()