

```python
# 本代码通过构建三层神经网络结构，来学习mnist数据集
```


```python
import numpy
import scipy.special
import matplotlib.pyplot
%matplotlib inline
```


```python
class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes
        
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        
        self.lr = learningrate
        
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    
    def train(self,input_list,target_list):
        inputs = numpy.array(input_list,ndmin=2).T
        target = numpy.array(target_list,ndmin=2).T
        hidden_input = numpy.dot(self.wih,inputs)
        hidden_ouput = self.activation_function(hidden_input)
        
        final_input = numpy.dot(self.who,hidden_ouput)
        final_ouput = self.activation_function(final_input)
        
        output_errors = target - final_ouput
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr*numpy.dot((output_errors*final_ouput*(1.0-final_ouput)),numpy.transpose(hidden_ouput))
        
        self.wih += self.lr*numpy.dot((hidden_errors*hidden_ouput*(1.0-hidden_ouput)),numpy.transpose(inputs))
        pass
    
    def query(self,input_list):
        inputs = numpy.array(input_list,ndmin=2).T
        hidden_input = numpy.dot(self.wih,inputs)
        hidden_ouput = self.activation_function(hidden_input)
        final_input = numpy.dot(self.who,hidden_ouput)
        final_ouput = self.activation_function(final_input)
        return final_ouput
    
    
```


```python
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file = open("E:/work/mnist_dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_value = record.split(',')
        inputs = (numpy.asfarray(all_value[1:])/255.0*0.99)+0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_value[0])] = 0.99
        n.train(inputs,targets)
        
        pass
    pass

test_data_file = open("E:/work/mnist_dataset/mnist_test.csv",'r',)
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print(correct_label,"correct label")
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    outputs = n.query(inputs)

    label = numpy.argmax(outputs)
    print(label,"nework's answer")
    
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = numpy.asarray(scorecard)
print("performance = ",scorecard_array.sum()/scorecard_array.size)
```


```python
import cv2
import numpy as np
img = cv2.imread("E:/work/mnist_dataset/2.jpg",0)
print(img.shape)
tempimg = cv2.resize(img,(28,28),0)
cv2.imshow("test",tempimg)
cv2.waitKey()
print(type(tempimg))
tempimg = np.reshape(tempimg,(1,input_nodes),0)
tempimg = 255.0 - tempimg
input_data = tempimg /255.0*0.99 + 0.01
output_data = n.query(input_data)
print(output_data)

label = numpy.argmax(output_data)
print(label,"network's answer")
```
