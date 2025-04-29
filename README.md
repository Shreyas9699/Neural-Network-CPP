# Neural-Network-CPP
In short Neural Network is set of small number of nodes, working together to calculate a desired ouput.<br/>
The ***Perceptron*** is a fundamental building block of artificial neural networks. It's a simple linear classifier that can learn to separate linearly separable data. <br/>
<br/>

### Neural Network Types:
Neural networks come in a wide variety, each suited for different types of tasks and data. The one that is implete in this project is: <br>
1. ***Feedforward Neural Networks***: 
    The most basic type, with information flowing from input to output layers in a forward direction.<br />
    Examples:<br />
        - **Perceptron**: Single-layer network for simple linear classification.<br />
        - **Multi-Layer Perceptron (MLP)**: Multiple hidden layers for learning complex relationships.<br />
        - **Convolutional Neural Networks (CNNs)**: Specialized for image recognition, with layers that extract features like edges and textures. <br />

2. ***Recurrent Neural Networks (RNNs)***:
    Designed for sequential data, like text or time series, where information needs to flow both forward and backward.<br />



### Architecture:
The basic design of a *Perceptron*: <br/>
- A single neuron with one or more input connections and a single output.
- Each input connection has a weight associated with it.
- The output is calculated by a weighted sum of the inputs and a bias term, passed through an activation function (typically, Heaviside step function). 
<br/>

### Simple Implementations:
1. Input Layer: Takes in input vector
2. Hidden Layes: Interconnected Neurons
3. Output Layes: Output Values

`header/MLPerceptrons.cp` is a header file where the Perceptron and MultiLayer Perceptron is implemented. <br/>
The ***Perceptron*** can be best explained by implementing Logic Gates. Refer to code `logicGateTester.cpp` for its implmentations.

### Applications:
<li> Simple pattern recognition tasks, like classifying images of handwritten digits.
<li> Preprocessing for more complex neural networks.
<li> Educational tool for understanding the basics of neural network learning.


### Limitations:
* Can only handle linearly separable data (data that can be divided by a straight line).
* Cannot represent complex relationships between features.
* Prone to local minima when learning, where it gets stuck in a state where it cannot further improve its accuracy.


In our case, XOR can not be implemented by a single Perceptron, hence we use ***Multi-Layer Perceptron (MLP)***
Logic behind the XOR:
`logicGateTester.cpp` -> the weights are hardcoded, this is to check if the NN is working and gives desired output. <br/>
*Cmd*: 
> `g++ logicGateTester.cpp header/MLPerceptrons.cpp -o logicGateTester -I./header` <br/>
> `./logicGateTester`

![Alt text](Images/image-1.png)

### Activation Function:
A non linear function which makes neuron to learn the dramatic distinct between the categories near the boundary. This function provides non-linearity to the neuron.<br/>
Types of activation functions:
- *Binary Step Function*: Limits output values to exactly 0 or 1.
- *Sigmoid/Logistic Function*: Limits output values between 0 and 1, i.e. all real number between 0 and 1.
- *Hyperbolic Tangent Function*: Limits output values between -1 and 1, i.e. all real number between -1 and 1.
- *Rectified Linear Unit Function (ReLU)*: Limits output to be positive values, unbounded for +ve values only.

For this project the **Sigmoid Function** is implemented.

## How to make Neural Network learn?
Now that we have Multi-Layer Perceptron and Activation Function to implement a complex Logic gates such as XOR, which can not be implemented by a single Perceptron. But what makes NN stand out is its ability to learn. How can NN learn? <br/>
Let's see, for 1 single Training Sample: {X, Y} (Where X is the feature and Y is the expected/desired value)
* Feed an input sample X to the network
* Compare the output to the correct values Y
* Calculate Error
* Use the error to adjust the weights
* The Objective: to classify a litter better in the future run

### Error Function:
It is a measure on how bad a classifier is doing. It is essential for training process. Example: *Gradient Descent*. <br />
#### How Error Function works?
For Error of 1 Sample:
- Sample {X, Y}
- Assume we got Output value = 0.6
- Expected value is Y = 1
- So, the error is calculated as Error = y - output  
The goal is to make trining function to y, i.e output = y  


***Means Squared Error (MSE)*** is used to assess the performance of the NN

#### Gradient Descent:
It is a Method to minimize the error function. Consists of adjusting the weights to find the minimum error. Its is like "going downhill" on the error function to lowest valley

##### Possible Issue:
*Local Minima*: Stucking at a point which is assumed to be the lowest point, but is a lowest local point

![Alt text](Images/image-2.png)

#### The Delta Rule:
- Simple update formula used for adjusting the weights in a neuron
- Values considered to calculate the delats is:
	* Output error
	* One input
	* Learning rate

- The dealtaW will be -ve if the output is more than the desired value
- The dealtaW will be +ve if the output is less than the desired value

This dealtaW is then updated to Weights, which will intern affect the output to make the output closer to desired value.

### Learning Rate:
Learning Rate a uniqe constant, which is same for all neurons. It directly effects the learning rate as the name suggests. Higher the values means larger leaps towards the weights, whereas, lower values means smaller leaps towards the weights.


### Backpropagation:
A general form of the delta rule, which has several requirements on the neuron model. It Calculates all weight updates through the network, Which is done by propagating the error back through the layers in backward direction. Hence the name Backpropagation.

#### Algorithm:
1.  Feed sample to the network
2.  Calculate MSE
3.  Calculate the error term of each output neuron
4.  Iteratively calculate the error terms in the hidden layers
5.  Apply the delta rule
6.  Adjust the weights.


Using above algorithm,the XOR Gate is Implemented by training the Network.<br/>
*Cmd*: 
> `g++ NeuralNetwork.cpp header/MLPerceptrons.cpp -o NeuralNetwork -I./header` <br/>
> `./NeuralNetwork`


### Application:
Optical Character Recognition: <br/>
- Recognize characters in picture
- Digitizing books or documents 
- Taking notes by hand
- Reading (Text from images to/ text string to Speech)


Finally, here is the implementation of recognize number from a seven segment display.
- Given a vector of double floats, each indicating the brightness of one of each segments.
- Based on which the Neural Network will give the output number.

Seven segments are arranged as follows:
```
 --a--
|     |
f     b
|     |
 --g--
|     |
e     c
|     |
 --d--
```
Where the input pattern follows the order: `a b c d e f g`

![alt text](Images/image.png)

*Cmd*:
> `g++ SegDisplayRecog.cpp header/MLPerceptrons.cpp -o SegDisplayRecog -I./header` <br/>
> `./SegDisplayRecog`

Works best after training for 12000 epochs, and 7 to 10 NN is accurate most of the time.

### Sample inputs and expected outputs:
Input (a b c d e f g) | Output |
--- | --- |
1 1 1 1 1 1 0  | `0` |
0.55 0.55 0.55 0.55 0.55 0.55 0 | `0`|

### Noisy Inputs (with missing or extra segments):
Input (a b c d e f g) | Output | Expected (visually closest) |
--- | --- | --- |
1 0 1 1 1 1 0 | `0`/`6` | `0` (seg b missing) or `6` (seg g missing)|
0 1 1 0 0 1 0 | `6`/`1` | `4` (seg g missing) |
1 1 0 1 1 0 0 | `0` | `2` (seg g missing)
1 1 1 1 1 0 1 | `4`/`6` | `8` (seg f missing)

### Partial Inputs (with intensity variations)
Input (a b c d e f g) | Output |
--- | --- |
0.8 0.9 0.7 0.8 0.9 0.8 0.1 | `0` |
0.1 0.9 0.8 0.2 0.1 0.1 0.1 | `1` |
0.7 0.9 0.2 0.8 0.7 0.1 0.9 | `2` |

### Novel Inputs (patterns that don't match standard digits)
Input (a b c d e f g) | Output | Expected |
--- | --- | --- |
0 0 0 1 1 1 1 | `0`/`6` | `6` (c seg is missing) |
1 0 0 1 0 1 0 | `0`/`9` | `5` (seg c and f are missing) or `6` (seg c, e and g missing) or `9` (seg b, c and g missing) |
0 1 0 1 0 1 0 | `0`/`9` | `4` (seg g and c missing) / `9` (seg a, c and g missing) |


#### Note: Outputs are not precise since the training is done on very small NN, and it works well for Partial Inputs and regurla inputs and have varying outputs for other kind of inputs.