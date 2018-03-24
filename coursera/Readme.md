# ML - Coursera - Andrew NG

Machine learning - science of getting computers to learn, without being explicitly programmed

## WEEK 1

#### Introduction
A computer program is said to learn from experience E, with respect to some task T, and some performance measure P, if its performance on T as measured by P improves with experience E

eg., playing chess.
E - experience of playing chess from the past
T - Ability to play chess
P - probability of winning the game.

##### Supervised learning
Housing price prediction - Regression model - predicting continuous valued output
Breast cancer - Malign/Benign - Classification model - predicting Discrete valued output

##### Classification
Email accounts of the company is hacked or not
House with a certain price sells less or more?

##### Unsupervised learning
Providing dataset without structure

####### Clustering model
Google news, Genes, Computer clustering

###### Non clustering model
Cocktail party problem
Two speakers with two microphones (with different relative distance from the speaker).


#### Model and cost function - Linear regression model
Linear regression with one variable - Univariate linear regression
Training Set ->  Learning algorithm ( h  maps x's to y's) outputs a function.

Modelling a real world problem.
m is the training dataset size. h(x) is the hypothesis derived from the training algorithm.

Cost Function - Measure the accuracy of our hypothesis function with Cost function.
To minimize the difference between the hypothesis and the cost function.
J(θ0,θ1) = 1/2m ∑i=1m(y^i−yi)2 = 1/2m∑i=1m(hθ(xi)−yi)2

Squared error function or Mean squared error.

Contour plot

##### Parameter learning
The gradient is a fancy word for derivative, or the rate of change of a function. It’s a vector (a direction to move) that
Points in the direction of greatest increase of a function (intuition on why)
Is zero at a local maximum or local minimum (because there is no single direction of increase)

Gradient Descent - Algorithm for minimizing the cost function or any function in general. Mostly preferred algorithm everywhere in ML to improve accuracy of the hypothesis

alpha - learning rate - Determines the step
partial derivative -  Determines the direction - positive slope or negative slope based.
To achieve the local optimum.

Gradient Descent with linear regression problem.
Convex function - Bowl

Batch gradient descent - Each step of gradient descent looks at all the training data set.

Linear Algebra Review -matrices and vectors.

## WEEK 2

#### Installing Matlab and Octave  - GUI

#### Multivariate linear regression
y Housing price ($) prediction based on
x1 size(feet²)
x2 number of bedrooms
x3 number of floors
x4 age of home (years)

n= no of features
xᶦ

Now the hypothesis changes
Hypothesis y = Theta (Transpose) X
Parameters
Cost Function J(θ) =  

Gradient descent for Linear regression with Multiple features
Gradient descent equation for the parameters

#### Practice I -- Feature scaling
- Making sure features are on the same scale, so that gradient descent can easily find the optimum x between -1 and 1

Mean normalization - Normalize the training data set values with respect to its mean
x between -0.5 to 0.5

#### Practice II -- Learning rate.
How to decide the optimal learning rate.
Learning rate should be small, but will take longer to converge

Polynomial regression

Normal equation method - Computing parameters analytically
θ=(XTX)−1XTy
There is no need to do feature scaling with the normal equation.

#### Normal Equation vs Gradient Descent
Gradient Descent	Normal Equation
Need to choose alpha	No need to choose alpha
Needs many iterations	No need to iterate
O (kn2)	O (n3), need to calculate inverse of XTX
Works well when n is large	Slow if n is very large

Normal Equation - Non invertible problem. Happens when features are redundant or there are too many features (reduce the features or use regularization)

Octave & MATLAB
Both are scientific programming languages
Octave - Open Source - Docs - https://www.gnu.org/software/octave/#install
Matlab - proprietary software - Free with mathworks

Octave - Basic operation
1 + 2; 2 + 3;
a or b
Matrix A = [ 1 2 3; 4 5 6; 7 8 9;]
Vector A = ones(1, 3) = [1 1 1]
A = 2*ones(1, 3) = [2 2 2]
zeros(1,4) = [0 0 0 0]
rand(1, 3) - Uniformly drawn between 0 and 1
randn(1, 3) - Gaussian variables with mean = 0 and std = 1
hist(-6 + sqrt(10) * randn(1, 10000) ) --- Mean - 6 & 10 is the variance
A histogram is an accurate graphical representation of the distribution of numerical data. It is an estimate of the probability distribution of a continuous variable
eye(4) - identity matrix
help <command>

Octave - Move data around
size(A) - 3 x 3 matrix
length(A)
who - variables in scope
whos
clear <variable-name>
clear
load(sample_data.dat)

Octave - Computing on data
SKIPPED

Octave - Plotting data
plot(t,y) -- t = [0 : 0.01 : 0.98]  y = sin(2*pi*4*t) y = cos(2*pi*4*t)
hold on - to hold on to the same plot
xlabel('')
ylabel('')
subplot(1,2,2)
SKIPPED

Control statements
SKIPPED

Vectorization

#### Assignment - Ex1
Ex1.1 WarmUpExercise()

Ex1.2 Linear regression with one variable
H = (theta'*X')';
S = sum((H - y) .^ 2);
J = S / (2*m);
To predict the profits for a food truck with various cities population & profits data.

Training dataset
It's a linear regression problem with one feature (population). With the training data set of population and profits, we need to create a hypothesis function with minimum cost function. Feature normalization technique?

Plotting the data.
What plot? Scatter plot?

Linear regression with multiple variables

Assignments in git

## WEEK 3 - Logistic regression
Classification problem
Binary classification problem - Negative class or Positive class
Multi class problem

Linear regression can't solve classification problem.
Hypothesis H theta(x)= g(thetaTx)
Logistic function or Sigmoid function - g

0 <= Hypothesis function <= 1
Hypothesis function = g(z)
z > 0.5
Decision Boundary - Acts as a classifier between the two classes.

Convex function  - Converges to a global optimum
Non convex function  - Doesn't converge to a global optimum
Since the logistic progression is not a convex function we can't use the same COST function of

Cost Function
Cost(H theta (x), y) = -log(H) if y=1
Cost(H theta (x), y) = -log(1-H) if y=0

Simplified Cost function and gradient descent

Advanced Optimization
SKIPPED

MultiClass Classification problem - One vs all problem
Email foldering/tagging - Work, Friends, Promotions

Solving the fitting problem & Regularization (in linear and logistic regression)
Underfitting  - High bias
Overfitting - High Variance  - Makes accurate predictions on the training set but doesn't generalize well on the new predictions

Addressing overfitting problem.
1. Plotting the hypothesis to understand it
2. Reduce the no. of features - Manually pick features or use model selection algorithm
3. Regularization - Keep all the features but reduce the magnitude of the feature

Regularization
Smaller values of theta - Simpler hypothesis, less prone to overfitting
Adding a regularization parameter (lambda) to reduce the value of theta - to balance between underfitting and overfitting

Regularization - Linear regression

Regularization - Logistics regression

Programming assigments
Logistics Regression to predict whether a student gets admitted to a university or not


## WEEK 4 - Neural networks - Non linear hypothesis

One learning algorithm for all - Similar to brain
Sensor representation in brain

Seeing with your tongue
Human echolocation - Sonar - Pattern of sound bouncing of the environment
Haptic belt - Direction sense

Neuron is the computational unit of brain. They contain input wires called dendrite and output wires called axons. neurons communicate with a pulse of electricity called spike

Model representation I
A Neuron unit  model - Logistics model
Bias Unit or Bias Neuron - X0
Weights or Parameters or Theta
X1, X2, X3 - Features that are inputs
Output H = ?

Neural network
Layer 1 (input layer - input features) -> Layer 2 (Hidden layer - activation units) -> Layer 3 (Output layer - neurons)
Each layer gets its own matrix of weights,
Theta is the function of j mapping jth layer to (j+1)th layer- Dimension of theta
a(j)i="activation" of unit i in layer
jΘ(j)=matrix of weights controlling function mapping from layer j to layer j+1?

Model representation II
Forward propagation - Vectorized implementation
Multiple layer - Input layer - 2 hidden layers - Output layer

Examples and Intuitions I
XOR & XNOR & logical  AND & logical OR - For boolean valued input features
Draw a truth table to figure out the logical function from the weights and input features.
So each neuron in a neural network by using different weights, can compute several logical functions. Neural networks with multiple layers can help us compute complex functions.

Examples and Intuitions II
Compute negation (not x1) - Theta1 = 10 Theta2 = -20
Compute (not x1) AND (not x2) - Can use two layers of computation
X1 AND X2 --> X1 = not x1, X2 = not x2
x1 x2 Req-H  Theta1 = 10 Theta2 = -20 Theta = -20
0   0    1
0   1    0  
1   0    0  
1    1    0   

Compute x1 XNOR x2 - Two layers of computation
Handwritten digit classification - Used by US Gov for reading pin codes for sending mails.

Multi Class classification
Using neural networks for multi class classification model. Extension of one vs all method

To classify car, pedestrian, motorcycle or truck - Outputs a vector with 4 elements

Assignment - Multi class classification and neural networks
Multi Class classification problem - 10 class (0 to 9)
Two techniques to solve this
1. Using One vs All logisitic regression implementation with regularization
2. Neural networks

Load the training data set into X and y
X is a 20*20 images to predict a number between 0 to 9 So output will be

Select a random theta value and improve the hypothesis


## WEEK 5 - Neural networks learning
https://cloud.google.com/blog/big-data/2017/10/my-summer-project-a-rock-paper-scissors-machine-built-on-tensorflow

Cost function and backward propagation
Neural networks - L layers and K classes, Sl no of units
Cost function - Generalization of  CostFunctionLRReg (use k classes)
Equation -
The double sum simply adds up the logistic regression costs calculated for each cell in the output layer
The triple sum simply adds up the squares of all the individual Θs in the entire network.

To minimize CostFunctionNN - BackPropagation algorithm
1. Use Forward propagation across layers to compute the hypothesis with any given weights.
2. Compute error function (delta) from the hypothesis across layers using Backpropagation
3. Compute triangle and the partial derivative terms

Backpropagation - Intuition
Backpropagation works similar to fwd propagation using the weights

Backpropagation - In practice
Unrolling parameters
Unroll all the theta (theta1, theta2, theta3) and D (D1, D2, D3) matrices to vectors to make them easier to input or output to the costFunction

Gradient checking - Numerical gradient computation
To verify the gradient obtained from the back propagation algorithm. Turn off gradient checking after verifying the back propagation algorithm, since they are computationally costlier and slow.

Random initialization
Initializing the initial value for theta. Setting the weights to zero? Won't work
Symmetry breaking - make sure the weights are not the same (zero or the same weight wont work since they produce the identical values in a layer)

Putting it together
Pick a network architecture
No of input units: dimensions of x(i)
No of output units: number of classes
No of hidden layer and units: default 1 hidden layer, have the same number of hidden units

Training a neural network
Randomly initialize the weights
Implement the forward propagation (Hypothesis)
Implement Cost function (J(theta))
Implement backpropagation  (partial derivative of J(theta))
Gradient checking  -
Advanced optimization - Gradient descent

Application of neural networks - Autonomous driving
Training data set - 32*32 image of the road and the steering direction (by an actual driver) (Almost 12 images per second rate)
Predicts whether to steer left or right based on the image
How to compact the speed? How to determine the opposite vehicle? so on?

Programming Assignment: Neural Network Learning
FeedFwd propagation
Back propagation to determine gradient


## WEEK 6 - Advice for applying machine learning
Best practices, debugging of learning algorithm

Deciding what to do next?
There is difference between the one who uses the algorithm efficiently and the one who knows the algorithm

Debugging a learning algorithm
When error rate increases in solving a regularized linear regression problem,
- Get more training examples (ex: when the training dataset is 100 vs 10000)
- Try smaller set of features
- Try getting additional features (ex: )
- Try adding polynomial features
- Increasing/Decreasing Lambda

Machine learning diagnostics - A test to figure out whether a learning algorithm is/isn't performing well.

Evaluating your hypothesis
Addressing the overfitting and underfitting problem
Split the overall dataset 70% into training (Training set) and 30% test dataset (Test set).

Training/Testing procedure for linear regression/ Logistic regression/ nueral network
Linear
Learn parameter Theta from training data.
Compute test set error

Logistic
Learn parameter Theta from training data.
Compute test set error (0/1 misclassification error)

Model selection and training/validation/test sets
Generalization error
Training set, Cross Validation set (CU) and Test set

Diagnosing Bias vs Variance
Diagnosing Underfitting vs Overfitting problem
Choosing the degree of polynomial -- d and Train error/CV error
Choosing the regularization parameter -- lambda and Train error/CV error
Learning curves - w.r.t dataset size (m)

Building a spam classifier
Prioritising what to work on -
Error Analysis
