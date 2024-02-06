# TracIn : Estimating Training Data Influence by Tracing Gradient Descent

- Syrine Enneifer 2049467
- Giordano Pagano 2077179

In this project is implemented the method introduced in the paper Estimating Training Data Influence by Tracing Gradient Descent (https://arxiv.org/pdf/2002.08484.pdf).

# Introduction:
We present a method called TracIn, designed to determine the influence of a training example on a prediction made by the model.
We aim to analyze the evolution of the loss of a test point over the training process whenever the training example of
interest was utilized.

# Idealized Notion of Influence
Given a set of n training points S = {z1, z2, . . . , zn ∈ Z}, we train the predictor by finding parameters w that minimize the training loss ΣL(w, zi), i=1,...,n via an iterative optimization procedure (such as Stochastic Gradient Descent) which utilizes one training example zt ∈ S in iteration t, updating the parameter vector from wt to wt+1. Then the idealized notion of influence of a particular training example z ∈ S on a given test example z0 ∈ Z is defined as the total reduction in loss on the test example z0 that is induced by the training process whenever the training example z is utilized,
i.e.TracInIdeal(z, z0) = Σ[t:zt=z]   L(wt, z0) − L(wt+1, z0)

## Example:
This method is exemplified in the context of an image classification task, where the model's objective is to predict the subject of the test image ("zucchini" here).
We would like to identify the influence of a training data point on F(data point at inference time).

<img src="figures/goal.png" width="400"/>

# Proponents and Opponents

As the model advances during training, it encounters different training examples, affecting the loss on the test image. Indeed, whenever a particular training example is visited, this leads to adjustments in the model's parameters, subsequently affecting the prediction and loss for the test example. 

By tracing the training example's impact throughout the training process, the change in loss for the test example can be attributed to that specific training example. Proponents, such as images of zucchinis, decrease loss, while opponents, like images of seatbelts, increase loss. For instance, an image labeled "sunglasses" serves as a proponent because it includes a seatbelt, pushing the model to better differentiate between zucchinis and seatbelts.

<img src="figures/idea.png" width="800"/>

TracIn is simple to implement; all it needs is the ability to work with gradients, checkpoints, and loss
functions.

# Implementation
First of all we don't know what are the parameters at timestep t+1 in timestep t, so we have to approximate the computation of the second term in the previous formula. Then we have to face up that at training time we can't access to testing samples, so the main idea is to store in memory the weights of our model at every timestep; but for very long training processes this would be infeasible; so the solution is to use CHECKPOINTS.
We will use only few parameter weights from all the parameters that we can store over all the training process; and using this checkpoints, finally, we can use this formula:
<img src="figures/tracincp.png" width="800"/>

# Evaluations
In the colab files of this repository we compare TracIn with Influence Functions and the Representer Point Selection method, highlighting the performance of our method compared to the others.
( In the colab file there are also implementations of Influence Functions and Representer Point Selection )

# Results
As shown in the graphs, the TracIn method is the most efficient of all, in particular the one with the same learning rate between checkpoints

<img src="figures/MislabbelledDataIdentification.PNG" width="820"/>

# How to run the code
The file extension is '.ipybn' so it can be uploaded in a colab session and the user can run each cell sequentially, in order to run the code in the right way.


