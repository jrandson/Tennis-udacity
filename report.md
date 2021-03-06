#### Overview
this project was solved using pytoch framework, which is a powerfull tool for deep learning development.Using the DDPG algorith, the environment was solved in  925 episodes with a score of +0.5.

`Episode 925	Average Score: 0.50	Score: 0.40	 Model reached the score goal in 925 episodes!`


#### Learing Algorithm
For the solution of this projects, it was applied the DDPG (Deep Deterministic Policy Gradient) algorithm, which is very suitable for continuous spaces. This algorithm uses two similar neural network, one for the Actor and another for the Critic model.

In the DDPG, the Actor model is trained to represent the polyce itself, in wich is responsable to map states to the best possible actions. It is used to appoximate the optimal policy deterministically. The Critic model, on the other hand, works to learn to evaluate the pair (state, action) predicted by the Actor. That's mean that the both input and output of the actor are used by the critc model, in which the action is the target value.

Since, neural networks as classified as supervised learning, we need the target data in order to train these models. So, the target Q values are computed using the bellman equation:

<img src="bellman_eq.png" width="400" height="45" />

There are two copies of each networks, one for the actor and another for the actor. These copies are the regular/online networks and the other is the target network. In each steps the target networks are slighted updated by the online network with a small percetange of their weights and this is called a soft update strategy.

These nework have three full connected layers with relu as activation function at the hidden layer and a tanh in the output layer.

In DDPG, it used a buffer with store a bunch of steps responses (state, action, reward, next_state, done), which are sampled randomly by the Actor model to avoid correlation between them during the model training. This buffer ensure the model are being trained using data that are independents of each other.

Inproviments in the exploration is done by adding some noise to the action. for this purpose it was used the Ornstein-Uhlenbeck Process, which has been proved to improve the action exploration according to this ddpg papper

This environment was solved using a very similar solution used to solve the [Reacher project](https://github.com/jrandson/Reacher-udacity). The difference, this time, is that now we have two agents competing one with the other.

The archtecture of the projec counts with a agent class that uses these to solve the environmet. The main parans used is described bellow:

* the max number of episodes was setted to 5000
* within each episode, it was allowed to try only 1000 steps at the most
* discount rate to the improviment of the polyce: 0.99
* Reply buffer for a random selection of episode with size 10^6. The seelction batch in this buffer was 128
* Both neuron nets for Actor and Critic models have 256 and 128 neurons in their two hiddne layers, respectivelly
* The MSE loss function was used to optimize the models.
* The result of training can be seen bellow

Some test were done in order to achieve some iprovements, but it was not successfully. These approaches include introducing droput layers and use regularizations in order to avoid metalearn. Also, some test were done initializing the weights of the layers with a normal distribution, but none of these approachs, in deed brought, significant improvements, but rather introduce more instability to the Actor and Critic models training.

See the graph of scores x episodes score x epsodes
![scores x #episodes](score_x_episodes.png)

And here, you can observe data above, now in a smorthed graph score x epsodes
![scores x #episodes (smorthed)](score_x_episodes_smorthed.png)


For more details about continous control with DDPG algorithm, read the [DDPG papper](https://arxiv.org/pdf/1509.02971.pdf)


#### Further works

These method are good strategies to avoid overfitting and will keep the weights of the models low, making them simple and capable to generalize well the sapce into the actions.

In addition,  it is notable the quantity of hyperparams this algorithm requires, which encourages us to try a sistematic way to find the best fit to all of them. this cold be reach by using a grid searh are eve, a stocastic optimization method, such as PSO or genetic algorithm. Of course, it will be strong computational demmanding but can bring some interesting results.

Finally, there are others algorith for continuous control in deep reiforcement learng which would be interesting to test, such as A2C and PPO, for example
