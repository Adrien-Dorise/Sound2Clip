# Model folder
In this folder are placed all the files related to the construction of a machine learning or neural network model.   
Both paradigms are managed by their respective superclasses **MachineLearning** and **NeuralNetwork**.  
These two classes are made so that they can be interchanged easily. Therefore, it is easy to switch between these two paradigms. 

## Machine learning
Machine learning models are managed by the **MachineLearning** class in *machineLearning.py*.  
You can find most of the common models in the folder *machine_learning_architectures*.  
Here, three scripts contain the most commonly used methods for **Classification**, **Regression** and **Clustering**.
For machine learning, DragonflAI uses **scikit-learn** API.

## Neural network
Neural network models are managed by the **NeuralNetwork** class in *neuralNetwork.py*.  
DragonflAI takes on the responsibility of managing the training, backward pass, prediction phase, and more. Therefore, it limits the risk of error when experimenting with complex architectures.  
However, it does not mean the control is taken out of your hands. Indeed, the class implements many callbacks and gives you complete control of the parts you need.
For neural networks, DragonflAI uses the **PyTorch** API.  