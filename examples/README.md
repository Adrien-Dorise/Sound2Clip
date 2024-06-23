# examples folder
In this folder, you will find many use cases aimed to help you understand the best way to use DragonflAI.  
you can use these examples as starting points for your projects.  

## Validation
A validation file is available and test the expected results of each example.
When modifying DragonflAI, the validations must be run to verify that the new code does not break the package.  

## Examples
Here are the currently available examples:
## Machine Learning
### blob
This example uses all machine learning paradigms (classification, clustering, regression) on a synthetic blob dataset.
- Task: Classification + Regression + Clustering 
- Input: 3 packs of blobs
- Model: All ML models available in *dragonflai/model/machine_learning_architecture/* are tested at once.
- Output: Plots are available in the *outputs/* folder 

### mnist_grid_search
This example performs a grid search on all available machine learning classification models.
- Task: Grid search on classification model
- Input: MNIST dataset (digit images) splits between train and test set.
- Model: Grid search on all classification models, then, the best model (an SVM model) is selected and trained.
- Output: Grid search results as CSV files.

## Neural Network
### UNet_PET
Segmentation on animals pictures coming from the OxfordIIITPetsAugmented dataset
- Task: Segmentation
- Input: animal image with class for each pixel
- Model: UNet neural network
- Output: Class for each pixel in the image

### VIT_MNIST
Multi-class Classification on MNIST dataset using a Transformer
- Task: Multi-class classification
- Input: MNIST dataset (digit images)
- Model: Visual Transformer
- Output: Confusion matrix