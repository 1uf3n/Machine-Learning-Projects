# Machine-Learning-Projects
This repository presents a collection of scientific computing and machine learning projects focused on signal processing, dimensionality reduction, and neural network design. These projects implement core computational techniques using Python, with applications ranging from acoustic signal tracking to image classification. Each report provides a comprehensive workflow—covering theoretical foundations, algorithm development, visualization, and performance evaluation.

## Project 1 – 3D Fourier Analysis for Acoustic Signal Localization
Applies 3D Fourier Transform techniques to detect and track a moving object based on noisy acoustic measurements. A Gaussian filter is designed and implemented to isolate dominant frequency components, enabling accurate reconstruction of the object's 2D and 3D trajectory in space.

## Project 2 – Motion Classification Using PCA and Kernel Methods
Analyzes motion-capture data from a humanoid robot to classify activities such as walking, running, and jumping. Implements Principal Component Analysis for dimensionality reduction and builds a centroid-based classifier. Kernel PCA is also tested to explore nonlinear structures and compare classification accuracy.

## Project 3 – Handwritten Digit Recognition with PCA and Machine Learning
Explores digit classification using the MNIST dataset. Implements PCA for feature extraction and compares Ridge Regression, K-Nearest Neighbors, and Support Vector Machines. Both binary and multi-class classification tasks are evaluated, with confusion matrices and cross-validation used for performance analysis.

## Project 4 – Fully Connected Neural Networks for Image Classification
Develops a configurable fully connected neural network (FCN) to classify images from the FashionMNIST dataset. Explores the impact of optimizers, learning rates, dropout regularization, batch normalization, and weight initialization strategies on model accuracy and generalization.

## Project 5 – Comparative Study of FCNs and CNNs for FashionMNIST
Builds and compares several fully connected and convolutional neural network (CNN) architectures under fixed parameter budgets. Evaluates how model structure and size influence classification accuracy, training time, and overfitting behavior. Includes detailed benchmarking across learning rates and architecture configurations.

## Paper – Comparative Analysis of Neural Network Architectures for Image Classification
This paper presents a comparative analysis of two primary neural network architectures—Fully Connected Networks (FCNs) and Convolutional Neural Networks (CNNs)—for image classification on the FashionMNIST dataset. Multiple model variants with varying parameter bud- gets are designed and trained using the AdamW optimizer and StepLR learning-rate scheduling. Comparative experiments highlight how architectural choices and model capacity influence classifi- cation accuracy, training efficiency, and generalization performance.
