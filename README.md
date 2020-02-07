# Distilled-BagNets

The main code for the submission: "Distilled-BagNets: Leveraging Local and Global Learning Pathways in CNNs"

Important: It is assumed that the datasets have already been downloaded and each has separate directories. 

The main_test.py file runs the training of the distilled-BagNet architecture (VGG-8) on the STL-10 dataset. 
The main_test_mnist.py and main_test_fmnist.py runs the training of the distilled-BagNet architecture (VGG-7) on the MNIST
and Fashion-MNIST datasets respectively. 
The flowers_baseline.py file runs the transfer learning framework on the Flowers-102 dataset. 
The tiny_imagenet_baseline.py trains the distilled-BagNet architecture on the tiny-ImageNet framework. 


The rest of the files are libraries containing the various distilled-BagNet, vanilla-BagNet and vanilla-CNN architectures. THe imness.py file computes the spatial orderness of the feature maps (used in the analysis)


