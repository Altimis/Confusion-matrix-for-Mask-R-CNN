# Confusion-matrix-for-Matterport-implementation-of-Mask-R-CNN

This repo contains a solution to extract the True Posives, False Positives and False Negative of each classe (including the Background Class) and plot the confusion matrix. The background class is being counted to cover the cases when the model miss (detect background instead of an actual object or detect an object instead of background)

Here is an example of plotting a pretty confusion matrix for 3 classes (class B, C and D) + background (class A)  

![alt text](https://github.com/Altimis/Confusion-matrix-for-Matterport-implementation-of-Mask-R-CNN/blob/master/confm.png?raw=true)

The vertical axis represent the ground-truth classes and the horizontal axis represent the predicted classes. BG class is the background classe. It is not taken into account in the calculation of the mAP. 
