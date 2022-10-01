# computer_vision

## Git
* git remote add origin https://github.com/alexheijnen1980/computer_vision.git  
* git branch -M main  
* git push -u origin main 

## PEP 8 style guide
[https://peps.python.org/pep-0008/](https://peps.python.org/pep-0008/)

## Summary

## Build

## Functions and files

## Exploratory Data Analysis

## Training
cosine_decay_learning rate:
global_step = min(global_step, decay_steps)
cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
decayed = (1 - alpha) * cosine_decay + alpha
decayed_learning_rate = learning_rate * decayed

### Results experiment0 - learning_rate_base reduced to 0.005
* Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.109
* Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.222
* Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.100
* Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.043
* Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.431
* Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.542
* Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.027
* Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.114
* Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.161
* Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.096
* Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.525
* Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.601

Eval metrics at step 2500
* DetectionBoxes_Precision/mAP: 0.109373
* DetectionBoxes_Precision/mAP@.50IOU: 0.221568
* DetectionBoxes_Precision/mAP@.75IOU: 0.100043
* DetectionBoxes_Precision/mAP (small): 0.042982
* DetectionBoxes_Precision/mAP (medium): 0.431410
* DetectionBoxes_Precision/mAP (large): 0.542174
* DetectionBoxes_Recall/AR@1: 0.026803
* DetectionBoxes_Recall/AR@10: 0.113507
* DetectionBoxes_Recall/AR@100: 0.160855
* DetectionBoxes_Recall/AR@100 (small): 0.096337
* DetectionBoxes_Recall/AR@100 (medium): 0.524813
* DetectionBoxes_Recall/AR@100 (large): 0.600800
* Loss/localization_loss: 0.426503
* Loss/classification_loss: 0.236882
* Loss/regularization_loss: 0.255292

### Analysis experiment0
* Loss seems to start to stabilize towards the end of training.
* Large difference in mAP across small medium and large targets. This is visible in example images and inference videos.

## Model Improvements
