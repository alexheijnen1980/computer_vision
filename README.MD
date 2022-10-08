# computer_vision

## Project overview
Per the project introduction "objective of this project is to create a convolutional neural network to detect and classify objects using data from Waymo". Objects to be detected and classified include vehicles, pedestrians and cyclists (either stationary, or moving) in different types of urban environments. An autonomous vehicle needs to be aware of the pose, velocity and heading of different types of dynamic agents to ensure safe operation in the presence of dynamic agents. More specifically, to ensure safe operation an autonomous vehicle will need to adapt its own behavior (pose, velocity and heading) in response to the current state (but also considering possible future states) of these dynamic agents. 

## Set up
All code for this project can be downloaded from github using "git clone https://github.com/alexheijnen1980/computer_vision.git". All dependencies required to run the code in the two Jupyter Notebooks are listed in the included requirements.txt file. The workflow below should suffice to install these dependencies (on a windows machine, different commands are used to create and activate a virtual environment on MacOS):
* create virtual environment: python -m venv venv
* activate virtual environment: ./venv/Scripts/Activate.ps1
* download and install dependencies: pip install -r requirements.txt

Additionally, given github limits on file size, the repository does not all code to create inferences using the models referenced below. That being said, the pipeline_new.config files used to train the different models are included in the different subsfolders in the /experiments folder.

## Dataset

### Dataset analysis
The dataset was analyzed in a number of different ways in the Exploratory Data Analysis Jupyter Notebook. Below, a short summary of findings is provided, additionals details and images are available in the Jupyter Notebook.

*Qualitative Analysis*  
Review of one image from each of the tensorflow records indicates a large variability in:
* Type, number and relative position of dynamic agents. Images include (combinations of) vehicles, pedestrians and cyclists. Some images contain large numbers of dynamic agents, whereas others don't contain any dynamic agents at all. Some images contain nearby dynamic agents with a bounding box that takes up a large amount of the image (e.g. a bounding box around the back of a bus, taking up approximately 1/5th of image). Some images only contain dynamic agents in the far distance, with bounding boxes taking up a very small amount of the image.
![](/images/agents.png "variability in agent type, number and relative position") 
* Type of (urban) environment. Some images show residential neighborhoods, others show the ego vehicle traveling through city centers, in parking lots, or on highways. 
![](/images/urban_environments.png "variability in type of urban environment")
* Time of day. Most images show daytime conditions. Fewer images show the ego vehicle traveling after the sun has set. Very few images show lighting conditions corresponding to (nearing) sunset.  
![](/images/time.png "variability in time of day")
* Environmental conditions. Some images show clear and sunny conditions. Many images show overcast conditions. Fewer images show foggy and / or rainy conditions. In some of those images water droplets can be seen on the lens of the camera.
![](/images/environmental.png "variability in environmental conditions")

*Quantitative Analysis*  
Quantitative analysis confirmed some of the observations made as part of the qualitative analysis. Different images show a large variability in the number of dynamic agents. Furthermore, the median number of vehicles is far larger than the medium number of pedestrians and cyclists. The boxplot below shows variability - across 100 images per tensorflow record - in number of bounding boxes corresponding to each of the three types of dynamic agents. Large differences exist in the median number of vehicles (15), pedestrians (1) and cyclists (0)
![](/images/agent_boxplot.png "variability in number of agents per image")
Next, the bar chart below shows the frequency of bounding boxes (dynamic agents) grouped according to location of the center of the bounding box (mapped to one of the four quadrants of the image). The print out expresses similar information in percentages, that are further split by dynamic agent type. Almost 60% of bounding boxes have a center in the bottom half of the image and designate the presence of a vehicle.   
![](/images/location.png "variability in location of bounding box center")
```
Percentage of bounding boxes grouped by class and center location: 
class       quadrant    
cyclist     bottom left      0.241667
            bottom right     0.157973
            top left         0.103049
            top right        0.068002
pedestrian  bottom right     8.996610
            bottom left      7.694641
            top left         2.879083
            top right        2.228360
vehicle     bottom left     35.409997
            bottom right    23.602829
            top left        12.572448
            top right        6.045341
```

### Cross-validation
Given that more than 10,000 images are available across all tensorflow records, 90% of the tensorflow records are assigned to the training set and 'only' 10% of the tensorflow records are assigned to the valiation set (rather than the 80/20% rule of thumb). Furthermore, the training set and the validation set are split at the level of tensorflow records (trip id), rather than at the level of images. A tensorflow record is only part of the training set, or the validation set, never of both. This is done to prevent too much similarity between the training set and the validation set.

## Training
### Reference experiment
_This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances._

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.001
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.004
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.002
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.008
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.164
```
Eval metrics at step 2500
```
DetectionBoxes_Precision/mAP:          0.000259
DetectionBoxes_Precision/mAP@.50IOU:   0.000927
DetectionBoxes_Precision/mAP@.75IOU:   0.000113
DetectionBoxes_Precision/mAP (small):  0.000000
DetectionBoxes_Precision/mAP (medium): 0.000012
DetectionBoxes_Precision/mAP (large):  0.003928
DetectionBoxes_Recall/AR@1:            0.000090
DetectionBoxes_Recall/AR@10:           0.002416
DetectionBoxes_Recall/AR@100:          0.008235
DetectionBoxes_Recall/AR@100 (small):  0.000000
DetectionBoxes_Recall/AR@100 (medium): 0.000224
DetectionBoxes_Recall/AR@100 (large):  0.164400
Loss/localization_loss:                0.927825
Loss/classification_loss:              0.699224
Loss/regularization_loss:              2.105648
Loss/total_loss:                       3.732696
```

### Improve on the reference
_This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings._

#### Changes to learning rate in experiment0

Reducing learning_rate_base reduced to 0.005.  
![](/experiments/experiment0/Loss_and_learning_rate.png "")

![](/experiments/experiment0/Example.png "")

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.109
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.222
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.100
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.043
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.431
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.542
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.027
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.114
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.161
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.096
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.525
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.601
```
Eval metrics at step 2500
```
DetectionBoxes_Precision/mAP:          0.109373
DetectionBoxes_Precision/mAP@.50IOU:   0.221568
DetectionBoxes_Precision/mAP@.75IOU:   0.100043
DetectionBoxes_Precision/mAP (small):  0.042982
DetectionBoxes_Precision/mAP (medium): 0.431410
DetectionBoxes_Precision/mAP (large):  0.542174
DetectionBoxes_Recall/AR@1:            0.026803
DetectionBoxes_Recall/AR@10:           0.113507
DetectionBoxes_Recall/AR@100:          0.160855
DetectionBoxes_Recall/AR@100 (small):  0.096337
DetectionBoxes_Recall/AR@100 (medium): 0.524813
DetectionBoxes_Recall/AR@100 (large):  0.600800
Loss/localization_loss:                0.426503
Loss/classification_loss:              0.236882
Loss/regularization_loss:              0.255292
```

* Loss seems to start to stabilize towards the end of training.
* Large difference in mAP across small medium and large targets. This is visible in example images and inference videos.

#### Adding augmentations in experiment1
* random horizontal flip is alread applied
* random crop is already applied
* [this](https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509) article also references hue jitter

## Other

### Git
* git remote add origin https://github.com/alexheijnen1980/computer_vision.git  
* git branch -M main  
* git push -u origin main 

### PEP 8 style guide
[https://peps.python.org/pep-0008/](https://peps.python.org/pep-0008/)

### Annealing
cosine_decay_learning rate:  
global_step = min(global_step, decay_steps)  
cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))  
decayed = (1 - alpha) * cosine_decay + alpha  
decayed_learning_rate = learning_rate * decayed  
