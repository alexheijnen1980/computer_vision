# computer_vision

## Project overview
Per the project introduction "objective of this project is to create a convolutional neural network to detect and classify objects using data from Waymo". Objects to be detected and classified include vehicles, pedestrians and cyclists (either stationary, or moving) in different types of urban environments. An autonomous vehicle needs to be aware of the pose, velocity and heading of different types of dynamic agents (as well as its own) to ensure safe operation in the presence of dynamic agents. More specifically, to ensure safe operation an autonomous vehicle will need to adapt its own behavior (pose, velocity and heading) in response to the current state (but also considering possible future states) of these dynamic agents. 

## Set up
All code for this project can be downloaded from Github using "git clone https://github.com/alexheijnen1980/computer_vision.git". All dependencies required to run the code in the two Jupyter Notebooks are listed in the included requirements.txt file. The workflow below should suffice to install these dependencies (on a windows machine, different commands are used to create and activate a virtual environment on MacOS):
```
create virtual environment: python -m venv venv
activate virtual environment: ./venv/Scripts/Activate.ps1
download and install dependencies: pip install -r requirements.txt
```

Additionally, given Github limits on file size, the repository does not all code to create inferences using the models referenced below. That being said, the pipeline_new.config files used to train the different models are included in the different subfolders in the /experiments folder.

## Dataset

### Dataset analysis
The dataset was analyzed in a number of different ways in the Exploratory Data Analysis Jupyter Notebook. Below, a short summary of findings is provided, additional details and images are available in the Jupyter Notebook.

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
Quantitative analysis confirmed some of the observations made during the qualitative analysis. Different images show a large variability in the number of dynamic agents. Furthermore, the median number of vehicles is far larger than the median number of pedestrians and cyclists. The boxplot below shows variability - across 100 images per tensorflow record - in number of bounding boxes corresponding to each of the three types of dynamic agents. Large differences exist in the median number of vehicles (15), pedestrians (1) and cyclists (0)
![](/images/agent_boxplot.png "variability in number of agents per image")  
Next, the bar chart below shows the frequency of bounding boxes (dynamic agents) grouped according to location of the center of the bounding box (mapped to one of the four quadrants of the image). The print out expresses similar information in percentages that are further split up by dynamic agent type. Almost 60% of bounding boxes have a center in the bottom half of the image and designate the presence of a vehicle. This is likely due to the fact that often (a portion of) the top half of the image is taken up by the horizon. 
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
Given that more than 10,000 images are available across all tensorflow records, 90% of the tensorflow records are assigned to the training set and 'only' 10% of the tensorflow records are assigned to the validation set (rather than the 80/20% rule of thumb). Furthermore, the training set and the validation set are split at the level of tensorflow records (trip id), rather than at the level of images. A tensorflow record is only part of the training set, or the validation set, never of both. This is done to prevent too much similarity between the training set and the validation set.

## Training
### Reference experiment  
The first experiment was performed without making any modifications to the pipeline_new.config file. Some relevant sections from the pipeline_new.config file are copied below for reference.
```
train_config {
  batch_size: 2
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.04
          total_steps: 2500
          warmup_learning_rate: 0.013333
          warmup_steps: 200
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
}
``` 

*Loss function*  
After improving in the first 600 steps, the improvement in total loss slows down and starts to plateau. Per lesson 4 section 21 on visualizing the loss function, this type of curve is indicative of a learning rate that is too high, potentially resulting in an inability to find the global minimum (rather than a local minimum).
```
Loss/localization_loss:                 0.957403
Loss/classification_loss:               0.716043
Loss/regularization_loss:              19.057518
Loss/total_loss:                       20.730965
```    
![](/experiments/reference/Loss_and_learning_rate.png "")  

*Metrics*  
Independent of the root cause, the evaluation output below indicates that model performance is not good. Even at an Intersection over Union value of 0.50, all but one of the average precision and average recall values is below 0.01. 
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.001
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.003
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.001
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.007
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.005
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.131
```

*Example*  
Although only a single image, the example below indicates that - in conditions that were confirmed to be present in the training data - the model did not detect any vehicles. This is further confirmation of the poor performance of the model. Final confirmation of the lack of utility of the model came in the form of the complete lack of detections in (animations of) the three records in the test directory. These animations could not be uploaded to Github due to their size. 

![](/experiments/reference/Example.png "")

*Other observations*  
Finally, while rerunning the training process (using the same training data and pipeline_new.config file) consistently resulted in a model with extremely poor performance, loss values (and metrics) did vary. The regularization and total loss values below are much lower than the values presented above. I assume this is an artifact of a stochastic element in the training process (stochastic gradient descent). 
```
Loss/localization_loss:                0.927825
Loss/classification_loss:              0.699224
Loss/regularization_loss:              2.105648
Loss/total_loss:                       3.732696
```

### Experiment0: improve on the reference
Given the shape of the loss curve, a second experiment was performed after reducing the warmup learning rate from 0.013333 to 0.001, and after reducing the base learning rate from 0.04 to 0.005.
```
learning_rate {
  cosine_decay_learning_rate {
    learning_rate_base: 0.005
    total_steps: 2500
    warmup_learning_rate: 0.001
    warmup_steps: 200
  }
}
```

*Loss function*  
This time, the improvement in total loss remains more consistent throughout the entire training process. Especially the regularization loss and the total loss reach a final value that is far lower than in the reference experiment. 
```
Loss/localization_loss:                0.426503
Loss/classification_loss:              0.236882
Loss/regularization_loss:              0.255292
```
![](/experiments/experiment0/Loss_and_learning_rate.png "")

*Metrics*  
The evaluation output below indicates improved model performance as well. Especially for medium and large objects model performance (both precision and recall) has improved dramatically. Much lower scores for precision and recall indicate the model still struggles with small objects. 
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

*Example*  
This time the model detected some of the larger vehicles in the same example image. Final confirmation of the improved utility of the model was observed in (animations of) the three records in the test directory. These animations could not be uploaded to Github due to their size.

![](/experiments/experiment0/Example.png "")

### Experiment1: add hue shift to improve on experiment0
Both the reference experiment and experiment0 already included the following two augmentations: (1) random horizontal flip, and (2) random crop. After review of [this](https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509) article hue jitter was added to help the model generalize over colors (especially vehicles can have many different colors). This augmentation is expected to be beneficial for model performance, but not expected to fundamentally address the poor performance in detecting small objects. The example image below illustrates the resulting hue shift (especially visible in the sky).  
![](/images/hue.png "hue shift")
```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  data_augmentation_options {
    random_adjust_hue {
      max_delta: 0.5
    }
  }
```

*Loss function - including hue shift*
All reported loss values are higher than those in experiment0.
```
Loss/localization_loss:                0.501569
Loss/classification_loss:              0.350191
Loss/regularization_loss:              0.273898
Loss/total_loss:                       1.125657
```

*Metrics - including hue shift*
All reported precision and recall values are worse than those in experiment0. This augmentation did not result in a better performing model.
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.059      
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.110
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.058
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.017
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.216
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.292
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.013
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.059
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.095
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.051
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.274
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.350
```

### Experiment2: reduce jpeg quality to improve on experiment0
In this experiment image quality was intentionally degraded in an attempt to train using images with 'reduced information'. Objective was to improve the ability of the model to detect objects with less information. E.g. objects that are far away and represented by fewer pixels than objects that are closer by (and / or larger). The example image below illustrates the (maximum) reduction in image quality (especially visible in the asphalt).  
![](/images/quality.png "reduced jpeg quality")
```
data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  data_augmentation_options {
    random_jpeg_quality {
      min_jpeg_quality: 25
      max_jpeg_quality: 100
    }
  }
```

*Loss function - including reduced jpeg quality*  
All reported loss values are higher than those in experiment0.
```
Loss/localization_loss:                0.460015
Loss/classification_loss:              0.306136
Loss/regularization_loss:              0.265595
Loss/total_loss:                       1.031746
```

*Metrics - including hue shift*  
All reported precision and recall values are worse than those in experiment0. This augmentation did not result in a better performing model.
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.075
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.148
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.071
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.023
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.336
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.020
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.092
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.135
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.075
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.475
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.549
```

### Experiment3: increase number of steps to 6,000
Since the augmentations added in experiment1 and experiment2 respectively did not improve model performance, a final experiment (experiment3) was performed with the initial two augmentations only (random horizontal flip and random crop image). However, in this final experiment the total number of steps was increased from 2,500 steps to 6,000 steps.  

*Loss function - 6,000 steps* 
All reported loss values are very slightly better than those in experiment0. 
```
Loss/localization_loss:                0.402373
Loss/classification_loss:              0.236331
Loss/regularization_loss:              0.255268
Loss/total_loss:                       0.893973
```

![](/experiments/experiment3/Loss_and_learning_rate.png "")

*Metrics - 6,000 steps*  
All but one of the reported precision and recall values are better than those in experiment0. That being said, the improvements seem very small and the largest deficiency of the model - detection of small objects - remains.
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.113
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.223
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.102
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.435
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.026
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.119
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.170
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.106
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.531
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.630
```
![](/experiments/experiment3/Example.png "")

## Conclusion  
Even though a model was successfully trained, it proved difficult to improve upon the performance of the initial model. As is, even the performance of the model from experiment3 does not seem like it would be sufficient to safely operate around dynamic agents. I can think of a number of different ways to address this problem:
* improve model performance in different ways than investigated above, including selection of a different model architecture
* leverage additional sensing modalities (e.g. radar, lidar) and sensor fusion to further improve perception performance
* improve image resolution above 640 x 640 pixels to improve 'information density' for objects in the distance