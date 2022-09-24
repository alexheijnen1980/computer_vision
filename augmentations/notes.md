Generate inference video for each file:
1. segment-12012663867578114640_820_000_840_000_with_camera_labels
2. segment-1208303279778032257_1360_000_1380_000_with_camera_labels
3. segment-12200383401366682847_2552_140_2572_140_with_camera_labels

## reference experiment
### Augmentations
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
### Results
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000    
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.001    
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000    
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000    
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.004    
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.002    
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000    
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.001    
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.012    
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005    
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.011    
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.131    

## experiment0
### Augmentations
No augmentations
### Results
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.252  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.107  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.052  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.463  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.582  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.029  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.122  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.176  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.110  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.541  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.735   

## experiment1
### Augmentations
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
### Results
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.110  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.224  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.098  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.043  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.430  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.626  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.027  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.116  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.162  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.098  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.534   
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.691  