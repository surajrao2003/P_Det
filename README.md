# YOLOv8 People detection 

## Device Specifications
- **CPU**: 13th Gen Intel® Core™ i9-13900K 32
- **RAM**: 32 GB
- **Cores**: 24
- **Threads**: 24 (one per each core)
- **GPU**: Nvidia GeForce RTX 3060 (12 GB)

## Instructions to Run 
- Clone the repository
- Copy the model files and yaml file from the google drive link into the 'models' folder : [yaml file and model files link](https://drive.google.com/drive/folders/19HjjttEuMpDyQ-FggPisqYlM0GaeJPsa?usp=sharing)
- Make sure the required libraries are installed.
- Execute the main_execution.py file to run inference.

## Description
This repository contains a system for Yolov8 model inference with configurable options for different input types, model selection, optimization methods like quantization, skipping frames, etc  and multiprocessing. 

All the files are execution ready with relative paths including input data provided in the directories which you can use.

## Directories
### **code_files directory** 
- **main_execution.py** - This is the main executable file , where the user can set variable values according to their use case. The variables that can be set are :-
  - **model** - choose the model to be used from the switch cases.
  - **input type** - choose either video or rtsp or dataset or webcam as input.
    - **Video** - give path to the video in the switch case (can choose videos from videodata directory).
    - **RTSP** - give username and password of the IP camera in the switch case.
    - **dataset** - give path to the dataset (can use the images in imagedata directory).
    - **webcam** - no need of any path.
  - **coco_yaml_path** - give path to the config_custom_data.yaml file which is present in the models directory.
  - **output_dir** - give path of the 'outputfolder' directory where the outputs will be stored.
  - **device** - choose the device GPU or CPU using 'gpu' or  'GPU' or 'cpu' or 'CPU'.
  - **num_processes** - 0 for no multiprocessing and >0 for multiprocessing (from observations, maximum 3 processes are supported on nvidia geforce rtx 3060 gpu, so num_processes can be 1 or 2 or 3 for multiprocessing).                                                                                                                                                                                       Note:- num_processes = 0 or num_processes = 1 will give almost similar results , just that in the case of the latter, the python multiprocessing module will be used.
  - **streams_per_process** - number of streams concurrently per process (as of now maximum 2 streams are only possible per process). So choose 1 or 2.
  - **optim_method** - choose 'none' for no optimization method (i.e every frame is processed without any skipping of frames), choose 'ssim' or 'motiondetection' for skipping of frames.
  - **input_size** - can specify input image size. Generally (640,640) is used.
  - **iou** - specify iou threshold value. 0.7 gives good results.
  - **conf** - specify confidence threshold value. 0.3 gives good results.
  - **ssim** - specify ssim threshold. 0.88 gives good results. Range 0-1 (1 means similar and no skipping).
  - **motion detection** - specify motion detection threshold. 25 gives good results.

**After setting up these variables, you can execute the main_execution.py file.**

- **inferencing.py** - Consists of functions related to model initialization and model inferencing.
- **preprocesing.py** - Consists of functions which perform preprocessing such as scaling, padding and resizing on the input data.
- **postprocessing.py** - Consists of functions related to postprocessing such as filtering based on confidence, IOU and NMS, etc.
- **multiprocessing.py** - Consists of the multiprocessing code, so when the num_processes > 0 , this file is called and it is responsible for execution of multiple processes in parallel.
- **optimization_methods.py** - Consists of optimization methods which can skip the processing of some frames based on their similarities. So methods like 'ssim' and 'motiondetection' have been used. When any of these 2 optimization methods are chosen, this file is called.

### **imagedata directory** 
-  Consists of images and labels folder. Can give the path to the images folder in the case of dataset input.

### **videodata directory**
-  Consists of few video inputs that were used for inferencing and benchmarking.

### **models directory**
- Consists of the coco_yaml_file whose path must be provided in the main_execution.py. (copy from google drive link). 
- Contains multiple models (onnx models, quantized models) that can be used (copy them from google drive link into the models directory). Most of the models are of yolov8s version because that was observed to be suitable for edge deployment in terms of memory and accuracy.

### **outputfolder directory**
- For storing the outputs of multiple processes and multiple streams with their respective numbers.

### **yolov8_quantization_methods directory** 
- Consists of different onnx quantization techniques that I used to quantize yolov8s onnx model in order to reduce its model size and then these quantized models were used for inferencing.




