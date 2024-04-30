from inference import dataset, video, laptop_webcam, rtsp_stream, initialize_model, video_dual, rtsp_dual
import os
from multiprocessingfile import multiprocess
import shutil


def main():
    # specify the model to be used
    model='yolov8s'  #'yolov8s','yolov8n','yolov8m','yolov8x','dynamicyolov8s','dynamicquantizedyolov8s','fp16convertedyolov8s','staticquantizedyolov8s'
    match model:
        case 'yolov8s':
            model_path='models/yolov8s.onnx'  
        case 'yolov8n':
            model_path='models/yolov8n.onnx'
        case 'yolov8m':
            model_path='models/yolov8m.onnx'
        case 'yolov8x':
            model_path='models/yolov8x.onnx'
        case 'dynamicyolov8s':
            model_path='models/yolov8sdynamic.onnx'
        case 'dynamicquantizedyolov8s':
            model_path='models/dynamic_quantized_yolov8s.onnx'
        case 'fp16convertedyolov8s':
            model_path='models/fp16_converted_yolov8s.onnx'
        case 'staticquantizedyolov8s':
            model_path='models/static_quantized_yolov8s.onnx'
        case _:
            print("Invalid model type specified. Choose from options 'yolov8s','yolov8n','yolov8m','yolov8x','dynamicyolov8s','dynamicquantizedyolov8s','fp16convertedyolov8s','staticquantizedyolov8s'.")
            exit()

    # specify type of input
    input_type = 'video'  # webcam, video, dataset, rtsp
    match input_type:
        case 'video':
            video_path='videodata/africa_30fps.mp4'
        case 'rtsp':
            rtsp_username='admin'
            rtsp_password='mantra@123'
        case 'dataset':
            dataset_dir='imagedata/images'
        case 'webcam':
            pass
        case _:
            print("Invalid input_type specified. Choose from options 'video','webcam','rtsp','dataset'.")
            exit()

    # path to yaml file
    coco_yaml_path = 'models/config_custom_data.yaml'
    
    # output directory path
    output_dir = 'outputfolder'
    if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    device='gpu'  #'GPU' or 'gpu' or 'CPU' or 'cpu'
    
    num_processes = 1 # choose 0 for no multiprocessing,1 for 1 process (multiprocessing), 2 for 2 parallel processes, 3 for 3 parallel processes
    
    streams_per_process = 1 # 1 or 2
    
    # choose the optimization method (skipping frames)
    optim_method = 'none'  #'ssim' for SSIM, 'motiondetection' for motiondetection, 'none' for no optimization method (processing all frames) 
    
    # specify input size, iou threshold, confidence threshold, ssim threshold, motion detection threshold 
    input_size = (640, 640) #get the inputs from yaml file
    iou = 0.7   
    conf = 0.3  
    ssim = 0.88 # range 0-1 (1 means similar)
    motion_detection = 25 # range for md 
    
    #####################################################################################################################

    match optim_method:
        case 'ssim':
            motion_detection = None
        case 'motiondetection':
            ssim = None
        case 'none':
            motion_detection = None
            ssim = None
        case _:
            print("Invalid optim_method specified. Choose from options 'ssim', 'motiondetection' or 'none'.")
            exit()

    match num_processes:
        case 0:
            # no multiprocessing 
            model, classes = initialize_model(model_path, coco_yaml_path, device)
            match input_type:
                case 'video':
                    match streams_per_process:
                        case 1:
                            video(model, classes, video_path, input_size, conf, iou, ssim, motion_detection, output_dir)
                        case 2:
                            video_dual(model, classes, video_path, video_path, input_size, conf, iou, ssim, motion_detection, output_dir)    
                        case _:
                            print("Invalid number of streams_per_process specified, Choose either 1 or 2 streams.")
                            exit()
                case 'webcam':
                    match streams_per_process:
                        case 1:
                            laptop_webcam(model, classes, input_size, conf, iou, ssim, motion_detection, output_dir)
                        case 2:
                            print("Multiple streams are not possible with webcam, cannot replicate more than one webcam stream at a time. Try again with streams_per_process=1")
                            exit()
                        case _:
                            print("Invalid number of streams_per_process specified, Choose either 1 or 2 streams.")
                            exit()
                case 'rtsp':
                    match streams_per_process:
                        case 1:
                            rtsp_stream(model, classes, input_size, conf, iou, ssim, motion_detection, output_dir, rtsp_username, rtsp_password)
                        case 2:
                            rtsp_dual(model, classes, input_size, conf, iou, ssim, motion_detection, output_dir, rtsp_username, rtsp_password, rtsp_username, rtsp_password)
                        case _:
                            print("Invalid number of streams_per_process specified, Choose either 1 or 2 streams.")
                            exit()
                case 'dataset':
                    match streams_per_process:
                        case 1:
                            dataset(model, dataset_dir, output_dir, input_size, conf, iou, ssim, motion_detection, classes) 
                        case 2:
                            print("Multiple streams(datasets) are not supported with dataset input. Try again with streams_per_process=1")
                            exit()
                        case 3:
                            print("Invalid number of streams_per_process specified, Choose either 1 or 2 streams.")
                            exit()       
        case _:
            if num_processes > 0:
                # multiprocessing 
                match input_type:
                    case 'video':
                        multiprocess(input_type, model_path, coco_yaml_path, input_size, iou, conf, ssim, motion_detection, output_dir, num_processes, streams_per_process, device, video_path, None, None)
                    case 'rtsp':
                        multiprocess(input_type, model_path, coco_yaml_path, input_size, iou, conf, ssim, motion_detection, output_dir, num_processes, streams_per_process, device, None, rtsp_username, rtsp_password)
                    case 'webcam':
                        print("Multiprocessing is not possible with webcam input, cannot replicate more than one webcam stream at a time. Try again with num_processes=0 for no multiprocessing")
                        exit()
                    case 'dataset':
                        print("Multiprocessing is not supported with dataset input, Try again with num_processes=0")
                        exit()
            else:
                print("Invalid num_processes specified. Choose 0 for no multiprocessing, 1 for multiprocessing with 1 process, 2 for multiprocessing with 2 processes in parallel, 3 for multiprocessing with 3 processes in parallel (>0 for multiprocessing).")
                exit()
            

if __name__ == "__main__":
    main()

