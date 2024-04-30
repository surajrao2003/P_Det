from inference import dataset, video, laptop_webcam, rtsp_stream, initialize_model, video_dual, rtsp_dual
import os
import multiprocessing

def process_input(process_num, model_path, coco_yaml_path, input_type, output_dir, iou, conf, input_size, ssim, motion_detection, num_processes, streams_per_process, device, video_path, rtsp_username, rtsp_password):
    model, classes = initialize_model(model_path, coco_yaml_path, device)
    
    # creating process wise output directories
    process_output_dir = os.path.join(output_dir, f"process_{process_num}")
    os.makedirs(process_output_dir, exist_ok=True)
    
    match input_type:
        case 'video':
            match streams_per_process:
                case 1:
                    video(model, classes, video_path, input_size, conf, iou, ssim, motion_detection, process_output_dir, num_processes, process_num)
                case 2:
                    video_dual(model, classes, video_path, video_path, input_size, conf, iou , ssim, motion_detection, process_output_dir, num_processes, process_num)
                case _:
                    print("Invalid number of streams per process. Try again with 1 or 2")
                    exit()
        case 'rtsp':
            match streams_per_process:
                case 1:
                    rtsp_stream(model, classes, input_size, conf, iou, ssim, motion_detection, process_output_dir, rtsp_username, rtsp_password, num_processes, process_num)
                case 2:
                    rtsp_dual(model, classes, input_size, conf, iou , ssim, motion_detection, process_output_dir, rtsp_username, rtsp_password, rtsp_username, rtsp_password, num_processes, process_num)
                case _:
                    print("Invalid number of streams per process. Try again with 1 or 2")
                    exit()


################################# creating multiple processes ##################################################### 

def multiprocess(input_type, model_path, coco_yaml_path, input_size, iou, conf, ssim, motion_detection, output_dir, num_processes, streams_per_process, device, video_path, rtsp_username, rtsp_password): 
    inputs = []
    for i in range(num_processes):
        inputs.append((i+1, model_path, coco_yaml_path, input_type, output_dir, iou, conf, input_size, ssim, motion_detection, num_processes, streams_per_process, device, video_path, rtsp_username, rtsp_password))           
    
    processes = []
    for input_args in inputs:
        p = multiprocessing.Process(target=process_input, args=input_args)
        processes.append(p)

    # Start all processes
    for p in processes:
        p.start()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
