import torch   
import onnxruntime as ort
import yaml
import os
import numpy as np
import time
import cv2
from optimization_methods import ssim, motion_detect
from preprocessing import preprocess_frame
from postprocessing import postprocess_and_log_outputs, display_people_count_patch

################################### Model Initialization #####################################################################

def initialize_model(model_path, coco_yaml_path, device):
    if device == 'gpu' or device == 'GPU':
        model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    elif device == 'cpu' or device == 'CPU':
        model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    else:
        print("Invalid device type specified. Choose from options 'cpu','CPU','gpu','GPU'.")
        exit()     
    # loading class names from yaml file
    with open(coco_yaml_path, 'r') as file:
        classes = yaml.safe_load(file)['names']
    return model, classes

################################### Input methods #####################################################################

# Webcam input
def laptop_webcam(model, classes, input_size, conf, iou, ssim, motion_detection, output_dir):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    # creating output directory to store output video
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "Output_webcam.avi")

    Process_frame(cap, model, classes, input_size, conf, iou, ssim, motion_detection, output_file_path, output_dir)

# RTSP input
def rtsp_stream(model, classes, input_size, conf, iou, ssim, motion_detection, output_dir, username, password, num_processes=None, process_num=None):
    if username and password:
        rtsp_url = f"rtsp://{username}:{password}@192.168.1.64:554"
    cap = cv2.VideoCapture(rtsp_url)
    # creating output directory to store output video
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "Output_rtsp.avi")
    Process_frame(cap, model, classes, input_size, conf, iou, ssim, motion_detection, output_file_path, output_dir, num_processes, process_num)

# Video input
def video(model, classes, video_path, input_size, conf, iou, ssim, motion_detection, output_dir, num_processes=None, process_num=None):
    cap = cv2.VideoCapture(video_path)
    # creating output directory to store output video
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "Output_video.avi")
    Process_frame(cap, model, classes, input_size, conf, iou, ssim, motion_detection, output_file_path, output_dir, num_processes, process_num)

# Image (dataset) input
def dataset(model, dataset_dir, output_dir, input_size, conf, iou, ssim, motion_detection, classes):
    total_time = 0
    total_frames = 0

    for image_file in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image_file)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        start_time = time.time()
        input_frame = cv2.imread(image_path)
        # perform object detection on the images
        result_image, boxes, class_ids, scores, original_img_shape =  detection(model, input_frame, input_size, conf, iou)
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
        total_frames += 1
        # construct path to save the output image
        output_image_path = os.path.join(output_dir, f"{filename}.jpg")
        # Write result image to output directory
        cv2.imwrite(output_image_path, result_image)

    # calculating average_fps
    average_fps = total_frames / total_time
    print(f"Total Frames: {total_frames}")
    print(f"Average FPS: {average_fps}")

################################## Input methods (2 streams) ###########################################################

# Video input (2 streams)
def video_dual(model, classes, video_path1, video_path2, input_size, conf, iou, ssim, motion_detection, output_dir, num_processes=None, process_num=None):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    # creating output directory to store output videos
    os.makedirs(output_dir, exist_ok=True)
    output_file_path1 = os.path.join(output_dir, "Output1_video.avi")
    output_file_path2 = os.path.join(output_dir, "Output2_video.avi")
    Process_dual_frame(cap1, cap2, model, classes, input_size, conf, iou, ssim, motion_detection, output_file_path1, output_file_path2, output_dir, num_processes, process_num) 
 
# RTSP input (2 streams)
def rtsp_dual(model, classes, input_size, conf, iou, ssim, motion_detection, output_dir, username1, password1, username2, password2, num_processes=None, process_num=None):
    if username1 and password1:
        rtsp_url1 = f"rtsp://{username1}:{password1}@192.168.1.64:554"
    if username2 and password2:
        rtsp_url2 = f"rtsp://{username2}:{password2}@192.168.1.64:554"
    cap1 = cv2.VideoCapture(rtsp_url1)
    cap2 = cv2.VideoCapture(rtsp_url2)
    # creating output directory to store output videos
    os.makedirs(output_dir, exist_ok=True)
    output_file_path1 = os.path.join(output_dir, "Output1_rtsp.avi")
    output_file_path2 = os.path.join(output_dir, "Output2_rtsp.avi")
    Process_dual_frame(cap1, cap2, model, classes, input_size, conf, iou, ssim, motion_detection, output_file_path1, output_file_path2, output_dir, num_processes, process_num)

################################### Processing frame ###################################################################

def Process_frame(cap, model, classes, input_size, conf, iou, ssim1, motion_detection, output_file_path, output_dir, num_processes=None, process_num=None):
        # Get input video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
    
        # Initialize the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))

        frame_counter = 0  # number of frames
        average_fps = 0  
        skipping_frames = False # False-> no skipping of frames , True -> skipping of frames applicable
        skipped_frames = 0  # number of skipped frames
        processed_frames = 0 # number of processed frames
        last_processed_gray = None # grayscale version of previously processed frame
        last_detection_results = None # detection results of the previously processed frame
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if(ssim1 == None and motion_detection == None):
                skipping_frames = False # no optimization case
                # perform object detection
                result_frame, boxes, class_ids, scores, _ = detection(model, frame, input_size, conf, iou)
            else:
                skipping_frames = True # optimization (ssim or motiondetection)
                if(motion_detection == None and ssim1 != None):
                    # optimization method = SSIM
                    process_this_frame, current_gray = ssim(frame, last_processed_gray, ssim1) #comparing frames based on ssim score 
                else:
                    # optimization method = motion detection
                    process_this_frame, current_gray = motion_detect(frame, last_processed_gray, motion_detection) #comparing frames based on motion detection scores
                
                if process_this_frame is True:
                    processed_frames += 1
                    # perform object detection
                    result_frame, boxes, class_ids, scores, _ = detection(model, frame, input_size, conf, iou)
                    people_count = len([id for id in class_ids if id == 0]) # counts the number of "person" objects detected in the current frame based on their class IDs.
                    last_detection_results = (boxes, scores, people_count) # Store detection results for drawing on both processed and skipped frames
                    last_processed_gray = current_gray # update previously processed frame with current frame
                else:
                    skipped_frames += 1
                    # Draw the detection results on current skipping frame using results stored from the previously processed frame
                    boxes, scores, people_count = last_detection_results
                    for box, score in zip(boxes, scores):
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                        draw_text(frame, f"P: {score:.2f}", (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), (0, 0, 0))
                    display_people_count_patch(frame, people_count) # displaying the people count value on the frame
            frame_counter += 1
            # Display the current frame with average fps until that frame
            fps = display_fps(frame, frame_counter, start_time) 
            average_fps = fps
            out.write(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if num_processes is not None and num_processes > 0:
            # multiprocessing 
            if(skipping_frames == True):
                print(f"Process {process_num} - Total Frames: {frame_counter}, Processed Frames: {processed_frames}, Skipped Frames: {skipped_frames}")
            else:
                print(f"Process {process_num} - Total Frames: {frame_counter}, Processed Frames: {frame_counter}, Skipped Frames: {0}")
        else:    
            # no multiprocessing
            if(skipping_frames == True):
                print(f"Total Frames: {frame_counter}, Processed Frames: {processed_frames}, Skipped Frames: {skipped_frames}")
            else:
                print(f"Total Frames: {frame_counter}, Processed Frames: {frame_counter}, Skipped Frames: {0}")
        
        print(f"Average FPS: {average_fps:.2f}")

################################### Processing 2 frames #########################################################33

def Process_dual_frame(cap1, cap2, model, classes, input_size, conf, iou,ssim1,motion_detection,output_file_path1,output_file_path2, output_dir,num_processes=None,process_num=None):
    # Get input video properties for both separately
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    
    frame_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    # Initialize the VideoWriter object for both inputs
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter(output_file_path1, fourcc, fps1, (frame_width, frame_height))
    out2 = cv2.VideoWriter(output_file_path2, fourcc, fps2, (frame_width, frame_height))
    
    start_time = time.time()
    frame_counter1 = 0   # number of frames video-1
    frame_counter2 = 0   # number of frames video-2
    total_frame_counter = 0  # total number of frames both videos
    processed_frames1 = 0 # number of processed frame video-1
    processed_frames2 = 0 # number of processed frame video-2
    skipping_frames = False # False-> no skipping of frames , True -> skipping of frames applicable
    skipped_frames1 = 0 # number of skipped frames video-1
    skipped_frames2 = 0 # number of skipped frames video-2
    last_processed_gray1 = None # grayscale version of previously processed frame - video-1
    last_detection_results1 = None # detection results of previously processed frame - video-1
    last_processed_gray2 = None # grayscale version of previously processed frame - video-2
    last_detection_results2 = None # detection results of previously processed frame - video-2

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        # Break loop if either video stream ends
        if not ret1 or not ret2:
            break

        if(ssim1 == None and motion_detection == None):
            skipping_frames = False # no optimization
            # perform object detection 
            result_frame1, boxes1, scores1, class_ids1, _ = detection(model, frame1, input_size, conf, iou)
            result_frame2, boxes2, scores2, class_ids2, _ = detection(model, frame2, input_size, conf, iou)
        else:
            skipping_frames = True # optimization (ssim or motion detection)
            if(motion_detection == None and ssim1 != None):
                # ssim
                process_frame1, current_gray1 = ssim(frame1, last_processed_gray1, ssim1)
                process_frame2, current_gray2 = ssim(frame2, last_processed_gray2, ssim1)
            else:
                # motion detection
                process_frame1, current_gray1 = motion_detect(frame1, last_processed_gray1, motion_detection)
                process_frame2, current_gray2 = motion_detect(frame2, last_processed_gray2, motion_detection)
    
            # Process first stream
            if process_frame1 is True:
                # perform detection (video-1)
                result_frame1, boxes1, scores1, class_ids1, _ = detection(model, frame1, input_size, conf, iou)
                people_count1 = len([id for id in class_ids1 if id == 0]) # counts the number of "person" objects detected in the current frame based on their class IDs.
                last_detection_results1 = (boxes1, scores1, people_count1)  # Store detection results for drawing on both processed and skipped frames
                last_processed_gray1 = current_gray1 # update previously processed frame with current frame
                processed_frames1 += 1
            else:
                skipped_frames1 += 1
                # Draw detection results on skipped frame using results from previously processed frame
                boxes1, scores1, people_count1 = last_detection_results1
                for box, score in zip(boxes1, scores1):
                    cv2.rectangle(frame1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    draw_text(frame1, f"P: {score:.2f}", (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), (0, 0, 0))
                display_people_count_patch(frame1, people_count1) # display people count on frame (video-1)

                # Process second stream
            if process_frame2 is True:
                # perform detection (video-2)
                result_frame2, boxes2, scores2, class_ids2, _ = detection(model, frame2, input_size, conf, iou)
                people_count2 = len([id for id in class_ids2 if id == 0]) # counts the number of "person" objects detected in the current frame based on their class IDs.
                last_detection_results2 = (boxes2, scores2, people_count2) # Store detection results for drawing on both processed and skipped frames
                last_processed_gray2 = current_gray2 # update previously processed frame with current frame
                processed_frames2 += 1
            else:
                skipped_frames2 += 1
                # Draw detection results on skipped frame using results from previously processed frame
                boxes2, scores2, people_count2 = last_detection_results2
                for box, score in zip(boxes2, scores2):
                    cv2.rectangle(frame2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    draw_text(frame1, f"P: {score:.2f}", (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), (0, 0, 0))
                display_people_count_patch(frame2, people_count2) # display people count on frame (video-2)
        
        # Display the current frames with average fps until that frame for both streams 
        frame_counter1 += 1
        fps1 = display_fps(frame1, frame_counter1, start_time) 
        average_fps1 = fps1
        out1.write(frame1)
        cv2.imshow('Stream1', frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_counter2 += 1
        fps2 = display_fps(frame2, frame_counter2, start_time) 
        average_fps2 = fps2
        out2.write(frame2)
        cv2.imshow('Stream2', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Increment frame counter for both streams
        total_frame_counter += 2
    
    if num_processes is not None and num_processes > 0:
        # multiprocessing
        if skipping_frames == True:    
            print(f"Process {process_num} - Total Frames in video1: {frame_counter1}, Processed Frames: {processed_frames1}, Skipped Frames: {skipped_frames1}")
            print(f"Process {process_num} - Total Frames in video2: {frame_counter2}, Processed Frames: {processed_frames2}, Skipped Frames: {skipped_frames2}")
        else:
            print(f"Process {process_num} - Total Frames in video1: {frame_counter1}, Processed Frames: {frame_counter1}, Skipped Frames: {0}")
            print(f"Process {process_num} - Total Frames in video2: {frame_counter2}, Processed Frames: {frame_counter2}, Skipped Frames: {0}")
    else:
        # no multiprocessing
        if skipping_frames == True:    
            print(f"Total Frames in video1: {frame_counter1}, Processed Frames: {processed_frames1}, Skipped Frames: {skipped_frames1}")
            print(f"Total Frames in video2: {frame_counter2}, Processed Frames: {processed_frames2}, Skipped Frames: {skipped_frames2}")
        else:
            print(f"Total Frames in video1: {frame_counter1}, Processed Frames: {frame_counter1}, Skipped Frames: {0}")
            print(f"Total Frames in video2: {frame_counter2}, Processed Frames: {frame_counter2}, Skipped Frames: {0}")
    
    print(f"Average FPS of video1: {average_fps1:.2f}") 
    print(f"Average FPS of video2: {average_fps2:.2f}") 

    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

#################################### Detection ###############################################################

def detection(model, frame, input_size, conf, iou):
    custom_person_id = 0  # Assuming '0' is the ID for person
    
    # Preprocess the frame
    preprocessed_frame, original_frame, original_frame_shape, scale, pad = preprocess_frame(frame, input_size)
    
    # Run the model on preprocessed frame
    outputs = model.run(None, {"images": preprocessed_frame})
    
    # Postprocess the model outputs and filter detections
    result_frame, filtered_detections = postprocess_and_log_outputs(original_frame, outputs, original_frame_shape, scale, pad, conf, iou, person_class_id = custom_person_id)
    
    people_count = 0 # initialize people count

    if not filtered_detections:
        # If there are no filtered detections, display people count patch and return
        display_people_count_patch(result_frame, people_count)
        return result_frame, [], [], [], original_frame_shape
    else:
        # Extract boxes, scores, and class IDs from filtered detections
        boxes = np.array([det[:4] for det in filtered_detections])
        scores = np.array([det[4] for det in filtered_detections])
        class_ids = np.array([det[5] for det in filtered_detections])
        
        # Draw bounding boxes around detections and update people count
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            score = scores[i]
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_text(frame, f"P: {score:.2f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), (0, 0, 0))
            people_count += 1
        
        # Display people count patch after processing all detections
        display_people_count_patch(result_frame, people_count)

        # Return the result frame, boxes, class IDs, scores, and original frame shape
        return result_frame, boxes, class_ids, scores, original_frame_shape
    
 ####################################### Display Output #####################################################################

def draw_text(frame, text, position, font, font_scale, text_color, bg_color):
    # draw text on frame
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness=2)
    x, y = position
    cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y + 10), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, 2)

def display_fps(frame, frame_counter, start_time):
    # calculate elapsed time
    elapsed_time = time.time() - start_time 
    # calculate frames per sec (fps)
    fps = frame_counter / elapsed_time if elapsed_time > 0 else 0
    fps_display = f"FPS: {fps:.2f}"
    draw_text(frame, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), (0, 0, 0))
    return fps