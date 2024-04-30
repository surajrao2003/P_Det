import cv2
import numpy as np

##################################### SSIM optimization ########################################################################
def ssim(current_frame, last_processed_gray, threshold):
    # Convert current frame to grayscale
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Always process if there's no last processed frame
    if last_processed_gray is None:
        return True, current_gray
    
    # Calculate SSIM between current frame and last processed frame
    ssim_score = cv2.quality.QualitySSIM_compute(current_gray, last_processed_gray)[0][0]
    
    # SSIM Ranges from -1 to 1
    # 1 indicates perfect similarity 
    # Values closer to -1 indicate a high degree of dissimilarity
    
    if ssim_score < threshold: # if SSIM score is lesser than given threshold between last processed frame and current frame 
        process_this_frame = True
        # Process the current frame
    else:
        process_this_frame = False # if SSIM score is greater than given threshold between last processed frame and current frame 
        # skips current frame

    return process_this_frame, current_gray

###################################### Motion detection optimization ###########################################################
def motion_detect(current_frame, last_processed_gray, threshold):
    # Convert current frame to grayscale
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # median_blurred_image = cv2.medianBlur(current_gray, 5) 
    
    if last_processed_gray is None:
        # Duplicate the initial grayscale frame side by side
        # frame_diff = cv2.absdiff(current_gray, current_gray)
        side_by_side = cv2.hconcat([current_gray, current_gray])
        process_this_frame = True
    else:
        # Calculate the absolute difference between the current frame and the last processed frame
        frame_diff = cv2.absdiff(current_gray, last_processed_gray)

        # Threshold the difference
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Apply Median Blur to the thresholded image
        thresh_blurred = cv2.medianBlur(thresh, 5)

        # Calculate change percentage based on the blurred threshold image
        change_percentage = np.sum(thresh_blurred) / (current_gray.size)

        # side_by_side = cv2.hconcat([frame_diff, thresh_blurred])
        if change_percentage > threshold / 100:  # Convert threshold to a percentage
            process_this_frame = True
        else:
            process_this_frame = False

    return process_this_frame, current_gray