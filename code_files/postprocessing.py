import cv2
import numpy as np

def xywh2xyxy(x):
    # Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

############################## scale bounding boxes #############################################################
def scale_boxes(boxes, img1_shape, img0_shape, scale, pad):
    boxes[:, [0, 2]] -= pad[0]  # Remove padding from x-coordinates
    boxes[:, [1, 3]] -= pad[1]  # Remove padding from y-coordinates
    boxes /= scale[0]  # Adjust for the scaling applied during resize
    # Clip boxes to be within the original image size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])  # Clip x-coordinates
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])  # Clip y-coordinates
    return boxes

############################### IOU calculation #################################################################
def intersection(box1, box2):
    # Convert input boxes to numpy arrays for easier manipulation
    box1 = (np.array(box1))  
    box2 = (np.array(box2))
    # Calculate coordinates of intersection rectangle
    x1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[2], box2[2])
    y2 = np.minimum(box1[3], box2[3])
    # Calculate the area of intersection rectangle
    inter_area = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    return inter_area

def union(box1, box2):
  # Convert input boxes to numpy arrays for easier manipulation
  box1 = np.array(box1)
  box2 = np.array(box2)
  # Calculate areas of input boxes using a larger data type to prevent overflow
  box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]).astype(np.int64)
  box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]).astype(np.int64)
  
  # Calculate intersection area using the intersection function with upcasted inputs
  inter_area = intersection(box1, box2) 
  # Calculate union area, handle potential overflow with warning
  union_area = box1_area + box2_area - inter_area
  
  # Check for potential overflow in the union area calculation
  if np.issubdtype(union_area.dtype, np.integer):
    if union_area < 0:
      # Raise a runtime warning if negative union area is encountered
      raise RuntimeWarning("Encountered negative union area, potentially due to overflow. Consider handling large bounding boxes differently.")
  return union_area

def iou(box1, box2):
    # Calculate intersection area using the intersection function
    inter_area = intersection(box1, box2)
    # Calculate union area using the union function
    union_area = union(box1, box2)
    # Prevent division by zero if union_area is 0
    if union_area == 0:
        return 0 
    # Calculate and return Intersection over Union (IoU) score
    return inter_area / union_area

#################################### NMS #########################################################################
def non_max_suppression(boxes, scores, overlap_thresh):
    # Check if there are any boxes to suppress
    if len(boxes) == 0:
        return [],[]
    # Sort the boxes based on their scores in descending order
    indices = np.argsort(scores)[::-1]
    sorted_boxes = boxes[indices]
    sorted_scores = scores[indices]
    picked_indices = []

    # Pick the boxes that are not suppressed
    while len(sorted_boxes) > 0:
    # Pick the box with the highest score and remove it from the list
        current_box = sorted_boxes[0]
        picked_indices.append(indices[0])
        sorted_boxes = sorted_boxes[1:]
        indices = indices[1:]

        # Compute IoU of the picked box with the rest
        ious = np.array([iou(current_box, box) for box in sorted_boxes])
        # Keep only boxes with IoU less than the threshold
        non_overlap_indices = np.where(ious < overlap_thresh)[0]
        sorted_boxes = sorted_boxes[non_overlap_indices]
        indices = indices[non_overlap_indices]
        sorted_scores = sorted_scores[non_overlap_indices]

    # Return the boxes and scores that were not suppressed
    return boxes[picked_indices], scores[picked_indices]
    
################################### Filter output by confidence #########################################################################################
def filter_by_confidence(boxes, scores, conf_threshold):
    # Find indices of detections with confidence value greater than the confidence threshold
    # valid_indices = scores >= conf_threshold
    valid_indices = [i for i, score in enumerate(scores) if score >= conf_threshold]
    
    # Filter boxes and scores based on valid indices
    filtered_boxes = boxes[valid_indices]
    filtered_scores = scores[valid_indices]
    
    return filtered_boxes, filtered_scores

################################## Display People Count ##################################################################################
def display_people_count_patch(original_image, people_count, position=(20, 30), patch_size=(220, 40), font_scale=0.7, font_thickness=2):
    # Calculate bottom right position of the patch based on top left position and patch size
    patch_top_left = position
    patch_bottom_right = (position[0] + patch_size[0], position[1] + patch_size[1])

    # Draw the rectangle patch in white
    cv2.rectangle(original_image, patch_top_left, patch_bottom_right, (255, 255, 255), -1)  # -1 fills the rectangle

    # Position the text to be roughly in the middle of the patch
    text_position = (position[0] + 10, position[1] + patch_size[1] - 10)

    # Draw the people count text in black
    cv2.putText(original_image, f"People Count: {people_count}", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

##################################  Postprocess ################################################################################################################
def postprocess_and_log_outputs(original_image, outputs, original_img_shape, scale, pad, conf_threshold=0.3, iou_threshold=0.7, person_class_id=0):
    output = outputs[0][0]  # Assuming the first output is the desired one
    detections = output.transpose()  # Adjusting dimensions for easier manipulation
    scores = detections[:, 4]  # Assuming the scores are at index 4
    class_ids = detections[:, 5].astype(int)  # Assuming class IDs are at index 5
    boxes = xywh2xyxy(detections[:, :4]) # Convert bounding box format from [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]
    
    # Filter detections for "person" class and based on confidence threshold
    person_indices = (class_ids == person_class_id) & (scores >= conf_threshold)
    filtered_boxes = boxes[person_indices]
    filtered_scores = scores[person_indices]
    filtered_class_ids = class_ids[person_indices]
    
    # Filter detections based on confidence threshold
    filtered_boxes, filtered_scores = filter_by_confidence(boxes, scores, conf_threshold)
   
    # Apply Non-Maximum Suppression (NMS)
    nms_boxes, nms_scores = non_max_suppression(filtered_boxes, filtered_scores, iou_threshold)
    
    # If no detections after NMS, return the original image and empty list
    if len(nms_boxes) == 0 and len(nms_scores) == 0:
        return original_image , []
    else:
        # Adjust boxes for the scale and padding
        new_unpad_shape = (int(original_img_shape[0] * scale[0]), int(original_img_shape[1] * scale[1]))
        # Calculate the actual new_shape after padding is applied
        new_shape_with_pad = (new_unpad_shape[0] + 2 * int(pad[1]), new_unpad_shape[1] + 2 * int(pad[0])) 
        scaled_boxes = scale_boxes(nms_boxes, new_shape_with_pad, original_img_shape, scale, pad)
        # Format filtered detections as [x_min, y_min, x_max, y_max, score, class_id]
        filtered_detections = [[*box, score, person_class_id] for box, score in zip(scaled_boxes, nms_scores)]
        # Return the modified image and filtered detections
        return original_image, filtered_detections