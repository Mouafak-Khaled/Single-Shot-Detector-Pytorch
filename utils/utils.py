import torch


"""
Arrange candidates for this class in order of decreasing likelihood.

Consider the candidate with the highest score. Eliminate all candidates with lesser scores that have a Jaccard overlap of more than, say, 0.5 with this candidate.

Consider the next highest-scoring candidate still remaining in the pool. Eliminate all candidates with lesser scores that have a Jaccard overlap of more than 0.5 with this candidate.

Repeat until you run through the entire sequence of candidates.

"""

"""
 - bboxes: (num of remaining bounding boxes x 4) -> boundinx boxes for each class in the image
 - scores: (num of remaining bounding boxes) -> score for each bounding box for the same class
"""
def NonMaxSuppression(bboxes, scores, threshold=0.5):

    x_coordinate_1 = bboxes[:, 0]
    x_coordinate_2 = bboxes[:, 1]
    y_coordinate_1 = bboxes[:, 2]
    y_coordinate_2 = bboxes[:, 3]

    areas = torch.mul((x_coordinate_2 - x_coordinate_1), (y_coordinate_2 - y_coordinate_1))

    ordered_indices = scores.argsort(descending=True)
    filtered_boxes = []

    while len(ordered_indices) != 0:

        highest_score_index = ordered_indices[0]
        filtered_boxes.append(bboxes[highest_score_index])
        ordered_indices = ordered_indices[1:]


        new_x_coordinate_1 = torch.index_select(x_coordinate_1, dim = 0, index = ordered_indices)
        new_x_coordinate_2 = torch.index_select(x_coordinate_2, dim = 0, index = ordered_indices)
        new_y_coordinate_1 = torch.index_select(y_coordinate_1, dim = 0, index = ordered_indices)
        new_y_coordinate_2 = torch.index_select(y_coordinate_2, dim = 0, index = ordered_indices)


        new_x_coordinate_1  = torch.max(new_x_coordinate_1, x_coordinate_1[highest_score_index])
        new_x_coordinate_2  = torch.min(new_x_coordinate_2 , x_coordinate_2[highest_score_index])
        new_y_coordinate_1  = torch.max(new_y_coordinate_1 , y_coordinate_1[highest_score_index])
        new_y_coordinate_2  = torch.min(new_y_coordinate_2, y_coordinate_2[highest_score_index])

        height = torch.clamp((new_y_coordinate_2 - new_y_coordinate_1), min=0.0)
        width = torch.clamp((new_x_coordinate_2 - new_x_coordinate_1), min=0.0)

        remaining_areas = torch.index_select(areas, dim = 0, index=ordered_indices)
        intersection_area = height * width
        union = torch.add(torch.sub(remaining_areas, intersection_area), areas[highest_score_index])

        IoU = intersection_area / union
        ordered_indices = ordered_indices[IoU < threshold]
    
    return filtered_boxes



def get_corresbonding_images(labels):

    """
    labels: A list of tensors -> num_images x num_objects' labels
    """
    images, N = [], len(labels)
    for i in range(N):
        images.extend([i] * labels[i].size(0))
    return torch.LongTensor(images)




def calculateIoU(set_1, set_2):
    """
    Find the Intersection over Union (IoU) overlap of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """


    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)) 
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1]) 
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  


    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  

    IoU = intersection / union
    return IoU  





"""
 - detected_boxes: list of tensors. Each tensor contains the bboxes of detected objects in an image -> num_images x num_objects' bboxes
 - y_pred: list of tensors. Each tensor contains the predicted class label of detected objects in an image -> num_images x num_objects' class labels
 - scores: list of tensors. Each tensor contains the predicted confidence score of detected objects in an image -> num_images x num_objects' score
 - gt_boxes: list of tensors. Each tensor contains the true bboxes of actual objects in an image -> num_images x num_objects' true bboxes
 - y_true: list of tensors. Each tensor contains the true label of actual objects in an image -> num_images x num_objects' true bboxes
 - device: cuda or cpu

 - CITATION: Half of this code implementation belong to:
   https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
   https://blog.paperspace.com/mean-average-precision/#:~:text=To%20evaluate%20object%20detection%20models,model%20is%20in%20its%20detections.&text=Mean%20Average%20Precision%20(mAP)%20for%20Object%20Detection
   https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

"""
def mAP_score(detected_boxes, y_pred, scores, gt_boxes, y_true, num_classes, device):


    true_images = get_corresbonding_images(y_true).to(device)
    detected_images = get_corresbonding_images(y_pred).to(device)
    gt_boxes = torch.cat(gt_boxes, dim=0)
    y_true = torch.cat(y_true, dim=0)
    detected_boxes = torch.cat(detected_boxes, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    scores = torch.cat(scores, dim=0)

    average_precisions = torch.zeros((num_classes - 1), dtype=torch.float)
    # true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)),
    #                                          dtype=torch.uint8).to(device) 
    for c in range(1, num_classes):

        true_class_images = true_images[y_true == c] 
        true_class_boxes = gt_boxes[y_true == c]  

       
        # Extract only detections with this class
        detected_class_images = detected_images[y_pred == c]  
        detected_class_boxes = detected_boxes[y_pred == c]  
        detected_class_scores = scores[y_pred == c]  
        num_class_detections = detected_class_boxes.size(0)

        if num_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        detected_class_scores, indices = torch.sort(detected_class_scores, dim=0, descending=True)  
        detected_class_images = detected_class_images[indices]  
        detected_class_boxes = detected_class_boxes[indices]  

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((num_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((num_class_detections), dtype=torch.float).to(device)  # (n_class_detections)

        for d in range(num_class_detections):
            detection_box = detected_class_boxes[d].unsqueeze(0)  # (1, 4)
            image = detected_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == image]  # (n_class_objects_in_img)

            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue



            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = calculateIoU(detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, index = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_index = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == image][index]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:

                # If this object has already not been detected, it's a true positive
                if true_class_boxes_detected[original_index] == 0:
                    true_positives[d] = 1
                    true_class_boxes_detected[original_index] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                else:
                    false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1
            
        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / num_class_detections  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    # average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


