import torch
import utils.utils 
from utils.utils  import *

TOTAL_NUM_OF_CLASSES = 91


def evaluate(model, validation_loader, device):

    model.eval()

    correct = 0
    for images, bboxes, labels in validation_loader:

        loc_hat, conf_hat = model(images)


        filtered_boxes_stack = []
        
        for label in labels[0]:
            pred_labels_pos = torch.argmax(conf_hat, 1)
            scores = conf_hat[:, label - 1] 
            
            filtered_boxes = NonMaxSuppression(loc_hat, scores)
            filtered_boxes_stack.append(filtered_boxes)


        average_precisions, mean_average_precision = mAP_score(
                loc_hat, pred_labels_pos, conf_hat, bboxes, labels, TOTAL_NUM_OF_CLASSES + 1, device)

    return average_precisions, mean_average_precision

