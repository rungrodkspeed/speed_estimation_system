import numpy as np

def iou(bb_test, bb_gt):
    """
    Computes IoU between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
              (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)

def greedy_associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_trackers = list(range(len(trackers)))

    if len(trackers) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    for d, det in enumerate(detections):
        best_iou = 0
        best_t = -1
        for t, trk in enumerate(trackers):
            iou_score = iou(det[:4], trk[:4])
            if iou_score > best_iou and iou_score > iou_threshold:
                best_iou = iou_score
                best_t = t
                
        if best_t != -1:
            matches.append([d, best_t])
            # del unmatched_detections[d]
            # del unmatched_trackers[best_t]
            if d in unmatched_detections:
                unmatched_detections.remove(d)

            if best_t in unmatched_trackers:
                unmatched_trackers.remove(best_t)
            
    matches = np.array(matches)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)