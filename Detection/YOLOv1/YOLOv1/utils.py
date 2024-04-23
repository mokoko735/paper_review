import torch
import numpy as np
import cv2
from collections import Counter


def IoU(bboxes_preds, bboxes_targets):
    '''
    parameters:
        bboxes_preds(tensor): (batch_size, 4), normalized [x_center, y_center, width, height]
        bboxes_targets(tensor): (batch_size, 4), normalized [x_center, y_center, width, height]

    returns:
        intersection_over_union(tensor): (batch_size, 1)
    '''

    # normalized [x_center, y_center, width, height] -> normalized [x_min, y_min, x_max, y_max]
    box1_x1 = bboxes_preds[..., 0:1] - bboxes_preds[..., 2:3] / 2
    box1_y1 = bboxes_preds[..., 1:2] - bboxes_preds[..., 3:4] / 2
    box1_x2 = bboxes_preds[..., 0:1] + bboxes_preds[..., 2:3] / 2
    box1_y2 = bboxes_preds[..., 1:2] + bboxes_preds[..., 3:4] / 2
    box2_x1 = bboxes_targets[..., 0:1] - bboxes_targets[..., 2:3] / 2
    box2_y1 = bboxes_targets[..., 1:2] - bboxes_targets[..., 3:4] / 2
    box2_x2 = bboxes_targets[..., 0:1] + bboxes_targets[..., 2:3] / 2
    box2_y2 = bboxes_targets[..., 1:2] + bboxes_targets[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp(0) -> bbox가 겹치지 않는 경우 최소값 0 설정
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    intersection_over_union = intersection / (box1_area + box2_area - intersection + 1e-6)

    return intersection_over_union

def NMS(bboxes, iou_threshold, prob_threshold):
    '''
    parameters:
        bboxes(tensor): (2*S*S, 6), [predicted_class, confidence_score, x, y, w, h]

    returns:
        bboxes_after_nms(list):
            NMS 수행 이후 남은 bbox(tensor)들을 list 형태로 구성
            (#, 6), [predicted_class, confidence_score, x, y, w, h]
    '''
    
    # pop 함수 사용을 위해 tensor로 구성된 list를 생성
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes
            if box[0] != chosen_box[0]
            or IoU(chosen_box[2:], box[2:]) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mAP(pred_bboxes, true_bboxes, iou_threshold, num_classes):
    '''
    parameters:
        pred_bboxes(list) & true_bboxes(list):
            tensor로 구성된 list 값
            (#, 7), [train_idx, class_prediction, prob_score, x, y, w, h]
    returns:
        average_precisions(float)
        mean_average_precision(float)
    '''
    
    average_precisions = []
    epsilon = 1e-6 # AP 계산 시 분모가 0이 되는 것을 방지

    # 연산 속도 향상을 위해 list type에서 tensor type으로 변경
    tensor_pred_bboxes = torch.zeros(len(pred_bboxes), 7)
    tensor_true_bboxes = torch.zeros(len(true_bboxes), 7)
    for i, bbox in enumerate(pred_bboxes):
        tensor_pred_bboxes[i] = bbox
    for i, bbox in enumerate(true_bboxes):
        tensor_true_bboxes[i] = bbox

    for c in range(num_classes):
        # 클래스 별 AP 계산을 위해 현재 클래스에 해당하는 detections와 ground_truths를 수집
        detections = tensor_pred_bboxes[tensor_pred_bboxes[:, 1] == c]
        ground_truths = tensor_true_bboxes[tensor_true_bboxes[:, 1] == c]

        # 특정 이미지(train_idx)의 박스 개수를 카운트하여 dict 형태로 변환
        # ex) {0:3, 1:5, 2:1,...}
        train_idx = np.array(ground_truths[:, 0], dtype=np.int32) # torch 타입은 Counter에 적용 불가
        amount_bboxes = Counter(train_idx)
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val) # 이미지별로 박스의 개수만큼 0을 설정
        
        # confidence score를 기준으로 성능 비교를 위해 내림차순 정렬
        indices = torch.argsort(detections, dim=0, descending=True)[:, 2] # 2번이 prob_score
        detections = detections[indices, :]
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for detection_idx in range(len(detections)):
            # AP 계산을 위해서는 같은 클래스와 같은 이미지의 박스들을 비교해야 함
            # 위에서 클래스는 구분했고 여기에서 현재 뽑은 detection의 박스와 같은 이미지의 gt 박스를 수집
            detection = detections[detection_idx]
            ground_truth_img = ground_truths[ground_truths[:, 0] == detection[0]]
            
            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                # 수집한 gt들을 detection과 비교하여 가장 높은 iou를 가지는 gt가 무엇인지 확인
                iou = IoU(detection[3:], gt[3:])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # best_iou가 iou_threshold 기준치를 넘는 경우
                # 몇번째 이미지의 몇번째 gt 박스가 활성화 되는지를 체크 후 TP 카운트
                # detection[0]는 tensor 타입이기 때문에 인덱스 적용을 위해 정수 변환
                if amount_bboxes[int(detection[0])][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[int(detection[0])][best_gt_idx] = 1

                # amount_bboxes가 이미 활성화 된 경우 같은 gt 박스에 대한 중복되는 결과물이므로 FP에 해당
                else:
                    FP[detection_idx] = 1

            # best_iou가 iou_threshold 기준치에 미달하는 경우 FP에 해당
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        # torch.trapz 함수를 이용하여 면적 계산 시 precisions과 recalls의 첫 값이 1과 0으로 시작해야 함
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
        
    average_precisions = np.array(average_precisions, dtype=np.float64)
    mean_average_precision = sum(average_precisions) / len(average_precisions)
    mean_average_precision = np.array(mean_average_precision, dtype=np.float64)

    return average_precisions, mean_average_precision

def cell_to_xywh(predictions, S=7, format='pred'):
    '''
    parameters:
        predictions:
            (batch_size, S, S, C+2*B), [class, score1, box1, score2, box2]
            predictions의 box 형태는 grid cell 형태
                - j, i: 0~6의 index
                - x, y: 0~1의 실수
                - w, h: 0~7의 실수
            grid cell 형태 -> normalized [x_center, y_center, width, height] 형태로 변환 수행
                - x = (j + x) / S
                - y = (i + y) / S
                - w = w / S
                - h = h / S
        
    returns:
        converted_preds:
            (batch_size, 2*S*S, 6), 49개의 그리드에 박스가 2개로 총 98개의 박스
            [predicted_class, confidence_score, x, y, w, h]
    '''

    batch_size = predictions.shape[0]

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    predicted_class = predictions[..., 0:20].argmax(-1).unsqueeze(-1) # (batch_size, S, S, 1)

    if format == 'target':
        # box1 변환
        confidence_score = predictions[..., 20:21] # (batch_size, S, S, 1)
        x = (predictions[..., 21:22] + cell_indices) / S # (batch_size, S, S, 1)
        y = (predictions[..., 22:23] + cell_indices.permute(0, 2, 1, 3)) / S # (batch_size, S, S, 1)
        w_h = predictions[..., 23:25] / S # (batch_size, S, S, 2)
        converted_preds = torch.cat((predicted_class, confidence_score, x, y, w_h), dim=-1) # (batch_size, S, S, 6)
        converted_preds = converted_preds.reshape(batch_size, S*S, 6)

    elif format == 'pred':
        # box1 변환
        confidence_score = predictions[..., 20:21] # (batch_size, S, S, 1)
        x = (predictions[..., 21:22] + cell_indices) / S # (batch_size, S, S, 1)
        y = (predictions[..., 22:23] + cell_indices.permute(0, 2, 1, 3)) / S # (batch_size, S, S, 1)
        w_h = predictions[..., 23:25] / S # (batch_size, S, S, 2)
        converted_pred1 = torch.cat((predicted_class, confidence_score, x, y, w_h), dim=-1) # (batch_size, S, S, 6)
        converted_pred1 = converted_pred1.reshape(batch_size, S*S, 6)
        
        # box2 변환
        confidence_score = predictions[..., 25:26]
        x = (predictions[..., 26:27] + cell_indices) / S
        y = (predictions[..., 27:28] + cell_indices.permute(0, 2, 1, 3)) / S
        w_h = predictions[..., 28:30] / S
        converted_pred2 = torch.cat((predicted_class, confidence_score, x, y, w_h), dim=-1)
        converted_pred2 = converted_pred1.reshape(batch_size, S*S, 6)

        converted_preds = torch.cat((converted_pred1, converted_pred2), dim=1) # (batch_size, 2*S*S, 6)
    
    return converted_preds

def get_bboxes(train_idx, targets, predictions, S, iou_threshold, prob_threshold):
    '''
    parameters:
        train_idx(long tensor): NMS 수행 시 같은 이미지끼리 비교를 위해 이미지에 index를 적용
        targets(tensor) & predictions(tensor): (batch_size, S, S, 6), [class_prediction, prob_score, x, y, w, h]
        
    returns:
        batch_pred_bboxes(list): (#, 7), [train_idx, class_prediction, prob_score, x, y, w, h]
        batch_true_bboxes(list): (#, 7), [train_idx, class_prediction, prob_score, x, y, w, h]
        train_idx(long tensor): 외부에서 index를 받아오기 위해 return
    '''

    targets = targets.clone().detach().cpu()
    predictions = predictions.clone().detach().cpu()

    train_idx = torch.Tensor([train_idx]).long()
    batch_size = predictions.shape[0]
    
    # 좌표 형식 변환
    true_bboxes = cell_to_xywh(targets, S, format='target') # (batch_size, S*S, 6)
    pred_bboxes = cell_to_xywh(predictions, S, format='pred') # (batch_size, 2*S*S, 6)

    batch_pred_bboxes, batch_true_bboxes = [], []
    for idx in range(batch_size):
        nms_bboxes = NMS(pred_bboxes[idx], iou_threshold, prob_threshold) # (#, 6)

        # pred_bboxes과 true_bboxes에 이미지 번호 추가
        for nms_bbox in nms_bboxes:
            batch_pred_bboxes.append(torch.cat((train_idx, nms_bbox), dim=-1)) # (#, 7)

        for true_bbox in true_bboxes[idx]:
            # 실제 객체가 있는 그리드만 confidence score가 1이기 때문에 해당 박스만 추림
            if true_bbox[1] == 1:
                batch_true_bboxes.append(torch.cat((train_idx, true_bbox), dim=-1)) # (#, 7)
        
        train_idx += 1

    return batch_pred_bboxes, batch_true_bboxes, train_idx

def get_iou_matrix(bboxes1, bboxes2):
    '''
    bbox 간 iou 행렬을 생성하는 함수
    https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/

    parameters:
        bboxes1(tensor): (num_bbox1, 4), normalized [x_center, y_center, width, height]
        bboxes2(tensor): (num_bbox2, 4), normalized [x_center, y_center, width, height]

    returns:
        intersection_over_union(tensor): (num_bbox1, num_bbox2, 1)
    '''
    num_bbox1 = bboxes1.shape[0]
    num_bbox2 = bboxes2.shape[0]

    box1 = torch.zeros_like(bboxes1) # (num_bbox1, 4)
    box2 = torch.zeros_like(bboxes2) # (num_bbox2, 4)

    # normalized [x_center, y_center, width, height] -> normalized [x_min, y_min, x_max, y_max]
    box1[..., 0:1] = bboxes1[..., 0:1] - bboxes1[..., 2:3] / 2 # (num_bbox1, 1)
    box1[..., 1:2] = bboxes1[..., 1:2] - bboxes1[..., 3:4] / 2
    box1[..., 2:3] = bboxes1[..., 0:1] + bboxes1[..., 2:3] / 2
    box1[..., 3:4] = bboxes1[..., 1:2] + bboxes1[..., 3:4] / 2
    
    box2[..., 0:1] = bboxes2[..., 0:1] - bboxes2[..., 2:3] / 2 # (num_bbox2, 1)
    box2[..., 1:2] = bboxes2[..., 1:2] - bboxes2[..., 3:4] / 2
    box2[..., 2:3] = bboxes2[..., 0:1] + bboxes2[..., 2:3] / 2
    box2[..., 3:4] = bboxes2[..., 1:2] + bboxes2[..., 3:4] / 2

    # 자기 상관 행렬 생성을 위해 순서가 바뀐 박스를 생성
    box1 = box1.unsqueeze(1).repeat(1, num_bbox2, 1) # (num_bbox1, num_bbox2, 4)
    box2 = box2.unsqueeze(0).repeat(num_bbox1, 1, 1) # (num_bbox1, num_bbox2, 4)

    box1_area = abs((box1[..., 2:3] - box1[..., 0:1]) * (box1[..., 3:4] - box1[..., 1:2])) # (num_bbox1, num_bbox2, 1)
    box2_area = abs((box2[..., 2:3] - box2[..., 0:1]) * (box2[..., 3:4] - box2[..., 1:2]))

    x1 = torch.max(box1[..., 0:1], box2[..., 0:1]) # (num_bbox1, num_bbox2, 1)
    y1 = torch.max(box1[..., 1:2], box2[..., 1:2])
    x2 = torch.min(box1[..., 2:3], box2[..., 2:3])
    y2 = torch.min(box1[..., 3:4], box2[..., 3:4])

    # clamp(0) -> bbox가 겹치지 않는 경우 최소값 0 설정
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) # (num_bbox1, num_bbox2, 1)

    intersection_over_union = intersection / (box1_area + box2_area - intersection + 1e-6) # (num_bbox1, num_bbox2, 1)

    return intersection_over_union

def get_class_matrix(class1, class2):
    '''
    클래스 간 일치하는 좌표 행렬을 생성하는 함수

    parameters:
        class1(tensor): (num_class1, 1)
        class2(tensor): (num_class2, 1)

    returns:
        classes(tensor): (num_class1, num_class2, 1)
    '''

    num_class1 = class1.shape[0]
    num_class2 = class2.shape[0]

    class1 = class1.unsqueeze(1).repeat(1, num_class2, 1) # (num_class1, num_class2, 1)
    class2 = class2.unsqueeze(0).repeat(num_class1, 1, 1) # (num_class1, num_class2, 1)

    classes = (class1 == class2) # (num_class1, num_class2, 1)

    return classes

def analyze_error(pred_bboxes, true_bboxes, img_paths, save_csv_path):
    '''
    parameters:
        pred_bboxes(list) & true_bboxes(list):
            tensor로 구성된 list 값
            (#, 7), [train_idx, class_prediction, prob_score, x, y, w, h]
    '''

    # error info csv column 생성
    column = [
        'img_path', 'train_idx',
        'gt_class', 'gt_conf', 'gt_x', 'gt_y', 'gt_w', 'gt_h',
        'dt_class', 'dt_conf', 'dt_x', 'dt_y', 'dt_w', 'dt_h',
        'iou',
        'error_type'
    ]
    with open(save_csv_path, 'a') as f:
        for idx, col_name in enumerate(column):
            if idx == (len(column) - 1):
                f.write(f'{col_name}')
            else:
                f.write(f'{col_name},')
        f.write('\n')

    iou_threshold = 0.5
    background_threshold = 0.1

    # 연산 속도 향상을 위해 list type에서 tensor type으로 변경
    tensor_pred_bboxes = torch.zeros(len(pred_bboxes), 7)
    tensor_true_bboxes = torch.zeros(len(true_bboxes), 7)
    for i, bbox in enumerate(pred_bboxes):
        tensor_pred_bboxes[i] = bbox
    for i, bbox in enumerate(true_bboxes):
        tensor_true_bboxes[i] = bbox

    num_imgs = int(max(tensor_true_bboxes[:, 0]))

    for train_idx in range(0, num_imgs+1):
        detections = tensor_pred_bboxes[tensor_pred_bboxes[:, 0] == train_idx]
        ground_truths = tensor_true_bboxes[tensor_true_bboxes[:, 0] == train_idx]
        img_path = img_paths[train_idx]
        
        # iou 행렬 계산
        ious = get_iou_matrix(ground_truths, detections) # (num_gt, num_det, 1)

        # class 일치 여부 확인용 행렬 계산
        det_class = detections[:, 1:2] # (num_det, 1)
        gt_class = ground_truths[:, 1:2] # (num_gt, 1)
        classes = get_class_matrix(gt_class, det_class) # (num_gt, num_det, 1)

        # error 분류
        # 여러 분류에 한번도 포함되지 않은 ground truths, detections를 나타내기 위한 인덱스
        gt_keep = torch.ones(len(ground_truths), dtype=bool)
        dt_keep = torch.ones(1, len(detections), dtype=bool)
        for error_type in ['correct', 'localization', 'other']:
            if error_type == 'correct':
                index = (ious > iou_threshold) & classes & dt_keep
            elif error_type == 'localization':
                index = (background_threshold < ious) & (ious <= iou_threshold) & classes & dt_keep
            # elif error_type == 'similar': # voc detection에서는 유사 클래스 없으므로 미구현
            #     pass
            elif error_type == 'other':
                index = (ious > background_threshold) & (~classes) & dt_keep
                
            index = torch.where(index == True) # 위의 조건을 만족하는 (gt_index, det_index)를 반환

            error_info = []
            for i, j in zip(index[0], index[1]): # (gt_index, det_index)
                gt_keep[i] = False
                dt_keep[0, j] = False
                temp = []
                temp.extend([img_path]) # [img_path]
                temp.extend(ground_truths[i].tolist()) # [train_idx, class_label, confidence, x, y, w, h]
                temp.extend(detections[j, 1:].tolist()) # [class_label, confidence, x, y, w, h]
                temp.extend(ious[i, j].tolist()) # [iou]
                temp.extend([error_type]) # [error_type]
                error_info.append(temp)

            # error info 기록
            with open(save_csv_path, 'a') as f:
                for info in error_info:
                    for idx, val in enumerate(info):
                        if idx == (len(info) - 1):
                            f.write(f'{val}')
                        else:
                            f.write(f'{val},')
                    f.write('\n')

        # background
        index = (ious <= background_threshold) & dt_keep
        index = torch.where(index == True) # (gt_index, det_index)
        
        error_info = []
        for i, j in zip(index[0], index[1]): # (gt_index, det_index)
            temp = []
            temp.extend([img_path]) # [img_path]
            temp.extend(detections[j].tolist()) # [train_idx, class_label, confidence, x, y, w, h]
            temp.extend(detections[j, 1:].tolist()) # [class_label, confidence, x, y, w, h]
            temp.extend(ious[i, j].tolist()) # [iou]
            temp.extend(['background']) # [error_type]
            error_info.append(temp)

        # error info 기록
        with open(save_csv_path, 'a') as f:
            for info in error_info:
                for idx, val in enumerate(info):
                    if idx == (len(info) - 1):
                        f.write(f'{val}')
                    else:
                        f.write(f'{val},')
                f.write('\n')

        # not detected
        not_detected_gts = ground_truths[gt_keep]
        error_info = []
        for nd_gt in not_detected_gts:
            temp = []
            temp.extend([img_path]) # [img_path]
            temp.extend(nd_gt.tolist()) # [train_idx, class_label, confidence, x, y, w, h]
            temp.extend(nd_gt[1:].tolist()) # [class_label, confidence, x, y, w, h], 그냥 column, row 맞추려고 추가
            temp.extend([0]) # [iou]
            temp.extend(['not detected']) # [error_type]
            error_info.append(temp)

        with open(save_csv_path, 'a') as f:
            for info in error_info:
                for idx, val in enumerate(info):
                    if idx == (len(info) - 1):
                        f.write(f'{val}')
                    else:
                        f.write(f'{val},')
                f.write('\n')

def draw(img, bboxes, class_names, colors, mean, std, analysis=False):
    '''
    parameters:
        img (tensor): [C, H, W]
        bboxes (tensor): [class_prediction, prob_score, x, y, w, h]
        class_names (list): 데이터셋의 클래스명
        colors (list): 색상 리스트
    '''

    img_size = [img.shape[1], img.shape[2]] # height, width

    img = inverse_normalize(img, mean, std)
    img = np.array(img, dtype=np.uint8)
    img = np.transpose(img, (1, 2, 0)) # [C, H, W] -> [H, W, C]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for bbox in bboxes:
        class_label = int(bbox[0])
        prob = bbox[1]
        coord = bbox[2:6]

        x1, y1, x2, y2 = convert_box_format(img_size, coord)
        color = colors[class_label]

        if analysis:
            error_type = bbox[6]
            text = class_names[class_label] + f" | {prob:0.2f} | {error_type}"
            text_len = 12*len(text)
        else:
            text = class_names[class_label] + f" | {prob:0.2f}"
            text_len = 9*len(text)
            
        if y1 <= 16:
            rect_coord = [[x1, y1], [x1+text_len, y1+13]]
            text_coord = [x1, y1+11]
        else:
            rect_coord = [[x1, y1-16], [x1+text_len, y1]]
            text_coord = [x1, y1-3]

        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness=2)
        cv2.rectangle(img, rect_coord[0], rect_coord[1], color, thickness=-1)
        cv2.putText(
            img, text=text, org=text_coord, fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255)
        )

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inverse_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    parameters:
        img (tensor): [C, H, W]
    '''
    
    img[0] = ((img[0]) * std[0]) + mean[0]
    img[1] = ((img[1]) * std[1]) + mean[1]
    img[2] = ((img[2]) * std[2]) + mean[2]
    img = 255 * img

    return img

def set_color(num_classes):
    import random

    colors = []
    for i in range(num_classes):
        b = random.randint(0, 100) # 100 이상의 밝은 색은 눈부심
        g = random.randint(0, 100)
        r = random.randint(0, 100)
        colors.append([b, g, r])

    return colors

def convert_box_format(img_size, coordinate):
    # normalized [x_center, y_center, width, height] -> [x_min, y_min, x_max, y_max]
    img_height, img_width = img_size

    xmin = int(img_width * (2 * coordinate[0] - coordinate[2]) / 2)
    ymin = int(img_height * (2 * coordinate[1] - coordinate[3]) / 2)
    xmax = int(img_width * (2 * coordinate[0] + coordinate[2]) / 2)
    ymax = int(img_height * (2 * coordinate[1] + coordinate[3]) / 2)
    
    coord = [xmin, ymin, xmax, ymax]

    return coord