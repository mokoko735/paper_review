import torch
import torch.nn as nn
from utils import IoU


class YoloLoss1(nn.Module):
    '''
    S: split size of image
    B: number of boxes
    C: number of classes

    targets: 
        targets shape [batch_size, S, S, C+5]
        C+5 -> [classes(0:20), confidence(20), box_coord(21:25)]
    predictions:
        predictions shape [batch_size, S, S, C+(B*5)]
        C+(B*5) -> [classes(0:20), box1_confidence(20), box1_coord(21:25), box2_confidence(25), box2_coord(26:30)]

    loss 작성 시 참고 사항
        1. coord_loss
            2개의 bbox 중 ground_truth와 높은 iou 값을 갖는 bbox의 좌표에 대한 loss를 계산
            모델의 최종 출력이 leaky relu이기 때문에 루트 안에 음수가 들어가는 것에 대한 방지책 사용
        2. confidence_loss
            object_loss 계산 시에는 coord_loss처럼 iou가 높은 bbox에 대해 loss 계산
            no_object_loss 계산 시에는 두 bbox 모두에 대해 loss 계산
        3. class_loss
            객체가 존재하는 cell에 대해서만 loss 계산
    '''

    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum") # sum squared error
        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        # 수식의 Iobj는 bbox 2개 중 ground truth와 가장 높은 iou 값을 가지는 박스를 선택하는 마스크
        # 추론 결과의 98개 박스 중 loss에 적용할 49개의 박스를 추리는 과정
        iou_b1 = IoU(predictions[..., 21:25], targets[..., 21:25]) # iou shape [bs, 7, 7, 1]
        iou_b2 = IoU(predictions[..., 26:30], targets[..., 21:25]) # iou shape [bs, 7, 7, 1]
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # ious shape [2, bs, 7, 7, 1]
        iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox shape [bs, 7, 7, 1], box1의 인덱스 0, box2의 인덱스 1
        
        # 객체가 존재하는 cell과 아닌 cell을 구분하는 마스크
        obj = targets[..., 20].unsqueeze(3) # obj shape [4, 7, 7]
        no_obj = 1 - obj

        # coordinate loss ------------------------------------------------------------------------------------
        predictions_coord = bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
        targets_coord = targets[..., 21:25]

        # x, y, w, h 중 w, h에는 루트를 적용
        # 최종 출력이 leaky relu이기 때문에 sqrt에 음수가 들어가는 것을 방지
        predictions_coord[..., 2:4] = torch.sign(predictions_coord[..., 2:4]) * torch.sqrt(
            torch.abs(predictions_coord[..., 2:4] + 1e-6))
        targets_coord[..., 2:4] = torch.sqrt(targets_coord[..., 2:4])

        coord_loss = self.mse(predictions_coord, targets_coord)

        coord_loss = self.lambda_coord * coord_loss
    
        # confidence loss ------------------------------------------------------------------------------------
        # object loss
        predictions_conf = obj * (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])
        targets_conf = obj * targets[..., 20:21]

        object_loss = self.mse(predictions_conf, targets_conf)
        
        # no object loss
        predictions_conf1 = no_obj * predictions[..., 20:21]
        predictions_conf2 = no_obj * predictions[..., 25:26]
        targets_conf = no_obj * targets[..., 20:21] # 모든 텐서가 0이 되어야 함
        
        no_object_loss = self.mse(predictions_conf1, targets_conf)
        no_object_loss += self.mse(predictions_conf2, targets_conf)

        conf_loss = object_loss + self.lambda_noobj * no_object_loss

        # class loss -----------------------------------------------------------------------------------------
        predictions_class = obj * predictions[..., :20]
        targets_class = obj * targets[..., :20]

        class_loss = self.mse(predictions_class, targets_class)

        # total loss -----------------------------------------------------------------------------------------
        loss = coord_loss + conf_loss + class_loss

        return loss, coord_loss, conf_loss, class_loss
    

# -----------------------------------------------------------------------------------------------------------
class YoloLoss2(nn.Module):
    '''
    코드 원작자의 loss 함수
    '''

    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum") # sum squared error
        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        iou_b1 = IoU(predictions[..., 21:25], targets[..., 21:25]) # iou shape [bs, 7, 7, 1]
        iou_b2 = IoU(predictions[..., 26:30], targets[..., 21:25]) # iou shape [bs, 7, 7, 1]
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # ious shape [2, bs, 7, 7, 1]
        iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox shape [bs, 7, 7, 1], box1의 인덱스 0, box2의 인덱스 1

        obj = targets[..., 20].unsqueeze(3) # exists_box shape [4, 7, 7]
        no_obj = 1 - obj

        # coordinate loss ------------------------------------------------------------------------------------
        predictions_coord = obj * (bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25])
        targets_coord = obj * targets[..., 21:25]

        predictions_coord[..., 2:4] = torch.sign(predictions_coord[..., 2:4]) * torch.sqrt(
            torch.abs(predictions_coord[..., 2:4] + 1e-6))
        targets_coord[..., 2:4] = torch.sqrt(targets_coord[..., 2:4])

        coord_loss = self.mse(predictions_coord, targets_coord)

        coord_loss = self.lambda_coord * coord_loss

        # confidence loss ------------------------------------------------------------------------------------
        # object loss
        predictions_conf = obj * (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])
        targets_conf = obj * targets[..., 20:21]

        object_loss = self.mse(predictions_conf, targets_conf)
        
        # no object loss
        predictions_conf1 = no_obj * predictions[..., 20:21]
        predictions_conf2 = no_obj * predictions[..., 25:26]
        targets_conf = no_obj * targets[..., 20:21] # 모든 텐서가 0이 되어야 함
        
        no_object_loss = self.mse(predictions_conf1, targets_conf)
        no_object_loss += self.mse(predictions_conf2, targets_conf)

        conf_loss = object_loss + self.lambda_noobj * no_object_loss

        # class loss -----------------------------------------------------------------------------------------
        predictions_class = obj * predictions[..., :20]
        targets_class = obj * targets[..., :20]

        class_loss = self.mse(predictions_class, targets_class)

        # total loss -----------------------------------------------------------------------------------------
        loss = coord_loss + conf_loss + class_loss

        return loss, coord_loss, conf_loss, class_loss

    

# 학습에 실패한 loss들 -----------------------------------------------------------------------------------------

# class YoloLoss2(nn.Module):
#     '''
#     기존 loss 수식을 수정한 버전
#     1. coord_loss
#         - VOCdetection 데이터셋에서 49개 그리드 중 실제 객체가 존재하는 그리드는 몇개 없기 때문에
#           기존 loss를 사용할 경우 평균적으로 40개 이상의 그리드는 좌표값 0을 학습하게 되며
#           모델이 박스의 좌표가 아닌 0을 출력하도록 학습하게 된다.
#         - 따라서 객체가 있는 그리드를 구분하기 위한 마스크 obj, no_obj를 적용하여 객체가 있는 그리드에
#           강한 가중치를 부여하여 박스 좌표가 0으로 수렴하지 않도록 하는 것이 목표이다.
#         - 추가로 iou를 사용하여 98개 박스 중 49개 박스를 추려서 학습하는 기존 방식에서 박스1은 전부 0
#           박스2는 정상 좌표를 출력하는 문제가 발견되었는데 49개 박스를 추리는 과정에서 하나의 박스1이
#           지속적으로 버려지는 것으로 판단하여 98개 박스 전부 학습한다.

#     2. conf_loss, class_loss
#         - coord_loss가 loss의 대부분을 차지하여 학습이 되지 않는다고 판단하여
#           객체가 존재하는 그리드에 obj, no_obj 마스크를 적용하여 가중치를 차등 적용한다.
#     '''

#     def __init__(self, S=7, B=2, C=20):
#         super(YoloLoss2, self).__init__()
#         self.mse = nn.MSELoss(reduction="sum") # sum squared error
#         self.S = S
#         self.B = B
#         self.C = C

#         self.lambda_obj_coord = 10
#         self.lambda_no_obj_coord = 1

#         self.lambda_obj_conf = 5
#         self.lambda_no_obj_conf = 2

#         self.lambda_obj_class = 5
#         self.lambda_no_obj_class = 1

#     def forward(self, predictions, targets):
#         obj = targets[..., 20].unsqueeze(3) # exists_box shape [4, 7, 7, 1]
#         no_obj = 1 - obj

#         # coordinate loss ------------------------------------------------------------------------------------
#         predictions_coord1 = predictions[..., 21:25]
#         predictions_coord2 = predictions[..., 26:30]
#         targets_coord = targets[..., 21:25]

#         predictions_coord1[..., 2:4] = torch.sign(predictions_coord1[..., 2:4]) * torch.sqrt(
#             torch.abs(predictions_coord1[..., 2:4] + 1e-6))
#         predictions_coord2[..., 2:4] = torch.sign(predictions_coord2[..., 2:4]) * torch.sqrt(
#             torch.abs(predictions_coord2[..., 2:4] + 1e-6))
#         targets_coord[..., 2:4] = torch.sqrt(targets_coord[..., 2:4])

#         # object loss
#         obj_coord_loss = self.mse(obj * predictions_coord1, obj * targets_coord)
#         obj_coord_loss += self.mse(obj * predictions_coord2, obj * targets_coord)

#         # no object loss
#         no_obj_coord_loss = self.mse(no_obj * predictions_coord1, torch.zeros_like(predictions_coord1))
#         no_obj_coord_loss += self.mse(no_obj * predictions_coord2, torch.zeros_like(predictions_coord2))

#         coord_loss = (
#             self.lambda_obj_coord * obj_coord_loss +
#             self.lambda_no_obj_coord * no_obj_coord_loss
#         )

#         # confidence loss ------------------------------------------------------------------------------------
#         predictions_conf1 = predictions[..., 20:21]
#         predictions_conf2 = predictions[..., 25:26]
#         targets_conf = targets[..., 20:21]

#         # object loss        
#         obj_conf_loss = self.mse(obj * predictions_conf1, obj * targets_conf)
#         obj_conf_loss += self.mse(obj * predictions_conf2, obj * targets_conf)
        
#         # no object loss
#         no_obj_conf_loss = self.mse(no_obj * predictions_conf1, torch.zeros_like(predictions_conf1))
#         no_obj_conf_loss += self.mse(no_obj * predictions_conf2, torch.zeros_like(predictions_conf2))

#         conf_loss = (
#             self.lambda_obj_conf * obj_conf_loss +
#             self.lambda_no_obj_conf * no_obj_conf_loss
#         )

#         # class loss -----------------------------------------------------------------------------------------
#         predictions_class = predictions[..., :20]
#         targets_class = targets[..., :20]

#         obj_class_loss = self.mse(obj * predictions_class, obj * targets_class)

#         no_obj_class_loss = self.mse(no_obj * predictions_class, torch.zeros_like(predictions_class))

#         class_loss = (
#             self.lambda_obj_class * obj_class_loss +
#             self.lambda_no_obj_class * no_obj_class_loss
#         )

#         # total loss -----------------------------------------------------------------------------------------
#         loss = coord_loss + conf_loss + class_loss

#         return loss, coord_loss, conf_loss, class_loss
    

# # -----------------------------------------------------------------------------------------------------------
# class YoloLoss3(nn.Module):
#     '''
#     코드 원작자의 loss 함수
#     '''

#     def __init__(self, S=7, B=2, C=20):
#         super(YoloLoss3, self).__init__()
#         self.mse = nn.MSELoss(reduction="sum") # sum squared error
#         self.S = S
#         self.B = B
#         self.C = C

#         self.lambda_noobj = 0.5
#         self.lambda_coord = 5

#     def forward(self, predictions, targets):
#         iou_b1 = IoU(predictions[..., 21:25], targets[..., 21:25]) # iou shape [bs, 7, 7, 1]
#         iou_b2 = IoU(predictions[..., 26:30], targets[..., 21:25]) # iou shape [bs, 7, 7, 1]
#         ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # ious shape [2, bs, 7, 7, 1]
#         iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox shape [bs, 7, 7, 1], box1의 인덱스 0, box2의 인덱스 1

#         obj = targets[..., 20].unsqueeze(3) # exists_box shape [4, 7, 7, 1]
#         no_obj = 1 - obj

#         # coordinate loss ------------------------------------------------------------------------------------
#         predictions_coord = obj * (
#             bestbox * predictions[..., 26:30] +
#             (1 - bestbox) * predictions[..., 21:25]
#         )
#         targets_coord = obj * targets[..., 21:25]

#         predictions_coord[..., 2:4] = torch.sign(predictions_coord[..., 2:4]) * torch.sqrt(
#             torch.abs(predictions_coord[..., 2:4] + 1e-6))
#         targets_coord[..., 2:4] = torch.sqrt(targets_coord[..., 2:4])

#         coord_loss = self.mse(predictions_coord, targets_coord)

#         coord_loss = self.lambda_coord * coord_loss

#         # confidence loss ------------------------------------------------------------------------------------
#         # object loss
#         predictions_conf = obj * (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])
#         targets_conf = obj * targets[..., 20:21]

#         object_loss = self.mse(predictions_conf, targets_conf)
        
#         # no object loss
#         predictions_conf1 = no_obj * predictions[..., 20:21]
#         predictions_conf2 = no_obj * predictions[..., 25:26]
#         targets_conf = no_obj * targets[..., 20:21] # 모든 텐서가 0이 되어야 함
        
#         no_object_loss = self.mse(predictions_conf1, targets_conf)
#         no_object_loss += self.mse(predictions_conf2, targets_conf)

#         conf_loss = object_loss + self.lambda_noobj * no_object_loss

#         # class loss -----------------------------------------------------------------------------------------
#         predictions_class = obj * predictions[..., :20]
#         targets_class = obj * targets[..., :20]

#         class_loss = self.mse(predictions_class, targets_class)

#         # total loss -----------------------------------------------------------------------------------------
#         loss = coord_loss + conf_loss + class_loss

#         return loss, coord_loss, conf_loss, class_loss
    

# # -----------------------------------------------------------------------------------------------------------
# class YoloLoss4(nn.Module):
#     '''
#     코드 원작자의 loss 함수에서 confidence와 class에 대한 가중치를 증가시킨 버전
#     '''

#     def __init__(self, S=7, B=2, C=20):
#         super(YoloLoss4, self).__init__()
#         self.mse = nn.MSELoss(reduction="sum") # sum squared error
#         self.S = S
#         self.B = B
#         self.C = C

#         self.lambda_noobj = 0.5
#         self.lambda_coord = 5

#         self.lambda_conf = 8
#         self.lambda_class = 8

#     def forward(self, predictions, targets):
#         iou_b1 = IoU(predictions[..., 21:25], targets[..., 21:25]) # iou shape [bs, 7, 7, 1]
#         iou_b2 = IoU(predictions[..., 26:30], targets[..., 21:25]) # iou shape [bs, 7, 7, 1]
#         ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # ious shape [2, bs, 7, 7, 1]
#         iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox shape [bs, 7, 7, 1], box1의 인덱스 0, box2의 인덱스 1

#         obj = targets[..., 20].unsqueeze(3) # exists_box shape [4, 7, 7, 1]
#         no_obj = 1 - obj

#         # coordinate loss ------------------------------------------------------------------------------------
#         predictions_coord = obj * (
#             bestbox * predictions[..., 26:30] +
#             (1 - bestbox) * predictions[..., 21:25]
#         )
#         targets_coord = obj * targets[..., 21:25]

#         predictions_coord[..., 2:4] = torch.sign(predictions_coord[..., 2:4]) * torch.sqrt(
#             torch.abs(predictions_coord[..., 2:4] + 1e-6))
#         targets_coord[..., 2:4] = torch.sqrt(targets_coord[..., 2:4])

#         coord_loss = self.mse(predictions_coord, targets_coord)

#         coord_loss = self.lambda_coord * coord_loss

#         # confidence loss ------------------------------------------------------------------------------------
#         # object loss
#         predictions_conf = obj * (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])
#         targets_conf = obj * targets[..., 20:21]

#         object_loss = self.mse(predictions_conf, targets_conf)
        
#         # no object loss
#         predictions_conf1 = no_obj * predictions[..., 20:21]
#         predictions_conf2 = no_obj * predictions[..., 25:26]
#         targets_conf = no_obj * targets[..., 20:21] # 모든 텐서가 0이 되어야 함
        
#         no_object_loss = self.mse(predictions_conf1, targets_conf)
#         no_object_loss += self.mse(predictions_conf2, targets_conf)

#         conf_loss = object_loss + self.lambda_noobj * no_object_loss
#         conf_loss = self.lambda_conf * conf_loss

#         # class loss -----------------------------------------------------------------------------------------
#         predictions_class = obj * predictions[..., :20]
#         targets_class = obj * targets[..., :20]

#         class_loss = self.mse(predictions_class, targets_class)
#         class_loss = self.lambda_class * class_loss

#         # total loss -----------------------------------------------------------------------------------------
#         loss = coord_loss + conf_loss + class_loss

#         return loss, coord_loss, conf_loss, class_loss
    

# class YoloLoss5(nn.Module):
#     '''
#     원본 논문 loss 수식을 수정한 버전
#     1. coord_loss
#         - VOCdetection 데이터셋에서 49개 그리드 중 실제 객체가 존재하는 그리드는 몇개 없기 때문에
#           기존 loss를 사용할 경우 평균적으로 40개 이상의 그리드는 좌표값 0을 학습하게 되며
#           모델이 박스의 좌표가 아닌 0을 출력하도록 학습하게 된다.
#         - 따라서 객체가 있는 그리드를 구분하기 위한 마스크 obj, no_obj를 적용하여 객체가 있는 그리드에
#           강한 가중치를 부여하여 박스 좌표가 0으로 수렴하지 않도록 하는 것이 목표이다.

#     2. conf_loss, class_loss
#         - coord_loss가 loss의 대부분을 차지하여 학습이 되지 않는다고 판단하여
#           객체가 존재하는 그리드에 obj, no_obj 마스크를 적용하여 가중치를 차등 적용한다.
#     '''
#     def __init__(self, S=7, B=2, C=20):
#         super(YoloLoss5, self).__init__()
#         self.mse = nn.MSELoss(reduction="sum") # sum squared error
#         self.S = S
#         self.B = B
#         self.C = C

#         # 객체 개수 차이가 40~42 / 5~3 정도로 8~14배 가량 차이가 나기 때문에 배율을 그에 맞게 조정
#         self.lambda_obj_coord = 12
#         self.lambda_no_obj_coord = 2

#         self.lambda_obj_conf = 4
#         self.lambda_no_obj_conf = 1

#         self.lambda_obj_class = 4
#         self.lambda_no_obj_class = 1


#     def forward(self, predictions, targets):
#         iou_b1 = IoU(predictions[..., 21:25], targets[..., 21:25]) # iou shape [bs, 7, 7, 1]
#         iou_b2 = IoU(predictions[..., 26:30], targets[..., 21:25]) # iou shape [bs, 7, 7, 1]
#         ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # ious shape [2, bs, 7, 7, 1]
#         iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox shape [bs, 7, 7, 1], box1의 인덱스 0, box2의 인덱스 1

#         obj = targets[..., 20].unsqueeze(3) # exists_box shape [4, 7, 7, 1]
#         no_obj = 1 - obj

#         # coordinate loss ------------------------------------------------------------------------------------
#         predictions_coord = bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
#         targets_coord = targets[..., 21:25]

#         predictions_coord[..., 2:4] = torch.sign(predictions_coord[..., 2:4]) * torch.sqrt(
#             torch.abs(predictions_coord[..., 2:4] + 1e-6))
#         targets_coord[..., 2:4] = torch.sqrt(targets_coord[..., 2:4])

#         # object loss
#         obj_coord_loss = self.mse(obj * predictions_coord, obj * targets_coord)

#         # no object loss
#         no_obj_coord_loss = self.mse(no_obj * predictions_coord, torch.zeros_like(predictions_coord))

#         coord_loss = (
#             self.lambda_obj_coord * obj_coord_loss +
#             self.lambda_no_obj_coord * no_obj_coord_loss
#         )

#         # confidence loss ------------------------------------------------------------------------------------
#         # object loss
#         predictions_conf = bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
#         predictions_conf1 = predictions[..., 20:21]
#         predictions_conf2 = predictions[..., 25:26]
#         targets_conf = targets[..., 20:21]

#         # object loss        
#         obj_conf_loss = self.mse(obj * predictions_conf, obj * targets_conf)
        
#         # no object loss
#         no_obj_conf_loss = self.mse(no_obj * predictions_conf1, torch.zeros_like(predictions_conf1))
#         no_obj_conf_loss += self.mse(no_obj * predictions_conf2, torch.zeros_like(predictions_conf2))

#         conf_loss = (
#             self.lambda_obj_conf * obj_conf_loss +
#             self.lambda_no_obj_conf * no_obj_conf_loss
#         )

#         # class loss -----------------------------------------------------------------------------------------
#         predictions_class = predictions[..., :20]
#         targets_class = targets[..., :20]

#         obj_class_loss = self.mse(obj * predictions_class, obj * targets_class)

#         no_obj_class_loss = self.mse(no_obj * predictions_class, torch.zeros_like(predictions_class))

#         class_loss = (
#             self.lambda_obj_class * obj_class_loss +
#             self.lambda_no_obj_class * no_obj_class_loss
#         )

#         # total loss -----------------------------------------------------------------------------------------
#         loss = coord_loss + conf_loss + class_loss

#         return loss, coord_loss, conf_loss, class_loss
    