import os
from detectron2.config import get_cfg
from detectron2 import model_zoo

def setup_config():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = r"E:\Random Python Scripts\FarmLand-Segmentation-main\Detectron 2\Models\model_50000_epochs (7.5k)"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # keep 2 for detectron
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    return cfg
