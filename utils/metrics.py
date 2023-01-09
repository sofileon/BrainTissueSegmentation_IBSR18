import numpy as np
from medpy.metric.binary import dc, hd

def metrics_dice_hd(prediction, groundtruth):
    """
    Dice score for brain volumens to the labels csf, wm and gm.
    :param prediction: segmentation predicted
    :param groundtruth: ground truth segmentation
    :return: dice: list of the three dice scores, csf wm and gm
    """
    groundtruth = np.uint8(groundtruth)
    wm_data_gt = groundtruth.copy()
    gm_data_gt = groundtruth.copy()
    csf_data_gt = groundtruth.copy()

    # In the labels: 3->GRAY MATTER; 2-> WHITE MATTER; 1->CSF
    wm_data_gt[wm_data_gt != 3] = 0
    gm_data_gt[gm_data_gt != 2] = 0
    csf_data_gt[csf_data_gt != 1] = 0

    wm_data_PRED = prediction.copy()
    gm_data_PRED = prediction.copy()
    csf_data_PRED = prediction.copy()

    wm_data_PRED[prediction != 3] = 0
    gm_data_PRED[prediction != 2] = 0
    csf_data_PRED[prediction != 1] = 0

    dice_wm_PRED = dc(wm_data_PRED, wm_data_gt)
    dice_gm_PRED = dc(gm_data_PRED, gm_data_gt)
    dice_csf_PRED = dc(csf_data_PRED, csf_data_gt)

    hd_wm_PRED = hd(wm_data_PRED, wm_data_gt)
    hd_gm_PRED = hd(gm_data_PRED, gm_data_gt)
    hd_csf_PRED = hd(csf_data_PRED, csf_data_gt)

    dice = [dice_wm_PRED, dice_gm_PRED, dice_csf_PRED]
    hausdorff_distance = [hd_wm_PRED, hd_gm_PRED, hd_csf_PRED]

    return dice, hausdorff_distance
