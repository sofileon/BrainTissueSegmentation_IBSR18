import numpy as np
from medpy.metric.binary import dc, hd, ravd


def metrics_4_segmentation(prediction, groundtruth):
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

    wm_data_pred = prediction.copy()
    gm_data_pred = prediction.copy()
    csf_data_pred = prediction.copy()

    wm_data_pred[prediction != 3] = 0
    gm_data_pred[prediction != 2] = 0
    csf_data_pred[prediction != 1] = 0

    dice_wm_pred = dc(wm_data_pred, wm_data_gt)
    dice_gm_pred = dc(gm_data_pred, gm_data_gt)
    dice_csf_pred = dc(csf_data_pred, csf_data_gt)

    hd_wm_pred = hd(wm_data_pred, wm_data_gt)
    hd_gm_pred = hd(gm_data_pred, gm_data_gt)
    hd_csf_pred = hd(csf_data_pred, csf_data_gt)

    ravd_wm_pred = ravd(wm_data_pred, wm_data_gt)
    ravd_gm_pred = ravd(gm_data_pred, gm_data_gt)
    ravd_csf_pred = ravd(csf_data_pred, csf_data_gt)

    dice = [dice_wm_pred, dice_gm_pred, dice_csf_pred]
    hausdorff_distance = [hd_wm_pred, hd_gm_pred, hd_csf_pred]
    volume_difference = [ravd_wm_pred, ravd_gm_pred, ravd_csf_pred]

    return dice, hausdorff_distance, volume_difference
