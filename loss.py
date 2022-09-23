import mindspore.nn as nn
import mindspore.ops as ops


class JointsMSELoss(nn.Cell):
    """Joint MSELoss"""
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def construct(self, output, target, target_weight):
        if len(target.shape) != 4 or len(target_weight.shape) != 3:
            raise KeyError('target.dim or target_weight.dim() get wrong value')

        criterion = nn.MSELoss()

        batch_size, num_joints, _, _ = output.shape

        split = ops.Split(1, num_joints)
        heatmaps_pred = split(output.reshape((batch_size, num_joints, -1)))
        heatmaps_gt = split(target.reshape((batch_size, num_joints, -1)))

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            loss += criterion(heatmap_pred * target_weight[:, idx],
                              heatmap_gt * target_weight[:, idx])

        return loss / num_joints


class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label1, label2):
        output = self._backbone(data, label1, label2)
        return self._loss_fn(output, label1, label2)
