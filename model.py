import mindspore.ops as ops
import mindspore.nn as nn


class CustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer, sens=1.0):
        """入参有三个：训练网络，优化器和反向传播缩放比例"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                    # 定义前向网络
        self.network.set_grad()                   # 构建反向网络
        self.optimizer = optimizer                # 定义优化器
        self.weights = self.optimizer.parameters  # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True, sens_param=False)  # 反向传播获取梯度

    def construct(self, *inputs):
        loss = self.network(*inputs)                    # 执行前向网络，计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        loss = ops.depend(loss, self.optimizer(grads))  # 使用优化器更新梯度
        return loss
