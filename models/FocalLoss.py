import paddle
import paddle.nn as nn

class FocalLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, alpha=[1.]*2, gamma=2.0, ignore_index=255,smooth_rate=0,reduction='mean',online_reweighting=False):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.EPS = 1e-8
        if isinstance(alpha,list):
            self.alpha=paddle.to_tensor(alpha)
        self.alpha=paddle.reshape(self.alpha,(1,1,1,2))
        self.gamma=gamma
        self.smooth_rate=smooth_rate
        self.reduction=reduction

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """

        label_onehot=nn.functional.one_hot(label,2)
        k=15
        p=paddle.nn.functional.softmax(logit)*(1-2*self.EPS)+self.EPS
        loss=-paddle.pow(1-p,self.gamma)*paddle.log(p)
        loss=paddle.max(loss*label_onehot,axis=-1)
        if self.reduction=='mean':
            return paddle.mean(loss)
        elif self.reduction=='none':
            return loss