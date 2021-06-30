import sys
import paddle
import numpy

class metric():
    def __init__(self):
        self.value=0.
        self.count=0
        self.history_value=[]
        self.part_metric=None

    def __call__(self,logits,mask):
        if self.part_metric is not None:
            value=self.part_metric(logits,mask)
        else:
            logits,mask=logits.numpy(),mask.numpy()
            #acc=numpy.sum(logits.argmax(1)==mask)/mask.size
            value=self.calc_func(logits,mask)
        self.value=(self.value+numpy.sum(value,-1))
        self.count+=value.shape[-1]
        return value

    def calc_func(self,logits,mask):
        return numpy.average(mask)
    
    def new_step(self):
        self.history_value.append(numpy.average(self.value[0]/self.value[1]))
        self.value=0
        self.count=0
        return self.history_value[-1]

class ACC(metric):
    def __init__(self,part=True):
        super().__init__()
        if part:
            self.part_metric=ACC(False)
    
    def calc_func(self,logits,mask):
        value=numpy.ones((2,logits.shape[0]))
        value[0]=numpy.sum(logits.argmax(-1)==mask,(1,2))
        value[1,:]=mask[0].size
        return value


class MIOU(metric):
    def __init__(self,num_classes=2, part=True):
        super().__init__()
        self.num_classes=num_classes
        if part:
            self.part_metric=MIOU(num_classes,False)

    def calc_func(self, logits, mask):
        pred=logits.argmax(-1)
        iou=numpy.zeros((2,self.num_classes,logits.shape[0]))
        for i in range(self.num_classes):
            pred_i=pred==i
            mask_i=mask==i
            iou[0,i],iou[1,i]=numpy.sum(numpy.logical_and(pred_i,mask_i),(1,2)),numpy.sum(numpy.logical_or(pred_i,mask_i),(1,2))
        return iou

class Loss(metric):
    def __init__(self,part=True):
        super().__init__()
        if part:
            self.part_metric=Loss(False)
    
    def calc_func(self, loss, mask=paddle.ones((1,))):
        return loss

    def new_step(self):
        self.history_value.append(numpy.average(self.value/self.count))
        self.value=0
        self.count=0
        return self.history_value[-1]