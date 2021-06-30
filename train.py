import cv2
from paddle.nn import KLDivLoss
import paddle.nn.functional as F 
from tqdm import tqdm
import paddle
import argparse


from dataset import dataset
from models.portraitnet import PortraitNet
from models.FocalLoss import FocalLoss
from utils.metrics import ACC,MIOU,Loss
from val import val

def pharse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/Supervisely_face')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--edge', action='store_true')
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--checkpoint',type=str,default=None)
    return parser.parse_args()

def train(args):
    datas=dataset('%s/train_list.txt'%args.data_root,args.data_root)
    loader=paddle.io.DataLoader(datas,batch_size=args.batch_size,drop_last=True,shuffle=True)
    
    model=PortraitNet(edge=args.edge)
    if args.checkpoint is not None:
        model.set_state_dict(paddle.load(args.checkpoint))
    loss_func=paddle.nn.CrossEntropyLoss()
    if args.edge==True:
        edge_func=FocalLoss()
    if args.kl:
        kl_func=KLDivLoss()

    lr=args.lr
    optimizer=paddle.optimizer.AdamW(learning_rate=args.lr,parameters=model.parameters(),weight_decay=1e-6)
    lossm=Loss()
    acc=ACC()
    miou=MIOU()
    
    for epoch in range(args.epoch):
        pbar=tqdm(loader)
        pbar.set_description("epoch %d/%d:"%(epoch+1,args.epoch))
        lr=(1-(epoch/args.epoch)**0.95)*args.lr
        optimizer.set_lr(lr)
        model.train()
        
        for data in pbar:
            img,img_aug,mask,edge_mask=data
            edge_mask=paddle.unsqueeze(edge_mask,1)
            if args.kl is True:
                ouput_aug=model(img_aug)
            output=model(img)
            if args.edge==True:
                output,edge=output
                output=output.transpose([0,2,3,1])
                edge=edge.transpose([0,2,3,1])
                loss=loss_func(output,mask)+0.3*edge_func(edge,edge_mask)
            else:
                loss=loss_func(output,mask)
            if args.kl is True:
                loss=loss+loss_func(ouput_aug[0].transpose([0,2,3,1]),mask)+2*kl_func(F.log_softmax(ouput_aug[0].transpose([0,2,3,1])),F.softmax(output))
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            

            lossm(loss,mask)
            acc(output,mask)
            miou(output,mask)

            if (pbar.last_print_n+1)%args.log_iter==0:
                print('iter %d loss:%.5f, acc: %.5f, miou: %.5f '%(pbar.last_print_n,lossm.part_metric.new_step(),acc.part_metric.new_step(),miou.part_metric.new_step()))
            
        print('epoch %d loss:%.5f, acc: %.5f, miou: %.5f '%(epoch,lossm.new_step(),acc.new_step(),miou.new_step()))
        val(args,model)
        paddle.save(model.state_dict(),'output/model_%d.pdparams'%epoch)





if __name__ =='__main__':
    args=pharse_args()
    train(args)