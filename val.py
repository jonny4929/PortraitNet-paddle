import paddle
import argparse
from tqdm import tqdm

from dataset import dataset
from models.portraitnet import PortraitNet
from utils.metrics import ACC,MIOU

def pharse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/Supervisely_face')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_iter', type=int, default=5)
    parser.add_argument('--model_path', type=str)
    return parser.parse_args()


def val(args,model=None):
    datas=dataset('%s/val_list.txt'%args.data_root,args.data_root,aug=False)
    loader=paddle.io.DataLoader(datas,batch_size=args.batch_size,drop_last=False,shuffle=False)
    
    if model is None:
        model=PortraitNet(edge=True)
        model.set_state_dict(paddle.load(args.model_path))
    model.eval()

    acc=ACC()
    miou=MIOU()

    pbar=tqdm(loader)
    pbar.set_description('eval')
    for data in pbar:
        img,img_aug,mask,edge_mask=data
        output,_=model(img)
        output=output.transpose([0,2,3,1])
        acc(output,mask)
        miou(output,mask)

        if (pbar.last_print_n+1)%args.log_iter==0:
            print('iter %d  acc: %.5f, miou: %.5f '%(pbar.last_print_n,acc.part_metric.new_step(),miou.part_metric.new_step()))
            
    print('eval: acc: %.5f, miou: %.5f '%(acc.new_step(),miou.new_step()))
    return


    

if __name__=='__main__':
    args=pharse_args()
    val(args)