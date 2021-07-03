# PortraitNet-paddle[论文复现]

PortraitNet论文复现 [aistudio项目地址](https://aistudio.baidu.com/aistudio/projectdetail/1949642)

[论文地址](https://www.yongliangyang.net/docs/mobilePotrait_c&g19.pdf)  [官方github项目地址](https://github.com/dong-x16/PortraitNet)

## 数据准备

原数据需要进行一定的筛选和处理才能使用，已经处理好的版本发布在[aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/89377)，需要的话可以下载并解压缩在任意目录即可

目录结构如下

    |--EG1800 or Supervisely_face
        |--img
            |--xxx.jpg
        |--mask
            |--xxx.jpg
        |--train_list.txt
        |--val_list.txt

如果需要重写dataset，可以使用transform类中的数据增强方式，具体介绍参见[aistudio项目](https://aistudio.baidu.com/aistudio/projectdetail/1949642)

## 训练模型
调用训练脚本可以使用如下命令

    python train.py --edge --kl --epoch 10 --lr 0.001 --batch_size 32 --data_root [你的数据集路径] (例如:data/Supervisely_face/)
    --edge表示采用边缘分支
    --kl表示采用kl散度损失

## 验证模型
调用验证模型可以采用如下命令

    python val.py --model_path model_zoo/supervise_face.pdparams --batch_size 32 --log_iter 10 --data_root [你的数据集路径] (例如:data/Supervisely_face/)

## 模型库
|模型|数据集|MIOU|下载地址|
|---|---|---|---|
|PortraitNet-MobileNetV2|EG1800|95.12%|[aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/89377)|
|PortraitNet-MobileNetV2|Supervisely_face|93.83%|[aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/89377)|

## Todo List

- [ ] 添加直接使用的分割demo
- [ ] 添加更多backbone的模型