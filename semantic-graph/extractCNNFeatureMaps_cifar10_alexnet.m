function extractCNNFeatureMaps_cifar10_alexnet(netname, className)
isUseGPU=true; %true; Please set isUseGPU=true, when you use GPUs.
tarSize = [227 227];

mkdir('./mat/cifar_10_alexnet/', netname);
net = load(['../pretrained_cnns/imagenet-caffe-alex.mat']);
%%download from http://www.vlfeat.org/matconvnet/pretrained/ 


% extract images
conf.data.catedir='./data/cifar_10/';
conf.data.imgdir='./data/cifar_10/Image/';
conf.data.readCode='./data_input/data_input_cifar10/';
addpath(genpath(conf.data.readCode));
addpath(genpath('tool'));
objset=readAnnotation(netname,conf,className);
objNum=numel(objset);
images=zeros(tarSize(1),tarSize(2),3,objNum,'uint8');
for objID=1:objNum
    I=getI(objset(objID),conf,tarSize,false);
    images(:,:,:,objID)=uint8(I);
end
images_neg=getNegativeImages(tarSize);
save(['./mat/cifar_10_alexnet/',netname,'/images.mat'],'images','images_neg');
clear images;


%% extract CNN features
addpath(genpath('./tool'));
conf=getNetParameters('AlexNet',net); % this function returns structural configurations of VGG16, ResNets, and VAEGAN.
addpath(genpath(conf.convnet.codedir));
load(['./mat/cifar_10_alexnet/',netname,'/images.mat'],'images','images_neg');
[batch_f,batch_f_flip,stat_all]=getRoughCNN(net,conf,images,images_neg);
clear images;
save(['./mat/cifar_10_alexnet/',netname,'/roughCNN.mat'],'batch_f','batch_f_flip','conf','stat_all');
