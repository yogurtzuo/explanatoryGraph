function conf=getNetParameters(type,net)
if(nargin<2)
    net=[];
end

try
    parpool;
catch
    delete(gcp);
    parpool;
end
p=gcp;
conf.parallel.parpoolSize=p.NumWorkers;
switch(type)
    case 'AlexNet'
        conf=getConvNet_AlexNet(conf,net);
    case 'LeNet'
        conf=getConvNet_LeNet(conf);
 
    otherwise
        % please design your own net parameters
end
end

function convnet=getConvNetPara_alexnet(convnet,net)
len=length(convnet.convLayers);
convnet.lastLayer=length(net.layers)-1;
convnet.total_targetLayers = 1+convnet.convLayers;
convnet.targetLayers=1+convnet.convLayers;
convnet.targetScale=zeros(1,len);
convnet.targetStride=zeros(1,len);
convnet.targetCenter=zeros(1,len);
for i=1:len
    tarLay=convnet.matconvLayers(i);
    layer=net.layers{tarLay};
    if(((~strcmp(layer.type,'conv'))&&(~strcmp(layer.type,'dagnn.Conv')))||(var(layer.pad)>0)||(layer.stride(1)~=layer.stride(2)))
        error('Errors in function getConvNetPara.');
    end
    pad=layer.pad(1);
    scale=size(layer.weights{1}, 1);
    display(scale);
    stride=layer.stride(1);
    if(i==1)
        convnet.targetStride(i)=stride;
        convnet.targetScale(i)=scale;
        convnet.targetCenter(i)=(1+scale-pad*2)/2;
    else
        IsPool=false;
        poolStride=0;
        poolSize=0;
        poolPad=0;
        for j=convnet.matconvLayers(i-1)+1:tarLay-1
            if(strcmp(net.layers{j}.type,'pool'))
                IsPool=true;
                poolSize=net.layers{j}.pool(1);
                poolStride=net.layers{j}.stride(1);
                poolPad=net.layers{j}.pad(1);
            end
        end
        convnet.targetStride(i)=(1+IsPool*(poolStride-1))*stride*convnet.targetStride(i-1);
        convnet.targetScale(i)=convnet.targetScale(i-1)+IsPool*(poolSize-1)*convnet.targetStride(i-1)+convnet.targetStride(i)*(scale-1);
        if(IsPool)
            convnet.targetCenter(i)=(scale-pad*2-1)*poolStride*convnet.targetStride(i-1)/2+(convnet.targetCenter(i-1)+convnet.targetStride(i-1)*(poolSize-2*poolPad-1)/2);
        else
            convnet.targetCenter(i)=(scale-pad*2-1)*convnet.targetStride(i-1)/2+convnet.targetCenter(i-1);
        end
    end
end
convnet.targetLayers=convnet.targetLayers(convnet.validLayers);
convnet.targetScale=convnet.targetScale(convnet.validLayers);
convnet.targetStride=convnet.targetStride(convnet.validLayers);
convnet.targetCenter=convnet.targetCenter(convnet.validLayers);
end





function conf=getConvNet_AlexNet(conf,net)
convnet.codedir='../matconvnet-1.0-beta24/matlab/';
convnet.matconvLayers=[1,5,9,11,13]
convnet.convLayers=[1, 4, 7,9,11];
convnet.validLayers=[1,2,3,4,5];
convnet.imgSize=[227,227];
conf.convnet=getConvNetPara_alexnet(convnet,net);


conf.learn.positionCandNum=6;
conf.learn.patternDensity=[0.025,0.05,0.1,0.1,0.1]./1.0;
conf.learn.search_.maxRange=[0.3,0.3,0.3,0.3,0.3];
conf.learn.search_.deform_ratio=3;
conf.learn.deform_.init_delta=0.15;
conf.learn.deform_.max_delta=1.0./sqrt(conf.learn.patternDensity.*([56, 27, 13,13,13].^2));
conf.learn.deform_.min_delta=conf.learn.deform_.max_delta./10;
conf.learn.map_delta=0.025;
conf.learn.topN=[15,15,15,15,0];
conf.learn.topM=[600,600,600,600,0];
conf.learn.validTau=0.1;


end


function convnet=getConvNetPara_lenet(convnet)
len=length(convnet.convLayers);
convnet.total_targetLayers = 1+convnet.convLayers;
convnet.targetLayers=1+convnet.convLayers;
convnet.targetScale=[5,18];
convnet.targetStride=[1,3];
convnet.targetCenter=[3,7];
end



function conf=getConvNet_LeNet(conf)
convnet.codedir='../matconvnet-1.0-beta24/matlab/';
convnet.convLayers=[1, 4];
convnet.validLayers=[1,2];
convnet.imgSize=[32,32];
conf.convnet=getConvNetPara_lenet(convnet);


conf.learn.positionCandNum=6;
conf.learn.patternDensity=[0.05, 0.1]./1.0;
conf.learn.search_.maxRange=[0.3,0.3];
conf.learn.search_.deform_ratio=3;
conf.learn.deform_.init_delta=0.15;
conf.learn.deform_.max_delta=1.0./sqrt(conf.learn.patternDensity.*([28, 10].^2));
conf.learn.deform_.min_delta=conf.learn.deform_.max_delta./10;
conf.learn.map_delta=0.025;
conf.learn.topN=[15,0];
conf.learn.topM=[30,0];
conf.learn.validTau=0.1;


end



