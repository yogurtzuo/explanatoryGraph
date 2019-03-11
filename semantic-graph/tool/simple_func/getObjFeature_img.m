function [res,I_patch]=getObjFeature_img(I_patch,theConf,theNet)
I_patch=single(I_patch);


batch = size(I_patch,4);
res = cell(1,batch);
path = '../../';
save([path , '/tmp/img_batch.mat'], 'I_patch');
disp('save img_batch.mat');

diclength = 12;

system('python pytorch-pruning/forward_featureMap_batch.py');


features = load([path , '/tmp/feature_map_batch.mat']);
for imgId=1:batch
    for index=0:diclength
        name = 'x';
        eval([name, '=features.img_', num2str(imgId), '.layer_', num2str(index),';']);
        c = size(x,2);
        h = size(x,3);
        y = zeros(h, h, c);
        %for j=1:c
        %    y(:,:,j) = x(1,j,:,:);
        t = zeros(c,h,h);
        t(:,:,:) = x(1,:,:,:);
        y = permute(t,[2,3,1]);
        res_(index+2).x = y; 
        clear t,y;
    end 
    res{imgId} = res_;
end
clear res_;

I_patch=uint8(I_patch);
end
