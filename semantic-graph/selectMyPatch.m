function selectMyPatch(netname)


targetLayers = [2,5,8,10,12];
layerList=[1,2,3,4,5];
scale_show=150;
topPatchNum_forRank=300;
topPatchNum_show=5;
topPatternNum=20;


addpath(genpath('./tool'));
load(['./mat/cifar_10_lenet/',netname,'/roughCNN.mat'],'batch_f','conf');
load(['./mat/cifar_10_lenet/',netname,'/model_2_5_8_10_12.mat'],'model');
%lc=0;
for layerID=layerList

    oriLayer = targetLayers(layerID);
    [xh,~,channelNum] = size(batch_f{1}(oriLayer).x); 
    prob=model.layer(layerID).prob_record;
    patternNum = size(prob, 1)/channelNum;


    tmp=sort(prob,2,'descend');
    tmp=sum(tmp(:,1:topPatchNum_forRank),2);
    [~,idx_m]=sort(tmp,'descend');
    %idx_m is what we want
    [channelIm, indx] = sumPattern(tmp, patternNum, channelNum);
    save(['./mat/cifar_10_lenet/',netname,'/index_sumPattern_layer_', int2str(oriLayer), '.mat'], 'indx', 'channelIm', 'oriLayer');
    [totalIm, idx_Im] = singlePattern(tmp, idx_m, patternNum, channelNum);
    save(['./mat/cifar_10_lenet/',netname,'/index_singlePattern_layer_', int2str(oriLayer), '.mat'], 'idx_Im', 'totalIm', 'oriLayer');
end
end


function [channelIm, indx] = sumPattern(tmp, patternNum, channelNum)
    channelIm = zeros(1,channelNum);
    for i=0:channelNum-1
        channelSum = 0;
        for par=1:patternNum
            index = par + i*patternNum;
            channelSum = channelSum + tmp(index);
        end
        channelIm(i+1) = channelSum;
    end
    [channelIm, indx] = sort(channelIm,'descend');
end

function [totolIm, idx_Im] = singlePattern(tmp, idx_m, patternNum, channelNum)
    idx_Im = ceil(idx_m./patternNum);
    totolIm = sort(tmp, 'descend');
end











