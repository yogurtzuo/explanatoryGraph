function batch_f_c = selectTargetLayers(batch_f, conf)

batch_f_c = cell(length(batch_f), 1);
for in = 1:length(batch_f)
    res = batch_f{in};
    num = length(res);
    res_c = repmat(struct('size',[],'x',[],'rangeX',[],'minX',[]),[1,num]);
    for i=conf.convnet.targetLayers
        res_c(i).x = res(i).x;
        res_c(i).size = res(i).size;
        res_c(i).minX = res(i).minX;
        res_c(i).rangeX = res(i).rangeX; 
    end
    batch_f_c{in} = res_c;
end
clear batch_c
end
