function objset=readAnnotation(Name_batch,theConf, className)
MaxObjNum=2000;

minArea=50^2;

objset(MaxObjNum).folder=[];
objset(MaxObjNum).filename=[];
objset(MaxObjNum).name=[];
objset(MaxObjNum).bndbox=[];
objset(MaxObjNum).ID=[];
annotationfile=[theConf.data.imgdir, className, '/filenames.mat'];
disp(annotationfile)
file = load([annotationfile]);
filenames = file.filenames;
j=0;

nameLen = 0;
if length(filenames)>MaxObjNum
    nameLen = MaxObjNum;
else
    nameLen = length(filenames);
end

for i=1:nameLen
%    bndbox.xmin=int2str(truth(i).obj.bndbox.Wmin);
%    bndbox.xmax=int2str(truth(i).obj.bndbox.Wmax);
%    bndbox.ymin=int2str(truth(i).obj.bndbox.Hmin);
%    bndbox.ymax=int2str(truth(i).obj.bndbox.Hmax);
%    if(~IsAreaValid(bndbox,minArea))
%        continue;
%    end
    j=j+1;
    objset(j).folder=['./', className];
    objset(j).filename=filenames{i};
    objset(j).name=Name_batch;
%    objset(j).bndbox=bndbox;
    objset(j).ID=j;
end
objset=objset(1:j);
