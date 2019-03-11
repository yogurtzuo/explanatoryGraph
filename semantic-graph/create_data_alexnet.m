function create_data_alexnet()

classes = {'car', 'plane', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
className = {'class_1', 'class_0', 'class_2', 'class_3','class_4', 'class_5','class_6', 'class_7','class_8', 'class_9'};
for i=1:length(classes)
    disp('create_imagedata');
    disp(classes{i});
    disp(className{i});
    extractCNNFeatureMaps_cifar10_alexnet(classes{i}, className{i});
    learnExplan_byLayer(classes{i});
end
end

function learnExplan_byLayer(netname)
    learn_explanatoryGraph(netname);
end
