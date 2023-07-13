close all; clear;
addpath('C:\caffe-cmu\Build\x64\Release\matcaffe');
%addpath('C:\caffe-windows\build\Matlab');
caffe.set_mode_gpu();
caffe.set_device(1); 

%% load data
age = xlsread('C:\Users\won\Desktop\face_emotion_data\dataset\all\convolution\emotion_train.xlsx','B1:B1405');
[tmp, n_img] = xlsread('C:\Users\won\Desktop\face_emotion_data\dataset\all\convolution\emotion_train.xlsx','A1:A1405');
%%

net_weights = 'C:/Users/won/Desktop/face_emotion_data/dataset/VGG19/VGG_ILSVRC_19_layers.caffemodel';
net_model = 'C:/Users/won/Desktop/face_emotion_data/dataset/VGG19/deploy.prototxt';
net_solver = 'C:/Users/won/Desktop/face_emotion_data/dataset/VGG19/solver.prototxt';
caffe_solver = caffe.Solver(net_solver);
caffe_solver.net.copy_from(net_weights);

%vgg_net = caffe.Net(vgg_net_model, net_weights, phase);
%caffe_solver = caffe.Solver(model.solver_def_file);
%caffe_solver.net.copy_from(model.net_file);
%caffe_solver.net.copy_from(net_weights);

iter = 50000*20;
image_size = 224;
batchSize = 10;
n_scale = 1;
curIndex =1;

%% train
for iter=1:iter
    idx = 1:size(age,1);
    idx = idx(randperm(size(age,1)));
   
    batchLabel = zeros(1,1,1,batchSize,'single');
    for batchNo=1:batchSize
        if curIndex>size(age,1)
            curIndex = 1;
        end
        curFileName = sprintf('%s',n_img{idx(1,curIndex),1});
      %  im2 = imread(curFileName);
        im = caffe.io.load_image(curFileName);
        
       % imshow(im);
        
        batchIMs{1}(:,:,:,batchNo) = imresize(im, [image_size image_size], 'bilinear');  % resize im_data
        
        batchLabel(1,batchNo) = age(idx(1,curIndex),1);
        
        curIndex = curIndex + 1;
    end

    caffe_solver.net.blobs('data').set_data(batchIMs{1});
    caffe_solver.net.blobs('label').set_data(batchLabel);
     
    caffe_solver.step(1);
    
    loss = caffe_solver.net.blobs('prob_vgg19_emotion').get_data();
    
    p = sprintf('epoch: %d, iter: %d, loss: %f ',floor(iter*batchSize/size(age,1)), iter, loss);
    disp(p);
    
end