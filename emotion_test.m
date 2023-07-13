close all; clear;

addpath('C:\caffe-cmu\Build\x64\Release\matcaffe');

caffe.set_mode_gpu();
caffe.set_device(0); 

%% load data
emotion = xlsread('C:\Users\won\Desktop\face_emotion_data\dataset\all\convolution\emotion_test.xlsx','B1:B347');
[tmp, n_img] = xlsread('C:\Users\won\Desktop\face_emotion_data\dataset\all\convolution\emotion_test.xlsx','A1:A347');
%%

net_weights = 'C:/Users/won/Desktop/emotion_progrem/¿À¸®/vgg_16_emotion_iter_1000000.caffemodel';
net_model = 'C:/Users/won/Desktop/face_emotion_data/dataset/VGG16/deploy_emotion_test.prototxt';
net = caffe.Net(net_model, net_weights, 'test');

%%%%%%%%% test  morcap
iter = 347; % data ÃÑ °¹¼ö / batchSize 
image_size = 224;
batchSize = 1;
curIndex =1;

acu_total = 0;

acu_anger = 0;
acu_contempt = 0;
acu_disgust = 0;
acu_fear = 0;
acu_happiness = 0;
acu_neutral = 0;
acu_sadness = 0;
acu_surprise = 0;

count_1 = 0;
count_2 = 0;
count_3 = 0;
count_4 = 0;
count_5 = 0;
count_6 = 0;
count_7 = 0;
count_8 = 0;

%% test
for iter=1:iter
   
    batchLabel = zeros(1,1,1,batchSize,'single');
    
    for batchNo=1:batchSize
        if curIndex>size(emotion,1)
            curIndex = 1;
        end
        
       curFileName = sprintf('%s',n_img{curIndex,1});

       im = caffe.io.load_image(curFileName);
        
      %  imshow(im2);
        
        batchIMs{1}(:,:,:,batchNo) = imresize(im, [image_size image_size], 'bilinear');  % resize im_data
        
        batchLabel(1,batchNo) = emotion(curIndex,1);
        
        curIndex = curIndex + 1;
    end
    
    net.forward(batchIMs);
    
    %value = net.params('conv5_3', 1).get_data(); %net.params('conv1_1').value;
    %q = sprintf('conv1_2 value: %f', value);
    %disp(q);

    prob = net.blobs('fc8_emotion_test').get_data();
    
    for n_batch = 1:size(prob,2)
       [m_val,m_idx] = max(prob(:,n_batch));
       if (m_idx-1) == batchLabel(1,n_batch)
       %if (floor(prob) == batchLabel(1,n_batch))
           acu_total = acu_total +1;
           
           if(batchLabel(1,n_batch) == 1)
               acu_anger = acu_anger + 1;
               
           elseif(batchLabel(1,n_batch) == 2)
               acu_contempt = acu_contempt + 1;
               
           elseif(batchLabel(1,n_batch) == 3)
               acu_disgust = acu_disgust + 1;
           
           elseif(batchLabel(1,n_batch) == 4)
               acu_fear = acu_fear + 1;
           
           elseif(batchLabel(1,n_batch) == 5)
               acu_happiness = acu_happiness + 1;
           
           elseif(batchLabel(1,n_batch) == 6)
               acu_neutral = acu_neutral + 1;
           
           elseif(batchLabel(1,n_batch) == 7)
               acu_sadness = acu_sadness + 1;
           
           elseif(batchLabel(1,n_batch) == 8)
               acu_surprise = acu_surprise + 1;
           end
       end
       
       if(batchLabel(1,n_batch) == 1)
            count_1 = count_1 + 1;
       
       elseif(batchLabel(1,n_batch) == 2)
            count_2 = count_2 + 1;
            
       elseif(batchLabel(1,n_batch) == 3)
            count_3 = count_3 + 1;
       
       elseif(batchLabel(1,n_batch) == 4)
            count_4 = count_4 + 4;
            
       elseif(batchLabel(1,n_batch) == 5)
            count_5 = count_5 + 1;
       
       elseif(batchLabel(1,n_batch) == 6)
            count_6 = count_6 + 1;
       
       elseif(batchLabel(1,n_batch) == 7)
            count_7 = count_7 + 1;
            
       elseif(batchLabel(1,n_batch) == 8)
            count_8 = count_8 + 1;
       
       end
    end
    
end

acu_total = acu_total / (347); 
count_total = count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7 + count_8;

a = sprintf('toroal accuracy: %f, %d', acu_total, count_total);
disp(a);

b= sprintf('anger accuracy : %f, %d', acu_anger, count_1);
disp(b);

c= sprintf('contempt accuracy : %f, %d', acu_contempt, count_2);
disp(c);

d= sprintf('disgust accuracy : %f, %d', acu_disgust, count_3);
disp(d);

e = sprintf('fear accuracy : %f, %d', acu_fear, count_4);
disp(e);

f = sprintf('happiness accuracy : %f, %d', acu_happiness, count_5);
disp(f);

g = sprintf('neutral accuracy : %f, %d', acu_neutral, count_6);
disp(g);

h = sprintf('sadness accuracy : %f, %d', acu_sadness, count_7);
disp(h);

i = sprintf('surprise accuracy : %f, %d', acu_surprise, count_8);
disp(i);