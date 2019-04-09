clc;
clear;
close all;

actions = {'up', 'down', 'left', 'right', 'circle', 'cross'};
fid = fopen('6in1.csv');
name_start_end = textscan(fid, '%s %d32 %d32','delimiter', ',');


data_amp = zeros([length(name_start_end{1,1}), 52,192]);
data_pha = zeros([length(name_start_end{1,1}), 52,192]);
label_segmentation = zeros([length(name_start_end{1,1}), 192]);
label_mask = zeros([length(name_start_end{1,1}), 192]);
label_time = zeros([length(name_start_end{1,1}),2]);

for file_index = 1: length(name_start_end{1,1})
    file_name = name_start_end{1,1}{file_index,1};
    strt = name_start_end{1,2}(file_index);
    ends = name_start_end{1,3}(file_index);
    

    file_index/length(name_start_end{1,1})
    
    for i = 1:6
       if contains(file_name, actions{1,i})
            break; 
       end
    end
    
    
    load(['../', actions{1,i}, '/', file_name, '.mat']);
    amplitude = abs(ChannelEst);
    phase = unwrap(angle(ChannelEst));
  
    data_amp(file_index, :, :) = amplitude';
    data_pha(file_index, :, :) = phase';
    
    label_segmentation(file_index, strt:ends) = i;
    label_mask(file_index, strt:ends) = 1;
    label_time(file_index,:) = [strt,ends];

end


% test_data_amp = [];
% test_data_pha = [];
% test_label_segmentation = [];
% test_label_mask = [];
% test_label_time = [];
% 
% 
% train_data_amp = [];
% train_data_pha = [];
% train_label_segmentation = [];
% train_label_mask = [];
% train_label_time = [];


test_index = find( rem(1:length(name_start_end{1,1}),5)==0);
train_index = find( rem(1:length(name_start_end{1,1}),5)~=0);


test_data_amp = data_amp(test_index,:,:);
test_data_pha = data_pha(test_index,:,:);
test_label_instance = label_segmentation(test_index,:);
test_label_mask = label_mask(test_index,:);
test_label_time = label_time(test_index,:);


train_data_amp = data_amp(train_index,:,:);
train_data_pha = data_pha(train_index,:,:);
train_label_instance = label_segmentation(train_index,:);
train_label_mask = label_mask(train_index,:);
train_label_time = label_time(train_index,:);


% for file_index = 1: length(name_start_end{1,1})
%      if rem(file_index,5)==0
% %         test_index = [test_index, file_index];
%          
%         test_data_amp = [test_data_amp; data_amp(file_index,:,:)];
%         test_data_pha = [test_data_pha; data_pha(file_index,:,:)];
%         test_label_segmentation = [test_label_segmentation;  label_segmentation(file_index,:)];
%         test_label_mask = [test_label_mask; label_mask(file_index,:)];
%         test_label_time = [test_label_time; label_time(file_index,:)];
% 
%      else
%          
%         train_data_amp = [train_data_amp; data_amp(file_index,:,:)];
%         train_data_pha = [train_data_pha; data_pha(file_index,:,:)];
%         train_label_segmentation = [train_label_segmentation;  label_segmentation(file_index,:)];
%         train_label_mask = [train_label_mask; label_mask(file_index,:)];
%         train_label_time = [train_label_time; label_time(file_index,:)];
% 
%      end  
% end

save('train_data.mat', 'train_data_amp', 'train_data_pha', 'train_label_instance', 'train_label_mask', 'train_label_time');
save('test_data.mat', 'test_data_amp', 'test_data_pha', 'test_label_instance', 'test_label_mask', 'test_label_time');







