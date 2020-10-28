%Load data into MATLAB
clear
load('incorrect_train.mat')
%train_index=index;%Remove later as temp fix for CNN prefinal model issue
load('incorrect_val.mat')
%val_index=index;%Remove later
load('incorrect_test.mat')
%test_index=index;%Remove later
load('f.mat')
load('t.mat')

%convert char array to string array for train, validation and test class labels
temp1=[];
temp2=[];
for i=1:length(y_train_inc)
   z1=y_train_inc(i,:);
   z1=z1(find(~isspace(z1)));%remove spacing
   temp1=[temp1;convertCharsToStrings(z1)];
   z2=y_train_inc_pred(i,:);
   z2=z2(find(~isspace(z2)));%remove spacing
   temp2=[temp2;convertCharsToStrings(z2)];   
end
y_train_inc=temp1;
y_train_inc_pred=temp2;

temp1=[];
temp2=[];
for i=1:length(y_val_inc)
   z1=y_val_inc(i,:);
   z1=z1(find(~isspace(z1)));%remove spacing
   temp1=[temp1;convertCharsToStrings(z1)];
   z2=y_val_inc_pred(i,:);
   z2=z2(find(~isspace(z2)));%remove spacing
   temp2=[temp2;convertCharsToStrings(z2)];   
end
y_val_inc=temp1;
y_val_inc_pred=temp2;

temp1=[];
temp2=[];
for i=1:length(y_test_inc)
   z1=y_test_inc(i,:);
   z1=z1(find(~isspace(z1)));%remove spacing
   temp1=[temp1;convertCharsToStrings(z1)];
   z2=y_test_inc_pred(i,:);
   z2=z2(find(~isspace(z2)));%remove spacing
   temp2=[temp2;convertCharsToStrings(z2)];   
end
y_test_inc=temp1;
y_test_inc_pred=temp2;


%%
%Choose what dataset to analyse
type='train';
%type='val';
%type='test';

%Determine amount of each class in incorrect dataset
switch(type)
    case 'train'
        a=unique(y_train_inc);
        truth_amount=[];
        for i=1:length(a)
           truth_amount=[truth_amount;sum(y_train_inc(:)==a(i))]; 
        end
        b=unique(y_train_inc_pred);
        pred_amount=[];
        for i=1:length(b)
           pred_amount=[pred_amount;sum(y_train_inc_pred(:)==b(i))]; 
        end
    case 'val'
        a=unique(y_val_inc);
        truth_amount=[];
        for i=1:length(a)
           truth_amount=[truth_amount;sum(y_val_inc(:)==a(i))]; 
        end
        b=unique(y_val_inc_pred);
        pred_amount=[];
        for i=1:length(b)
           pred_amount=[pred_amount;sum(y_val_inc_pred(:)==b(i))]; 
        end
    case 'test'
        a=unique(y_test_inc);
        truth_amount=[];
        for i=1:length(a)
           truth_amount=[truth_amount;sum(y_test_inc(:)==a(i))]; 
        end
        b=unique(y_test_inc_pred);
        pred_amount=[];
        for i=1:length(b)
           pred_amount=[pred_amount;sum(y_test_inc_pred(:)==b(i))]; 
        end
end

%Print number of examples for each class
disp(strcat({'Actual Classes Amount for '},type,' dataset'))
for i=1:length(a)
   disp(strcat(a(i)," : ",int2str(truth_amount(i)))); 
end
% disp("Prediction Classes Amount")
% for i=1:length(b)
%    disp(strcat(b(i)," : ",int2str(pred_amount(i)))); 
% end
fprintf('\n')
%%
%Loop through examples
%Please note that a break point should be placed on line 126 to view individual incorrect examples

ac_class="2_walking";%Class that is going to be tested: Change to desired class to analyse 

switch(type) %Determine offset in class array for indexing
    case 'train'
        offset=find(strcmp(y_train_inc,ac_class));
        offset=offset(1);
    case 'val'
        offset=find(strcmp(y_val_inc,ac_class));
        offset=offset(1);
    case 'test'
        offset=find(strcmp(y_test_inc,ac_class));
        offset=offset(1);
end
bypass=0;%To jump to a certain example
skip=1; %skip every n examples
for i=0:skip:truth_amount(a==ac_class)-1-bypass
    x=i+offset+bypass;
    close all
    switch(type)
        case 'train'
            S_dB= x_train_inc(x,:,:);%Choose example
            class= y_train_inc(x);%Choose example
            pred_class= y_train_inc_pred(x);%Choose example
        case 'val'
            S_dB= x_val_inc(x,:,:);%Choose example
            class= y_val_inc(x);%Choose example
            pred_class= y_val_inc_pred(x);%Choose example
        case 'test'
            S_dB= x_test_inc(x,:,:);%Choose example
            class= y_test_inc(x);%Choose example
            pred_class= y_test_inc_pred(x);%Choose example
    end
    S_dB=squeeze(S_dB);
    if(strcmp(class,'2_walking'))
        class='2\_walking'; 
    end
    if(strcmp(class,'sphere_swing'))
        class='sphere\_swing'; 
    end
    %Plot Spectrogram
    figure();
    imagesc(t,f,S_dB); % Abdul Gaffar: better to use imagesc than surf 
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
    xlabel('Time (s)'); % Abdul Gaffar (s)
    ylabel('Frequency (Hz)') % Abdul Gaffar (Hz)
    title(strcat({'Spectrogram of '},{type},{' dataset for '},class,{' class'}))

    hcol = colorbar;
    colormap('jet'); % Abdul Gaffar: change the colors used in the colorbar 
    set(hcol, 'FontName', 'Times New Roman', 'FontSize', 14)
    ylabel(hcol, 'Magnitude, dB')
end

%%
%Get indices for train and validation datasets

%CNN incorrect fail examples
%train_loc=[545,551,556,565,570,573,574,583,584,585,586,587,591,622,623,629,635,636,641,645,657,681,687,736, 762,774,777,786,801,810,816,903,924,927,930,933,118,158,186,190,226,230,238,246,334,338,422,989,1013,1041,1053,1057,1061,1065,1069,1173,1201,1209,1213,1225,1293,1313,1333,1341,1349,68,70,72,74,84,96,108,112,114,116,1,7,9,11,25,29,49,51,53,55,59,63,476,478,480,482,486,488,490,492,494,518,520];
%val_loc=[468,531,549,552,570,588,125,149,153,169,630,664,668,672,682,684,686,688,1,2,4,5,6,7,11,12];

%LSTM incorrect fail examples
train_loc=[2111,2223,2303,2431,2447,2495,2511,2527,2543,2655,2751,2815,2879,3055,3071,3087,3215,3327,3359,3743,622,650,664,678,692,762,804,916,930,944,958,972,1028,1070,1084,1098,1112,1140,1154,1168,1182,1266,1350,1406,1420,1434,1504,1532,1546,1560,1798,1966,2008,3782,3786,3794,3798,3826,3854,3862,3870,3874,3878,3882,3886,3898,3914,3946,3950,3962,3966,3970,3982,4022,4026,4038,4066,4074,4090,4094,4102,4106,4110,4134,474,476,502,504,506,508,514,518,522,524,526,528,530,534,536,538,596,610,612,21,26,31,36,41,61,66,71,76,126,2036,2039,2040,2041,2047,2048,2049,2050,2051,2059,2060,2061,2062,2063,2064,2065,2066,2082,2083,2087,2088,2089,2090,2091];
val_loc=[522,576,604,620,636,644,646,660,662,680,682,684,686,688,690,692,710,716,718,734,278,282,290,294,298,314,318,378,343,790,791,819,822,824,826,828,829,135,136,142,143,153,154,3,23,31,35,45,47,59,71,519,520];

%Remove once new data imported
% for i=1338:length(train_index)
%     train_index(i)=train_index(i)-18800+150*200;%Fix last batch error
% end

%Extract index locations in original dataset
train_remove_index=train_index(train_loc);
val_remove_index=val_index(val_loc);

%Save indices of the examples declared as a fail.
save('train_remove_index.mat','train_remove_index');
save('val_remove_index.mat','val_remove_index');
disp("DONE");