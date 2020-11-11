clear
load('train_data.mat')
load('train_labels.mat')
load('val_data.mat')
load('val_labels.mat')
load('test_data.mat')
load('test_labels.mat')
load('f.mat')
load('t.mat')
%Choose example index (Look at the corresponding dataset labels to determine what example to choose based on the class required) 
x=20000;
%Choose dataset by uncommeting relevent option
type='train';
%type='val';
%type='test';

%Choose dataset by uncommeting relevent option
S_dB= cell2mat(train_data(x));%Choose example
%S_dB= cell2mat(val_data(x));%Choose example
%S_dB= cell2mat(test_data(x));%Choose example

%Choose dataset by uncommeting relevent option
class= cell2mat(train_labels(x));%Choose example
%class= cell2mat(val_labels(x));%Choose example
%class= cell2mat(test_labels(x));%Choose example

%Compensate for the underscore syntax 
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