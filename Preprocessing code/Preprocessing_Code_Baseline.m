clear
path = 'C:\Users\Mohamed Zahier\Documents\UCT\4th year\EEE4022S\Thesis\Code\New Matlab\Radar_Data';%location of radar data
folders = dir2(strcat(path,'\*'));%extract folder folder info 
data={};%All data
labels={};%%All labels
train_data={};train_labels={};val_data={};val_labels={};test_data={};test_labels={};%Split data and labels
Time_limit=2;%3.65time limit is seconds
nfft=128;%Number of frequancy points length
OL=80;%Overlap of bins
win_len=100;%Window function length
win = hamming(win_len);%Window
val_perc=10;%Validation data percentage
test_perc=10;%test data percentage
count=0;%Number of examples skipped
countclass={};%classes of the examples skipped
Example_overlap=0;%Amount that examples overlap with each other(0->1)
for i=1:length(folders) %Loop through folders
    trackdatas=dir2(strcat(path,'\',folders(i).name));
    if(length(trackdatas)==1)%Extract trkdata name
        trackdata=trackdatas(1).name;%No radardata folder
    elseif(length(trackdatas)==2)
        trackdata=trackdatas(1).name;%When there is no rjktrkdata
    else
        trackdata=trackdatas(2).name;%When there is rjktrkdata
    end
    load(strcat(path,'\',folders(i).name,'\',trackdata)) %Load trkdata struct
    for example_number=1:length(trkdata) %Loop through samples
        fs=trkdata(example_number).PRF;
        example=double(trkdata(example_number).trk_data_real)+double(1i*trkdata(example_number).trk_data_imag);
        class=char(trkdata(example_number).class);
        start_time=str2double(trackdata(12:13))*100*100+str2double(trackdata(15:16))*100+str2double(trackdata(18:19));
        if(length(example)<Time_limit*fs)
            count=count+1;
            countclass=[countclass;class];
            continue;%Skip small examples
        end
        %Get spectrogram
        %Custom spectrogram
        [S_dB, f, t] = stft_own(example,win, OL, nfft, fs);
        S_dB = 20*log10(abs(S_dB));
        %Perform trimming/padding
        %time limit is seconds
        samples={};
        if(t(end)>=Time_limit)%Example longer then time limit specified
            Tindex=find(t <= Time_limit, 1, 'last');%Frame that will be used for limiting
            Limhop=Tindex-floor(Tindex*Example_overlap);
            numsamp=floor(length(t)-Tindex)/Limhop+1; %numsamp=floor(t(end)/Time_limit);%Number of samples available to extract
            for j=0:numsamp-1 %j=1:numsamp
                samples=[samples;S_dB(:,1+j*Limhop:Tindex+j*Limhop)];%[samples;S_dB(:,((j-1)*Tindex+1):j*Tindex)];%Split samples
            end
        else
            Time_frameL=floor((Time_limit*fs-length(win)/2)*1/(length(win)-OL))+1;
            samples=[samples;[S_dB,zeros(length(f),Time_frameL-length(t))]];%Pad the time axes
        end
        %Save into matrix
        for ns=1:length(samples)
            data=[data;cell2mat(samples(ns))];
            labels=[labels;class];
        end
        S_dB=[]; 
    end
end


%Split original dataset into train, validation and test datasets based on the classes.
a=unique(labels,'stable'); %extract unique classes in original dataset
amount=cell2mat(cellfun(@(x) sum(ismember(labels,x)),a,'un',0));%determine number of examples per class in original dataset
for i=1:length(a)
    val_len=floor(amount(i)*val_perc/100);
    test_len=floor(amount(i)*test_perc/100);
    train_len=amount(i)-val_len-test_len;
    indices = find(strcmp(labels, char(a(i))));
    train_ind=indices(1:train_len);%indices(val_len+1:(train_len+val_len));%
    val_ind=indices(train_len+1:(train_len+val_len));%indices(1:val_len);%
    test_ind=indices((train_len+val_len+1):(train_len+val_len+test_len));
    
    for j=1:train_len
        train_data=[train_data;data(train_ind(j))];
        train_labels=[train_labels;labels(train_ind(j))];
    end
    for j=1:val_len
        val_data=[val_data;data(val_ind(j))];
        val_labels=[val_labels;labels(val_ind(j))];
    end
    for j=1:test_len
        test_data=[test_data;data(test_ind(j))];
        test_labels=[test_labels;labels(test_ind(j))];
    end        
end

%Save datasets as .mat file
save('train_data.mat', 'train_data','-v7.3');
save('train_labels.mat', 'train_labels');
save('val_data.mat', 'val_data');
save('val_labels.mat', 'val_labels');
save('test_data.mat', 'test_data');
save('test_labels.mat', 'test_labels');

%Display number of examples per class
a=unique(labels,'stable');
amount=cellfun(@(x) sum(ismember(labels,x)),a,'un',0);
for i=1:length(a)
   disp(strcat(char(a(i))," : ",int2str(cell2mat(amount(i))))) 
end


%% STFT Function definition 
function [S, f, t] = stft_own(y,win, overlap, nfft, fs)

% Abdul Gaffar: in your case, you want to compute positive and negative frequencies

y = y(:);                   % Coverting the signal y to a column-vector 
ylen = length(y);           % Signal Length
wlen = length(win);         % window function length should be 1024

% Calculate the number of important FFT points
% nip = ceil((1+nfft)/2); % Abdul Gaffar: applicable if you only want to compute positive frequencies
                          % Abdul Gaffar: in our case, we want to compute both positive and negative frequencies

nip = nfft;   % Abdul Gaffar, updated the formula 
                          
% Calculate the number of frames to be taken, given the signal size and amount of overlap
hop=wlen-overlap;
frames = 1+floor((ylen-wlen)/(hop)); 

% Initiation of the STFT matrix to store frames after FFT 
S = zeros(nip,frames); 

% Executing the STFT 
for i = 0:frames-1
    windowing = y(1+i*hop : wlen+i*hop).*win;  % windowing of the sampled data that moves 'overlap' samples for respective frame  
    Y = fftshift(fft(windowing, nfft));                % Abdul Gaffar: fftshift used because our frequency axis has both positive and negative values
                                                       % Calculating fft with 1024 points 
    S (:, 1+i) = Y(1:nip);                             % Updating STFT matrix with unique fft points (one-sided spectrum) 
end 

% Calculating f and t vectors 
t = (wlen/2:hop:wlen/2+(frames-1)*hop)/fs;

% f = (0:nip-1)*fs/nfft; % Abdul Gaffar: correct for positive frequencies only
f = (-nfft/2:1:(nfft/2-1))*fs/nfft; % Abdul Gaffar: correct for positive frequencies only


% S = abs(S); % Abdul Gaffar, return the complex values and not the magnitude 

end 
%% Custom dir function to get rid of .  and ..
% Source: https://stackoverflow.com/questions/27337514/matlab-dir-without-and
function listing = dir2(varargin)

if nargin == 0
    name = '.';
elseif nargin == 1
    name = varargin{1};
else
    error('Too many input arguments.')
end

listing = dir(name);

inds = [];
n    = 0;
k    = 1;

while n < 2 && k <= length(listing)
    if any(strcmp(listing(k).name, {'.', '..'}))
        inds(end + 1) = k;
        n = n + 1;
    end
    k = k + 1;
end

listing(inds) = [];
end 