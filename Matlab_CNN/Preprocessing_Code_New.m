clear
path = 'C:\Users\Mohamed Zahier\Documents\UCT\4th year\EEE4022S\Thesis\Code\New Matlab\Radar_Data';
folders = dir2(strcat(path,'\*'));
data={};
labels={};
Time_limit=3;%time limit is seconds
nfft=100;%Window function length
OL=80;%Overlap of bins
win = hamming(nfft);%Window
for i=1:length(folders) %Loop through folders
    trackdatas=dir2(strcat(path,'\',folders(i).name));
    if(length(trackdatas)==2)%Extract trkdata name
       trackdata=trackdatas(2).name;
    else
        trackdata=trackdatas(1).name;%When there is no rjktrkdata
    end
    load(strcat(path,'\',folders(i).name,'\',trackdata)) %Load trkdata struct
    for example_number=1:length(trkdata) %Loop through samples
        fs=trkdata(example_number).PRF;
        example=double(trkdata(example_number).trk_data_real)+double(1i*trkdata(example_number).trk_data_imag);
        class=char(trkdata(example_number).class);
        %Perform additional processing if required eg filtering
        %Get spectrogram
        %Custom spectrogram
        [S_dB, f, t] = stft_own(example,win, OL, nfft, fs);
        S_dB = 20*log10(abs(S_dB));
        %Perform trimming/padding
        %time limit is seconds
        samples={};
        if(t(end)>=Time_limit)
            numsamp=floor(t(end)/Time_limit);%Number of samples available to extract
            Tindex=find(t <= Time_limit, 1, 'last');%Frame that will be used for limiting
            for j=1:numsamp
                samples=[samples;S_dB(:,((j-1)*Tindex+1):j*Tindex)];%Split samples
            end
        else
            Time_frameL=floor((Time_limit*fs-length(win)/2)*1/(length(win)-OL))+1;
            samples=[samples;[S_dB,zeros(length(f),Time_frameL-length(t))]];%Pad the time axes
        end
        %Perform downsampling if required
        %freq_ds=2;%Amount to downsample frequancy axis
        %time_ds=2;%%Amount to downsample time axis
        %S_dB=S_dB(1:freq_ds:end,1:time_ds:end);
        %Save into matrix
        for ns=1:length(samples)
            data=[data;cell2mat(samples(ns))];
            labels=[labels;class];
        end
        S_dB=[]; 
    end
end
%FIX diffrent sample size due to diffrent PRF
[minsize, minidx] = min(cellfun('size', data, 2));%Temporary solution by trimming to smallest frame
data=cellfun(@(x) x(:,1:minsize),data,'uni',false);
%Save as .mat file
save('data.mat', 'data');
save('labels.mat', 'labels');

%Determine amount of diffrent classes
% a=unique(labels,'stable');
% amount=cellfun(@(x) sum(ismember(labels,x)),a,'un',0);
% for i=1:length(a)
%    disp(strcat(char(a(i))," : ",int2str(cell2mat(amount(i))))) 
% end
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