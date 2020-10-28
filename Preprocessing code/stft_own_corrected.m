%% STFT Function definition 
function [S, f, t] = stft_own_corrected(y,win, overlap, nfft, fs)

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