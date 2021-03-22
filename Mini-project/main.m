%%
clear;
close all;

image = imread('sample1.png');

rows = size(image, 1);
cols = size(image, 2);
channels = size(image, 3);
patch = 15;
%% estimation of dark channel, atmosphere light, t'(x)
dark = dark_channel(image, patch);
A = estimate_atmos(image, dark);
t_hat = estimate_trate(image, patch, A);
%% soft matting and dehazing
t = soft_matting(t_hat, double(image));
%% dehazing
haze_free = zeros(rows, cols, channels);
t0 = 0.1;

for i = 1 : rows
    for j = 1:cols
        haze_free(i, j, :) = (int32(image(i, j, :) - A))/max(t(i, j), t0) + A;
        fprintf("%d %d %d \n", int32(haze_free(i,j,1)), int32(haze_free(i,j,2)), int32(haze_free(i,j,3)));
    end
end