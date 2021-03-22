%%
clear;
close all;

image = imread('./images/sample3.jpeg');

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
% t = t_hat;
%% dehazing
haze_free = zeros(rows, cols, channels, 'uint8');
t0 = 0.1;

for i = 1 : rows
    for j = 1:cols
        haze_free(i, j, 1) = uint8((double(image(i, j, 1)) - double(A(1, 1)))/max(t(i, j), t0) + double(A(1,1)));
        haze_free(i, j, 2) = uint8((double(image(i, j, 2)) - double(A(1, 2)))/max(t(i, j), t0) + double(A(1,2)));
        haze_free(i, j, 3) = uint8((double(image(i, j, 3)) - double(A(1, 3)))/max(t(i, j), t0) + double(A(1,3)));
        fprintf("%d %d %d \n", int32(haze_free(i,j,1)), int32(haze_free(i,j,2)), int32(haze_free(i,j,3)));
    end
end


figure, imshow(image);
figure, imshow(haze_free);