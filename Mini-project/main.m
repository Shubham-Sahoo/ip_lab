%%
clear;
close all;

image = imread('./images/sample2.jpeg');

rows = size(image, 1);
cols = size(image, 2);
channels = size(image, 3);
patch = 15;
%% estimation of dark channel, atmosphere light, t'(x)
dark = dark_channel(image, patch);
%disp(size(dark));
%figure, imshow(dark,[0 255]);
%%
A = estimate_atmos(image, dark);
%figure, imshow(A,[0 255]);
%%
t_hat = estimate_trate(image, patch, A);
%f = @() estimate_trate(image, patch, A);
%time_val = timeit(f);
%disp(time_val);
figure, imshow(t_hat);
%% soft matting and dehazing
t = soft_matting(t_hat, double(image));
%f = @() soft_matting(t_hat, double(image));
%time_val = timeit(f);
%disp(time_val);
figure, imshow(t);
% t = t_hat;
%% dehazing without softmating
haze_free_s = zeros(rows, cols, channels, 'uint8');
t0 = 0.1;

for i = 1 : rows
    for j = 1:cols
        haze_free_s(i, j, 1) = uint8((double(image(i, j, 1)) - double(A(1, 1)))/max(t_hat(i, j), t0) + double(A(1,1)));
        haze_free_s(i, j, 2) = uint8((double(image(i, j, 2)) - double(A(1, 2)))/max(t_hat(i, j), t0) + double(A(1,2)));
        haze_free_s(i, j, 3) = uint8((double(image(i, j, 3)) - double(A(1, 3)))/max(t_hat(i, j), t0) + double(A(1,3)));
        %fprintf("%d %d %d \n", int32(haze_free(i,j,1)), int32(haze_free(i,j,2)), int32(haze_free(i,j,3)));
    end
end


figure, imshow(haze_free_s);
%% dehazing
haze_free = zeros(rows, cols, channels, 'uint8');
t0 = 0.1;

for i = 1 : rows
    for j = 1:cols
        haze_free(i, j, 1) = uint8((double(image(i, j, 1)) - double(A(1, 1)))/max(t(i, j), t0) + double(A(1,1)));
        haze_free(i, j, 2) = uint8((double(image(i, j, 2)) - double(A(1, 2)))/max(t(i, j), t0) + double(A(1,2)));
        haze_free(i, j, 3) = uint8((double(image(i, j, 3)) - double(A(1, 3)))/max(t(i, j), t0) + double(A(1,3)));
        %fprintf("%d %d %d \n", int32(haze_free(i,j,1)), int32(haze_free(i,j,2)), int32(haze_free(i,j,3)));
    end
end


figure, imshow(image);
figure, imshow(haze_free);
