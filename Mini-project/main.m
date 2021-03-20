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
%%