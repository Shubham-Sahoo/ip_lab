%%
clear;
close all;

A = imread('sample1.png');

rows = size(A, 1);
cols = size(A, 2);
channels = size(A, 3);
%%
% estimation of 