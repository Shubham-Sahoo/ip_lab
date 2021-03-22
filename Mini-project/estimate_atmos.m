function y = estimate_atmos(image, dark)

%size of image = [n,m,3];
%size of dark = [n,m];
% get top 0.1% elements in dark and identify the pixels in the input image
% and take those average as the atmospheric light

n = size(dark, 1);
m = size(dark, 2);
numel = floor(0.1 * 0.01 * n * m);

dark1 = reshape(dark, [n*m, 1]);
image1 = reshape(image, [n*m, 3]);
[~, k] = maxk(dark1, numel);

y = zeros(1,3);
s = size(k,1);
count = 0;
for i = 1:s
%     fprintf("%d %d %d \n", image1(k(i, 1), 1), image1(k(i, 1), 2), image1(k(i, 1), 3));
    y(1, 1) = y(1, 1) + int32(image1(k(i,1), 1));
    y(1, 2) = y(1, 2) + int32(image1(k(i,1), 2));
    y(1, 3) = y(1, 3) + int32(image1(k(i,1), 3));
    count = count + 1;
end
y (1, 1) = uint8(floor(y(1,1)/count));
y (1, 2) = uint8(floor(y(1,2)/count));
y (1, 3) = uint8(floor(y(1,3)/count));
end