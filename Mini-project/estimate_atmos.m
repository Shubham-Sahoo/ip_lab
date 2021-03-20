function y = estimate_atmos(image, dark)

%size of image = [n,m,3];
%size of dark = [n,m];
% get top 0.1% elements in dark and identify the pixels in the input image
% and take those average as the atmospheric light

n = size(dark, 1);
m = size(dark, 2);
numel = 0.1 * 0.01 * n * m;

dark1 = reshape(dark, [n*m, 1]);
image1 = reshape(image, [n*m, 3]);
[~, k] = maxk(dark1, numel);

y = zeros(1,3);
s = size(k,1);
count = 0;
for i = 1:s
    y = y + image1(k(i));
    count = count + 1;
end
y = y / count;
end