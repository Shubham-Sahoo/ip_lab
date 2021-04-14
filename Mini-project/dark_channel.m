function y = dark_channel(image, patch)
    n = size(image, 1);
    m = size(image, 2);
    channels = size(image, 3);
    w = floor(patch/2);
    y = zeros(n,m);
    for i = 1:n
        for j = 1:m
            min_value = 255;
            for c = 1:channels
                for p = -w : w
                    for q = -w : w
                        r = i + p;
                        s = j + q;
                        if r < 1 || r > n || s < 1 || s > m
                            continue
                        end
                        if int32(image(r,s,c)) < min_value
                            min_value = int32(image(r,s,c));
                        end
                    end
                end
            end
%             fprintf("%d \n",min_value);
            y(i, j) = uint8(min_value);
        end
    end
end