function y = dark_channel(image, patch)
    n = size(image, 1);
    m = size(image, 2);
    channels = size(image, 3);
    w = patch/2;
    y = zeros(n,m);
    for i = 1:n
        for j = 1:m
            min_value = 255;
            for c = 1:channels
                for p = -w : w
                    for q = -w:w
                        r = i + p;
                        s = j + q;
                        if r < 0 || r > n || s < 0 || s > m
                            continue
                        end
                        if image(r,s,c) < min_value
                            min_value = image(r,s,c);
                        end
                    end
                end
            end
            y(i, j) = min_value;
        end
    end
end