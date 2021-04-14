function y = estimate_trate(image, patch, A)
    n = size(image, 1);
    m = size(image, 2);
    channels = size(image, 3);
    w = 0.95;
    width = floor(patch / 2);
    y = zeros(n,m);
    for i = 1:n
        for j = 1:m
            min_value = 255;
            for c = 1:channels
                for p = -width : width
                    for q = -width : width
                        r = i + p;
                        s = j + q;
                        if r < 1 || r > n || s < 1 || s > m 
                            continue
                        end
                        if double(image(r,s,c)) / A(1, c) < min_value 
                            min_value = double(image(r,s,c))/ A(1, c);
                        end
                    end
                end
            end
            y(i, j) = 1 - w * min_value;
        end
    end 
end