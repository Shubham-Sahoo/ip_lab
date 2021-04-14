## Things to do


1. Reading the paper
2. Doubt/ concept clarification session
3. Implementation pipeline


## Implementation

1. Given the maximum height and width of samples are 500
2. Patch size used for dark channel computation is 15 X 15
3. t'(x) can be estimated using the equation 11
4. Using soft matting, we obtain t(x) from t'(x).
5. The dark channel of a hazy image approximates the haze denseness
6. Pick-up the top 0.1% brightest pixels in the dark channel, and pick the brightest ones among these
7. Finally J(x) is obtained using the equation 16, where the value of t0 = 0.1 