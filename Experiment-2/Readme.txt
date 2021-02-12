##Histogram Equilisation and on Images

###Steps to run the code

1) Open the Experiment Folder

2) Make two folders "output" and "output_2"

3) Compile the equilisation code

g++ code/test.cpp `pkg-config --cflags --libs opencv` -o hist

4) Compile the histogram matching code

g++ code/test_mat.cpp `pkg-config --cflags --libs opencv` -o hist_match

5) Run the Executable

./hist or ./hist_match

6) Give Input Image name and target Image name in case of histogram matching code 

7) Output images are generated in the corresponding Output Folders

8) Histogram equalisation is performed for all images in 'images' folder. Each result is displayed for 1 sec and saved automatically in output folder.

9) output_2 folder contains the results for histogram matching
