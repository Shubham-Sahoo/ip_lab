#include <iostream>
#include <cstdio>
#include <math.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#define PI 3.14
using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("lena_gray_512.jpg", IMREAD_GRAYSCALE);
	int n = img.rows;
	int m = img.cols;
	cout << n << " " << m << endl;
}
