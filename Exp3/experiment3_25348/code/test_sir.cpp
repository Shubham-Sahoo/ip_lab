/*
Task: Spatial Filtering

Things to do:

1. Design spatial filters using C++
2. Implement sliders to select the image, filter, neighborhood size
3. Apply the required algorithm according to outputs from the slider
4. Display the input image and the output image.



################ Slider Desgin ########################

1. To select the image:
2. To select the filter
3. To select the kernel size

so.. we can use a struct containing all these variables.
*/

#include <iostream>
#include <cstdio>

#include <dirent.h>
#include <opencv2/opencv.hpp>
#define PI 3.14
using namespace std;
using namespace cv;

typedef struct {
 int* file_id;
 int* filter_id;
 int* kernel_size;
 int* sigma;
}userdata;

bool endsWith(std::string str, std::string suffix)
{
   return str.find(suffix, str.size() - suffix.size()) != string::npos;
}

int ListDir(const std::string& path, vector<string>& v) {
  struct dirent *entry;
  DIR *dp;

  string root = path.c_str();

  dp = ::opendir(path.c_str());
  if (dp == NULL) {
    perror("opendir: Path does not exist or could not be read.");
    return -1;
  }

  while ((entry = ::readdir(dp))) {
  	string file = entry->d_name;
  	if(endsWith(file,"tif") or endsWith(file,"tiff") or endsWith(file,"jpg") or endsWith(file, "bmp"))
  	{
	  	v.push_back(root + file);
  	}
  }
  ::closedir(dp);
  return 0;
}

uint8_t* convolve(Mat image, double* h, int size, int type)
{
	uint8_t* pixel = (uint8_t*)image.data;
	int n = image.rows;
	int m = image.cols;

	uint8_t* newimage = new uint8_t[n*m];
	int s = size / 2;

	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			float sum = 0;
			if(i - s < 0 or i + s >= n or j - s < 0 or j + s >= m)
				continue;
			for(int p = -s; p <= s; p++)
			{
				for(int r = -s; r<=s; r++)
				{
					int posim = (i + p) * m + (j + r);
					int posh =  (p + s) * size + (r + s);
					sum += h[posh] * (int)pixel[posim];
				}
			}
			if(type == 7){
				if(sum >  255){
					sum = 255;
				}
				else if(sum < -255){
					sum = -255;
				}
				if(sum < 0)
					sum = -sum;
			}
			else{
				if(sum >  255)
					sum = 255;
				else if(sum < 0)
					sum = 0;
			}
			newimage[i * m + j] = (uint8_t)floor(sum);
		}
	}
	return newimage;
}

uint8_t* applymean(Mat image, int size, int type)
{
	double* h = new double[size * size];
	int mid = (size)/2;
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
			h[i*size + j] = float(1/float(size * size));
	}
	return convolve(image, h, size, type);
}

uint8_t* applymedian(Mat image, int size, int type)
{
	uint8_t* pixel = (uint8_t*)image.data;
	int n = image.rows;
	int m = image.cols;

	uint8_t* newimage = new uint8_t[n*m];
	int s = size / 2;
	
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			float sum = 0;
			if(i - s < 0 or i + s >= n or j - s < 0 or j + s >= m)
				continue;
			vector<int> v;
			for(int p = -s; p <= s; p++)
			{
				for(int r = -s; r<=s; r++)
				{
					int posim = (i + p) * m + (j + r);
					v.push_back((int)pixel[posim]);
				}
			}
			sort(v.begin(), v.end());
			float mid;
			if(v.size() % 2)
				mid = v[v.size()/2];
			else
				mid = (v[v.size()/2] + v[v.size()/2 - 1])/2.0;
			newimage[i * m + j] = (uint8_t)floor(mid);
		}
	}
	return newimage;
}


uint8_t* applygaussian(Mat image, int size, int type, int sigma)
{
	double* h = new double[size * size];
	int mid = (size)/2;
	float normal = 0;
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			int x = abs(i-mid);
			int y = abs(j-mid);
			float q = 2 * PI * pow(sigma, 2);
			h[i*size + j] = (1/q) * exp( - (pow(x,2) + pow(y,2)) / q);
			normal += h[i*size + j];
		}
	}
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			h[i*size + j] /= normal;
		}
	}
	return convolve(image, h, size, type);
}

uint8_t* applylaplacian(Mat image, int size, int type)
{
	double* h = new double[size * size];
	int mid = (size)/2;
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
			h[i*size + j] = 1;
	}
	h[mid*size + mid] = -(pow(size, 2) - 1);

	return convolve(image, h, size, type);
}



uint8_t* applyLoG(Mat image, int size, int type, int sigma)
{
	uint8_t* pixel = applygaussian(image, size, 1, sigma);
	int n = image.rows;
	int m = image.cols;
	Mat res(n, m, CV_8UC1, Scalar(0));
	res.data = pixel;
	return applylaplacian(res, size, 7);

	// double* h = new double[size * size];
	// float sigma = 1.0;
	// // for(int i=0;i<size;i++)
	// // {
	// // 	for(int j=0;j<size;j++)
	// // 	{
	// // 		int x = abs(i-mid);
	// // 		int y = abs(j-mid);
	// // 		float q = 2 * PI * pow(sigma, 2);
	// // 		int p = pow(x, 2) + pow(y, 2);
	// // 		h[i*size + j] = (float)(-(1/(PI * pow(sigma, 4))) * (1 - p/q) * exp(-p/q));
	// // 		h[i*size + j] *= 426.3;
	// // 		cout << -(1/(PI * pow(sigma, 4))) * (1 - p/q) * exp(-p/q) << " " << i << " " << j << " " << p << endl;
	// // 	}
	// // }
	// int kernelSize = size;
	// for(int i = -(kernelSize/2); i<=(kernelSize/2); i++)
 //    {

 //        for(int j = -(kernelSize/2); j<=(kernelSize/2); j++)
 //        {

 //            double L_xy = -1/(PI * pow(sigma,4))*(1 - ((pow(i,2) + pow(j,2))/(2*pow(sigma,2))))*exp(-((pow(i,2) + pow(j,2))/(2*pow(sigma,2))));
 //            L_xy*=426.3;
 //            h[(i + kernelSize/2)*size  + (j + kernelSize/2)] = L_xy;
 //        }

 //    }
	// float sum = 0;
	// for(int i=0;i<size;i++)
	// {
	// 	for(int j=0;j<size;j++)
	// 	{
	// 		sum += (float)h[i*size + j];
	// 		cout << (float)h[i*size + j] << " ";
	// 	}
	// 	cout << endl;
	// }
	// cout << "Filter Sum" << " " << sum << endl;
	// return convolve(image, h, size);	

	/*

	 {	{-1, -1, -1},
							{-1, 8, -1},
							{-1, -1, -1}};

	 {	{-1, 3, -4, -3, -1},
					{-3, 0,  6,  0, -3},
					{-4, 6, 20,  6, -4},
					{-3, 0,  6,  0, -3},
					{-1,-3, -4, -3, -1}};

	{	{-2, -3, -4, -6, -4, -3, -2},
					{-3, -5, -4, -3, -4, -5, -3},
					{-4, -4,  9, 20,  9, -4, -4},
					{-6, -3, 20, 36, 20, -3, -6},
					{-4, -4,  9, 20,  9, -4, -4},
					{-3, -5, -4, -3, -4, -5, -3},
					{-2, -3, -4, -6, -4, -3, -2}};
	{{0	1	1	2	2	2	1	1	0
				1	2	4	5	5	5	4	2	1
				1	4	5	3	0	3	5	4	1
				2	5	3	-12	-24	-12	3	2	1
				2	5	0	-24	-40	-24	0	5	2
				2	5	3	-12	-24	-12	3	2	1
				1	4	5	3	0	3	5	4	1
				1	2	4	5	5	5	4	2	1
				0	1	1	2	2	2	1	1	0}}
	*/
}

void applyfilter(int fileid, int filterid, int kernel, int sigma)
{
	vector<string> imgs;
	ListDir("./Noisy Images/", imgs);
	ListDir("./Normal Images/", imgs);

	vector<string> filters = {"Mean", "Gaussian", "Median", "Prewitt", "Sobel-h", "Sobel-v", "Sobel-d", "Laplacian", "LoG"};

	cout << imgs[fileid] << endl;
	Mat image = imread(imgs[fileid], IMREAD_GRAYSCALE);
	imshow("Tracker", image);
	int n = image.rows;
	int m = image.cols;
	uint8_t* newimage;
	switch(filterid){
		case 0:
			newimage = applymean(image, kernel, 0);
			break;
		case 1:
			newimage = applygaussian(image, kernel, 1, sigma);
			break;
		case 2:
			newimage = applymedian(image, kernel, 2);
			break;
		case 7:
			newimage = applylaplacian(image, kernel, 7);
			break;
		case 8:
			newimage = applyLoG(image, kernel, 8, sigma);
			break;
		default:
			cout << "Invalid Filter" << endl;
			return;
	}
	Mat res(n, m, CV_8UC1, Scalar(0));
	res.data = newimage;
	namedWindow("Result");
	imshow("Result", res);
}

void myFunc(int value, void *ud)
{
	 userdata u = *static_cast<userdata*>(ud);

     // cout << *(u.file_id) << " " << *(u.filter_id) << " " << *(u.kernel_size) << endl;
     if(*u.kernel_size % 2 == 0 or *(u.kernel_size) < 2)
     {
     	cout << "Invalid Kernel size" << endl;
     	return;
     }
     applyfilter(*u.file_id,*u.filter_id,*u.kernel_size,*u.sigma);
}


int main()
{	

	int fname = 0;
	int filter_id = 0;
	int ksize = 0;
	int sigma = 1;
	userdata u;
	u.file_id = &fname;
	u.filter_id = &filter_id;
	u.kernel_size = &ksize;
	u.sigma = &sigma;
	int id;
	cout << "Enter any ID" << endl;
	cin >> id;
	namedWindow("Tracker", 1);
	vector<string> imgs;
	ListDir("./Noisy Images/", imgs);
	ListDir("./Normal Images/", imgs);
	vector<string> filters = {"Mean", "Gaussian", "Median", "Prewitt", "Sobel-h", "Sobel-v", "Sobel-d", "Laplacian", "LoG"};
	createTrackbar("File-ID", "Tracker", u.file_id, imgs.size() - 1, myFunc, &u);
	createTrackbar("Filter-ID", "Tracker", u.filter_id, filters.size() - 1, myFunc, &u);
	createTrackbar("Kernel_size", "Tracker", u.kernel_size, 10, myFunc, &u);
	createTrackbar("Sigma", "Tracker", u.sigma, 5, myFunc, &u);
	Mat image = imread(imgs[id], IMREAD_GRAYSCALE);
	imshow("Tracker", image);
	waitKey();
	// createTrackbar("Filter", "Tracker", &filter, filters.size() - 1, myfunc, &u);
	// createTrackbar("Kernel Size", "Tracker", &ksize, 9, myfunc, &u);

	return 0;
}