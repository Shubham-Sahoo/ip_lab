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
#include <math.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#define PI 3.14
using namespace std;
using namespace cv;

typedef struct {
 int* file_id;
 int* filter_id;
 int* kernel_size;
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


uint8_t* applygaussian(Mat image, int size, int type)
{
	double* h = new double[size * size];
	int mid = (size)/2;
	float normal = 0;
	int sigma = 1;
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



uint8_t* applyLoG(Mat image, int size, int type)
{
	uint8_t* pixel = applygaussian(image, size, 1);
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
}

uint8_t* applyprewitt(Mat image, int size, int type)
{
	double* h = new double[size * size];
	double* v = new double[size * size];

	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			if(j==0)
			{
				h[i*size+j] = 1;
			}
			if(i==0)
			{
				v[i*size+j] = 1;
			}
			if(j==size-1)
			{
				h[i*size+j] = -1;
			}
			if(i==size-1)
			{
				v[i*size+j] = -1;
			}
			if(i!=0 && j!=0 && i!=size-1 && j!=size-1)
			{
				h[i*size+j] = 0;
				v[i*size+j] = 0;
			}

		}	
	}


	uint8_t *imh = convolve(image, h, size, type);
	uint8_t *imv = convolve(image, v, size, type);
	double *res_p = new double[image.rows * image.cols];
	uint8_t *res = new uint8_t[image.rows * image.cols];

	for(int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			res_p[i*image.cols+j] = sqrt(pow(imh[i*image.cols+j],2)+pow(imv[i*image.cols+j],2))/sqrt(2);
			
		}
	}

	for(int i=0;i<image.rows;i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			res[i*image.cols+j] = uint8_t(res_p[i*image.cols+j]);
		}
	}

	return res;
}

uint8_t* applysobelh(Mat image, int size, int type)
{
	double* h = new double[size * size];

	h[0] = 1;	h[1] = 0;	h[2] = -1;
	h[3] = 2;	h[4] = 0;	h[5] = -2;
	h[6] = 1;	h[7] = 0;	h[8] = -1;

	return convolve(image, h, size, type);
}

uint8_t* applysobelv(Mat image, int size, int type)
{
	double* h = new double[size * size];

	h[0] = 1;	h[1] = 2;	h[2] = 1;
	h[3] = 0;	h[4] = 0;	h[5] = 0;
	h[6] = -1;	h[7] = -2;	h[8] = -1;

	return convolve(image, h, size, type);
}

uint8_t* applysobeld(Mat image, int size, int type)
{
	double* h = new double[size * size];
	double* v = new double[size * size];

	h[0] = 0;	h[1] = 1;	h[2] = 2;
	h[3] = -1;	h[4] = 0;	h[5] = 1;
	h[6] = -2;	h[7] = -1;	h[8] = 0;

	v[0] = 2;	v[1] = 1;	v[2] = 0;
	v[3] = 1;	v[4] = 0;	v[5] = -1;
	v[6] = 0;	v[7] = -1;	v[8] = -2;

	uint8_t *imh = convolve(image, h, size, type);
	uint8_t *imv = convolve(image, v, size, type);
	double *res_p = new double[image.rows * image.cols];
	uint8_t *res = new uint8_t[image.rows * image.cols];

	for(int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			res_p[i*image.cols+j] = sqrt(pow(imh[i*image.cols+j],2)+pow(imv[i*image.cols+j],2))/sqrt(2);
			
		}
	}

	for(int i=0;i<image.rows;i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			res[i*image.cols+j] = uint8_t(res_p[i*image.cols+j]);
		}
	}

	return res;
}

void applyfilter(int fileid, int filterid, int kernel, bool valid)
{	
	


	vector<string> imgs;
	ListDir("./Noisy Images/", imgs);
	ListDir("./Normal Images/", imgs);

	vector<string> filters = {"Mean", "Gaussian", "Median", "Prewitt", "Sobel-h", "Sobel-v", "Sobel-d", "Laplacian", "LoG"};

	Mat image = imread(imgs[fileid], IMREAD_GRAYSCALE);

	if(valid==0)
	{	
		
		Mat res(image.rows, image.cols, CV_8UC1, Scalar(0));
		cv::putText(res, //target image
            "Invalid kernel size!", //text
            cv::Point(10, image.cols / 2), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(255, 255, 255), //font color
            2);
		Mat result(Size(image.cols*2,image.rows),CV_8UC1,Scalar::all(0));
		Mat mat_im = result(Rect(0,0,image.cols,image.rows));
		image.copyTo(mat_im);
		mat_im = result(Rect(image.cols,0,image.cols,image.rows));
		res.copyTo(mat_im);
		imshow("Tracker", result);
		return;
	}

	//imshow("Tracker", image);
	int n = image.rows;
	int m = image.cols;
	uint8_t* newimage;
	switch(filterid){
		case 0:
			newimage = applymean(image, kernel, 0);
			break;
		case 1:
			newimage = applygaussian(image, kernel, 1);
			break;
		case 2:
			newimage = applymedian(image, kernel, 2);
			break;
		case 3:
			newimage = applyprewitt(image, kernel, 3);
			break;
		case 4:
			newimage = applysobelh(image, kernel, 4);
			break;
		case 5:
			newimage = applysobelv(image, kernel, 5);
			break;	
		case 6:
			newimage = applysobeld(image, kernel, 6);
			break;
		case 7:
			newimage = applylaplacian(image, kernel, 7);
			break;
		case 8:
			newimage = applyLoG(image, kernel, 8);
			break;
		default:
			cout << "Invalid Filter" << endl;
			return;
	}
	Mat res(n, m, CV_8UC1, Scalar(0));
	res.data = newimage;

	Mat result(Size(image.cols*2,image.rows),CV_8UC1,Scalar::all(0));
	Mat mat_im = result(Rect(0,0,image.cols,image.rows));
	image.copyTo(mat_im);
	mat_im = result(Rect(image.cols,0,image.cols,image.rows));
	res.copyTo(mat_im);
	imshow("Tracker", result);

	//namedWindow("Result");
	//imshow("Result", res);
}

void myFunc(int value, void *ud)
{
	 userdata u = *static_cast<userdata*>(ud);

     // cout << *(u.file_id) << " " << *(u.filter_id) << " " << *(u.kernel_size) << endl;
     if(*u.kernel_size % 2 == 0 or *(u.kernel_size) < 2)
     {
     	applyfilter(*u.file_id,*u.filter_id,*u.kernel_size,0);
     	return;
     }
     if((*u.filter_id >= 3 || *u.filter_id <= 6)&& *(u.kernel_size) != 3)
     {
     	applyfilter(*u.file_id,*u.filter_id,*u.kernel_size,0);
     	return;
     }
     applyfilter(*u.file_id,*u.filter_id,*u.kernel_size,1);
}


int main()
{	

	int fname = 0;
	int filter_id = 0;
	int ksize = 0;
	userdata u;
	u.file_id = &fname;
	u.filter_id = &filter_id;
	u.kernel_size = &ksize;
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
	Mat image = imread(imgs[id], IMREAD_GRAYSCALE);
	Mat res_im( image.cols,image.rows, CV_8UC1, Scalar(255));
	cv::putText(image, //target image
        "Move the track bars for output!", //text
        cv::Point(10, 600 / 2), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        1.0,
        CV_RGB(255, 255, 255), //font color
        2);
	
	Mat res(Size(image.cols*2,image.rows),CV_8UC1,Scalar::all(0));
	Mat mat_im = res(Rect(0,0,image.cols,image.rows));
	image.copyTo(mat_im);
	mat_im = res(Rect(image.cols,0,image.cols,image.rows));
	res_im.copyTo(mat_im);
	imshow("Tracker", res);
	waitKey();
	// createTrackbar("Filter", "Tracker", &filter, filters.size() - 1, myfunc, &u);
	// createTrackbar("Kernel Size", "Tracker", &ksize, 9, myfunc, &u);

	return 0;
}