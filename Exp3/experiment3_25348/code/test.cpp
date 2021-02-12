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



float* getKernel(string filter, int size)
{
	float* h = new float[size * size];
	int mid = (size)/2;
	if(filter == "Mean")
	{
		for(int i=0;i<size;i++)
		{
			for(int j=0;j<size;j++)
				h[i*size + j] = float(1/float(size * size));
		}
	}
	else if(filter == "Gaussian")
	{
		int sigma = 1;
		for(int i=0;i<size;i++)
		{
			for(int j=0;j<size;j++)
			{
				int x = abs(i-mid);
				int y = abs(j-mid);
				h[i*size + j] = (1/(2 * PI * pow(sigma, 2))) * exp( - (pow(x,2) + pow(y,2))/(2 * pow(sigma,2)));
			}
		}
	}
	return h;

}

void applyfilter(int fileid, int filterid, int kernel)
{
	vector<string> imgs;
	ListDir("./Noisy Images/", imgs);
	ListDir("./Normal Images/", imgs);

	vector<string> filters = {"Mean", "Gaussian", "Median", "Prewitt", "Sobel-h", "Sobel-v", "Sobel-d", "Laplacian", "LoG"};

	Mat image = imread(imgs[fileid], IMREAD_GRAYSCALE);
	float* h = getKernel(filters[filterid], kernel);
	imshow("Tracker", image);
	uint8_t* pixel = (uint8_t*)image.data;
	int n = image.rows;
	int m = image.cols;

	// for(int i=0;i<kernel;i++)
	// {
	// 	for(int j=0;j<kernel;j++)
	// 		cout << h[i*kernel + j] << " ";
	// 	cout << endl; 
	// }

	uint8_t* newimage = new uint8_t[n*m];

	int s = kernel / 2;

	cout << n << " " << m << " " << s << endl;
	
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			float sum = 0;
			if(i - s < 0 or i + s > n or j - s < 0 or j + s > m)
				continue;
			// cout << i << " " << j << endl;
			for(int p = -s; p <= s; p++)
			{
				for(int r = -s; r<=s; r++)
				{
					int posim = (i + p) * m + (j + r);
					int posh =  (p + s) * kernel + (r + s);
					// cout << i + p << " " << j + r << " " << p + 1 << " " << r + 1 << endl;  
					sum += h[posh] * (int)pixel[posim];
				}
			}
			newimage[i * m + j] = (uint8_t)floor(sum);
			// cout << i << " " << j << " " << sum << endl;
			// break;
		}
		// break;
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
     applyfilter(*u.file_id,*u.filter_id,*u.kernel_size);
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
	imshow("Tracker", image);
	waitKey();
	// createTrackbar("Filter", "Tracker", &filter, filters.size() - 1, myfunc, &u);
	// createTrackbar("Kernel Size", "Tracker", &ksize, 9, myfunc, &u);

	return 0;
}