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
 int* cutoff_size;
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

	int max_v = 0;
	int min_v = 0;
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


			if(type != 7 && type!=8)
			{
				if(sum<0)
				{
					sum=0;
				}
				if(sum>255)
				{
					sum=255;
				}
				newimage[i*m + j] = (uint8_t)floor(sum);
				if(type == 8)
					newimage[i*m + j] = (uint8_t)(255 - (int)newimage[i*m + j]);
			}
			else
			{
				if(sum<0)
					sum = -sum;
				if(sum>255)
					sum = 255;
				newimage[i*m + j] = (uint8_t)floor(sum);
				// cout << (int)newimage[i*m + j] << " ";
			}
		}
	}

	min_v = (min_v<0)?min_v:0;
	max_v = (max_v>255)?max_v:255;

	if(type==7)
	{
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < m; j++)
			{
				newimage[i * m + j] = (uint8_t)floor(double(newimage[i * m + j]*255)/(max_v-min_v));
			}
		}
	}
	return newimage;
}

uint8_t* fft2()
{

}

uint8_t* ifft2()
{
	
}

class ComplexFloat {
public:
	double real;
	double img;

public:
	ComplexFloat()
	{
		this->real = 0;
		this->img = 0;
	}
	ComplexFloat(double real, double img)
	{
		this->real = real;
		this->img = img;
	}
	ComplexFloat operator+(const ComplexFloat& b)
	{
		double r = real + b.real;
		double i = img + b.img;
		return ComplexFloat(r, i);
	}
	ComplexFloat operator-(const ComplexFloat& b)
	{
		double r = real - b.real;
		double i = img - b.img;
		return ComplexFloat(r, i);
	}
	ComplexFloat operator*(const ComplexFloat& b)
	{
		double k1 = b.real*(real + img);
		double k2 = real*(b.img - b.real);
		double k3 = img*(b.img + b.real);
		return ComplexFloat(k1 - k3, k1 + k2);
	}

	ComplexFloat operator*(const double& b)
	{
		return ComplexFloat(real*b, img*b);
	}

	void operator*=(const double& b)
	{
		real *= b;
		img *= b;
	}

	ComplexFloat operator/(const double& b)
	{
		return ComplexFloat(real / b, img / b);
	}

	void operator=(const double& b)
	{
		real = b;
		img = 0;
	}

	double magnitude()
	{
		return sqrt((this->real)*(this->real) + (this->img)*(this->img));
	}
	void print() {
		cout << real << " + " << img << "i";
	}

};

template<typename T>
void Transpose(T** matrix, int N)
{
	T temp;
	for (int i = 0; i < N; i++) {
		T* start = matrix[i] + i;
		for (int j = i + 1; j < N; j++) {
			temp = matrix[i][j];
			matrix[i][j] = matrix[j][i];
			matrix[j][i] = temp;
		}
	}
}

template<typename T> T** FFTShift(T** matrix, int N)
{
	T temp;
	int offset = N / 2;
	for (int i = 0; i < offset; i++) {
		T* start = matrix[i] + i;
		for (int j = 0; j < offset; j++) {
			temp = matrix[i][j];
			matrix[i][j] = matrix[i + offset][j + offset];
			matrix[i + offset][j + offset] = temp;
		}
	}

	for (int i = N / 2; i < N; i++) {
		T* start = matrix[i] + i;
		for (int j = 0; j < offset; j++) {
			temp = matrix[i][j];
			matrix[i][j] = matrix[i - offset][j + offset];
			matrix[i - offset][j + offset] = temp;
		}
	}
	return matrix;
}

Mat FFTShift(Mat matrix, int N)
{
	float temp;
	int offset = N / 2;
	for (int i = 0; i < offset; i++) {
		for (int j = 0; j < offset; j++) {
			temp = matrix.at<float>(i, j);
			matrix.at<float>(i, j) = matrix.at<float>(i + offset, j + offset);
			matrix.at<float>(i + offset, j + offset) = temp;
		}
	}

	for (int i = N / 2; i < N; i++) {
		for (int j = 0; j < offset; j++) {
			temp = matrix.at<float>(i, j);
			matrix.at<float>(i, j) = matrix.at<float>(i - offset, j + offset);
			matrix.at<float>(i - offset, j + offset) = temp;
		}
	}
	return matrix;
}

//ASSUMPTIONS
//WHEN CALLING THIS FUNCTION
//arrSize = N
//gap = 1
//zeroLoc = 0

ComplexFloat* FFT(uchar* x, int N, int arrSize, int zeroLoc, int gap)
{
	ComplexFloat* fft;
	fft = new ComplexFloat[N];

	int i;
	if (N == 2)
	{
		fft[0] = ComplexFloat(x[zeroLoc] + x[zeroLoc + gap], 0);
		fft[1] = ComplexFloat(x[zeroLoc] - x[zeroLoc + gap], 0);
	}
	else
	{
		ComplexFloat wN = ComplexFloat(cos(2 * M_PI / N), sin(-2 * M_PI / N));//exp(-j2*pi/N)
		ComplexFloat w = ComplexFloat(1, 0);
		gap *= 2;
		ComplexFloat* X_even = FFT(x, N / 2, arrSize, zeroLoc, gap); //N/2 POINT DFT OF EVEN X's
		ComplexFloat* X_odd = FFT(x, N / 2, arrSize, zeroLoc + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		ComplexFloat todd;
		for (i = 0; i < N / 2; ++i)
		{
			//FFT(0) IS EQUAL TO FFT(N-1) SYMMETRICAL AROUND N/2
			todd = w*X_odd[i];
			fft[i] = X_even[i] + todd;
			fft[i + N / 2] = X_even[i] - todd;
			w = w * wN;
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return fft;
}
ComplexFloat* FFT(ComplexFloat* x, int N, int arrSize, int zeroLoc, int gap)
{
	ComplexFloat* fft;
	fft = new ComplexFloat[N];

	int i;
	if (N == 2)
	{
		fft[0] = x[zeroLoc] + x[zeroLoc + gap];
		fft[1] = x[zeroLoc] - x[zeroLoc + gap];
	}
	else
	{
		ComplexFloat wN = ComplexFloat(cos(2 * M_PI / N), sin(-2 * M_PI / N));//exp(-j2*pi/N)
		ComplexFloat w = ComplexFloat(1, 0);
		gap *= 2;
		ComplexFloat* X_even = FFT(x, N / 2, arrSize, zeroLoc, gap); //N/2 POINT DFT OF EVEN X's
		ComplexFloat* X_odd = FFT(x, N / 2, arrSize, zeroLoc + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		ComplexFloat todd;
		for (i = 0; i < N / 2; ++i)
		{
			//FFT(0) IS EQUAL TO FFT(N-1) SYMMETRICAL AROUND N/2
			todd = w*X_odd[i];
			fft[i] = X_even[i] + todd;
			fft[i + N / 2] = X_even[i] - todd;
			w = w * wN;
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return fft;
}
ComplexFloat* IFFT(ComplexFloat* fft, int N, int arrSize, int zeroLoc, int gap)
{
	ComplexFloat* signal;
	signal = new ComplexFloat[N];

	int i;
	if (N == 2)
	{
		signal[0] = fft[zeroLoc] + fft[zeroLoc + gap];
		signal[1] = fft[zeroLoc] - fft[zeroLoc + gap];
	}
	else
	{
		ComplexFloat wN = ComplexFloat(cos(2 * M_PI / N), sin(2 * M_PI / N));//exp(j2*pi/N)
		ComplexFloat w = ComplexFloat(1, 0);
		gap *= 2;
		ComplexFloat* X_even = IFFT(fft, N / 2, arrSize, zeroLoc, gap); //N/2 POINT DFT OF EVEN X's
		ComplexFloat* X_odd = IFFT(fft, N / 2, arrSize, zeroLoc + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		ComplexFloat todd;
		for (i = 0; i < N / 2; ++i)
		{
			//FFT(0) IS EQUAL TO FFT(N-1) SYMMETRICAL AROUND N/2
			todd = w * X_odd[i];
			signal[i] = (X_even[i] + todd) * 0.5;
			signal[i + N / 2] = (X_even[i] - todd) * 0.5;
			w = w * wN; // Get the next root(conjugate) among Nth roots of unity
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return signal;
}

ComplexFloat** FFT2(Mat& source) {
	//cout << "Applying FFT2" << endl;

	if (source.rows != source.cols) {
		cout << "Image is not Valid";
		return nullptr;
	}
	int N = source.rows;
	//cout << "Image size:" << N << endl;
	ComplexFloat** FFT2Result_h;
	FFT2Result_h = new ComplexFloat*[N];

	// ROW WISE FFT
	for (int i = 0; i < N; i++) {
		uchar* row = source.ptr<uchar>(i);
		FFT2Result_h[i] = FFT(row, N, N, 0, 1);
	}

	//cout << "final: " << endl;
	Transpose<ComplexFloat>(FFT2Result_h, N);

	// COLUMN WISE FFT
	for (int i = 0; i < N; i++) {
		FFT2Result_h[i] = FFT(FFT2Result_h[i], N, N, 0, 1);
	}
	Transpose<ComplexFloat>(FFT2Result_h, N);

	return FFT2Result_h;
}

ComplexFloat** IFFT2(ComplexFloat** source, int N) 
{

	//cout << "Applying IFFT2" << endl;

	ComplexFloat** ifftResult;
	ifftResult = new ComplexFloat*[N];
	// ROW WISE FFT
	for (int i = 0; i < N; i++) {
		ifftResult[i] = IFFT(source[i], N, N, 0, 1);
	}

	//cout << "final: " << endl;
	Transpose<ComplexFloat>(ifftResult, N);

	int d = N*N;
	// COLUMN WISE FFT
	for (int i = 0; i < N; i++) {
		ifftResult[i] = IFFT(ifftResult[i], N, N, 0, 1);
		for (int j = 0; j < N; j++) {
			ifftResult[i][j] = ifftResult[i][j] / d;
		}
	}
	Transpose<ComplexFloat>(ifftResult, N);

	return ifftResult;
}

void Complex2Mat(ComplexFloat** source, Mat& dest, int N, bool shift = false, float maxF = 1.0) {
	// Convert a complex matrix to a Mat data structure (magnitude of 
	// the complex no. are used) for showing as an image
	if (shift) {
		FFTShift(source, N);
	}
	dest = Mat(N, N, CV_32F, cv::Scalar::all(0));
	float min = 99999;
	float max = 0;

	// Find min and max
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			source[i][j] = source[i][j] / N;
			float m = source[i][j].magnitude();
			if (m < min) {
				min = m;
			}
			if (m > max) {
				max = m;
			}
		}
	}


	// Normalize the image
	float range = (max - min);
	for (int i = 0; i < N; i++) {
		float *p = dest.ptr<float>(i);
		for (int j = 0; j < N; j++) {
			p[j] = (source[i][j].magnitude() - min) * maxF / range;
		}
	}
	//cout << "Min: " << min << " Max:" << max;
}

ComplexFloat **applyideal_low(ComplexFloat **fft_im, Mat dft, int n, int m, int cutoff, int type)
{
	Mat image(n, m, CV_32F, Scalar(0));
	Mat dft_shift = FFTShift(dft,n);
	ComplexFloat **fft_im_shift = fft_im;//FFTShift<ComplexFloat>(fft_im,n);
	uint8_t *fil = new uint8_t[n*m];
	cutoff = cutoff*sqrt(n);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			//float dist1 = sqrt((i-(float(n)/2))*(i-(float(n)/2)) + (j-(float(m)/2))*(j-(float(m)/2)));
			float dist1 = sqrt((i-(float(0)))*(i-(float(0))) + (j-(float(0)))*(j-(float(0))));
			float dist2 = sqrt((i-(float(0)))*(i-(float(0))) + (j-(float(m)))*(j-(float(m))));
			float dist3 = sqrt((i-(float(n)))*(i-(float(n))) + (j-(float(0)))*(j-(float(0))));
			float dist4 = sqrt((i-(float(n)))*(i-(float(n))) + (j-(float(m)))*(j-(float(m))));
			if((dist1<=float(cutoff))||(dist2<=float(cutoff))||(dist3<=float(cutoff))||(dist4<=float(cutoff)))
			{
				fil[i*n+j] = 1;
				fft_im_shift[i][j] *= 1;
				//cout<<i<<" "<<j<<"\n";
			}
			else
			{
				fil[i*n+j] *= 0;
				fft_im_shift[i][j] *= 0;
				// cout<<fft_im_shift[]
			}
		}
	}
	float max_val = 0;
	float min_val = 100000;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			image.at<float>(i,j) = dft_shift.at<float>(i,j)*float(fil[i*n+j]);
			//image.at<float>(i,j) = (fft_im_shift[i][j])*float(fil[i*n+j]);
			//cout<<image.at<float>(i,j)<<" ";
			if(image.at<float>(i,j)>max_val)
			{
				max_val = image.at<float>(i,j);
			}
			if(image.at<float>(i,j)<min_val)
			{
				min_val = image.at<float>(i,j);
			}
		}
	}

	return fft_im_shift;

}

ComplexFloat **applyideal_high(ComplexFloat **fft_im, Mat dft, int n, int m, int cutoff, int type)
{
	Mat image(n, m, CV_32F, Scalar(0));
	Mat dft_shift = FFTShift(dft,n);
	ComplexFloat **fft_im_shift = fft_im;//FFTShift<ComplexFloat>(fft_im,n);
	uint8_t *fil = new uint8_t[n*m];
	cutoff = cutoff*sqrt(n);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			//float dist1 = sqrt((i-(float(n)/2))*(i-(float(n)/2)) + (j-(float(m)/2))*(j-(float(m)/2)));
			float dist1 = sqrt((i-(float(0)))*(i-(float(0))) + (j-(float(0)))*(j-(float(0))));
			float dist2 = sqrt((i-(float(0)))*(i-(float(0))) + (j-(float(m)))*(j-(float(m))));
			float dist3 = sqrt((i-(float(n)))*(i-(float(n))) + (j-(float(0)))*(j-(float(0))));
			float dist4 = sqrt((i-(float(n)))*(i-(float(n))) + (j-(float(m)))*(j-(float(m))));
			if((dist1<=float(cutoff))||(dist2<=float(cutoff))||(dist3<=float(cutoff))||(dist4<=float(cutoff)))
			{
				fil[i*n+j] = 0;
				fft_im_shift[i][j] *= 0;
				//cout<<i<<" "<<j<<"\n";
			}
			else
			{
				fil[i*n+j] *= 1;
				fft_im_shift[i][j] *= 1;
				// cout<<fft_im_shift[]
			}
		}
	}
	float max_val = 0;
	float min_val = 100000;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			image.at<float>(i,j) = dft_shift.at<float>(i,j)*float(fil[i*n+j]);
			//image.at<float>(i,j) = (fft_im_shift[i][j])*float(fil[i*n+j]);
			//cout<<image.at<float>(i,j)<<" ";
			if(image.at<float>(i,j)>max_val)
			{
				max_val = image.at<float>(i,j);
			}
			if(image.at<float>(i,j)<min_val)
			{
				min_val = image.at<float>(i,j);
			}
		}
	}

	return fft_im_shift;

}

ComplexFloat **applygaussian_low(ComplexFloat **fft_im, Mat dft, int n, int m, int cutoff, int type)
{
	Mat image(n, m, CV_32F, Scalar(0));
	Mat dft_shift = FFTShift(dft,n);
	ComplexFloat **fft_im_shift = fft_im;//FFTShift<ComplexFloat>(fft_im,n);
	uint8_t *fil = new uint8_t[n*m];
	cutoff = cutoff*sqrt(n);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			//float dist1 = sqrt((i-(float(n)/2))*(i-(float(n)/2)) + (j-(float(m)/2))*(j-(float(m)/2)));
			float dist1 = sqrt((i-(float(0)))*(i-(float(0))) + (j-(float(0)))*(j-(float(0))));
			float dist2 = sqrt((i-(float(0)))*(i-(float(0))) + (j-(float(m)))*(j-(float(m))));
			float dist3 = sqrt((i-(float(n)))*(i-(float(n))) + (j-(float(0)))*(j-(float(0))));
			float dist4 = sqrt((i-(float(n)))*(i-(float(n))) + (j-(float(m)))*(j-(float(m))));
			if((dist1<=float(cutoff))||(dist2<=float(cutoff))||(dist3<=float(cutoff))||(dist4<=float(cutoff)))
			{
				fil[i*n+j] = 0;
				fft_im_shift[i][j] *= 1;
				//cout<<i<<" "<<j<<"\n";
			}
			else
			{
				fil[i*n+j] *= 0;
				fft_im_shift[i][j] *= 0;
				// cout<<fft_im_shift[]
			}
		}
	}
	float max_val = 0;
	float min_val = 100000;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			image.at<float>(i,j) = dft_shift.at<float>(i,j)*float(fil[i*n+j]);
			//image.at<float>(i,j) = (fft_im_shift[i][j])*float(fil[i*n+j]);
			//cout<<image.at<float>(i,j)<<" ";
			if(image.at<float>(i,j)>max_val)
			{
				max_val = image.at<float>(i,j);
			}
			if(image.at<float>(i,j)<min_val)
			{
				min_val = image.at<float>(i,j);
			}
		}
	}

	return fft_im_shift;

}



// uint8_t* applybutter_low(Mat image, int size, int type)
// {


// 	double* v = new double[size * size];
// 	double sigma = (double(size)/6);

// 	int kernelSize = size;
// 	for(int i = -(kernelSize/2); i<=(kernelSize/2); i++)
//     {

//         for(int j = -(kernelSize/2); j<=(kernelSize/2); j++)
//         {

//             double L_xy = -(1/(PI * pow(sigma,4)))*(1 - ((pow(i,2) + pow(j,2))/(2*pow(sigma,2))))*exp(-((pow(i,2) + pow(j,2))/(2*pow(sigma,2))));
//             //L_xy*=426.3;
//             v[(i + kernelSize/2)*size  + (j + kernelSize/2)] = L_xy;
//         }

//     }
// 	float sum = 0;
// 	for(int i=0;i<size;i++)
// 	{
// 		for(int j=0;j<size;j++)
// 		{
// 			sum += (float)v[i*size + j];
// 			//cout << (float)v[i*size + j] << " ";
// 		}
// 		//cout << endl;
// 	}
	
// 	return convolve(image, v, size, type);	
// }

// uint8_t* applybutter_high(Mat image, int size, int type)
// {
// 	double* h = new double[size * size];
// 	double* v = new double[size * size];

// 	for(int i=0;i<size;i++)
// 	{
// 		for(int j=0;j<size;j++)
// 		{
// 			if(j==0)
// 			{
// 				h[i*size+j] = 1;
// 			}
// 			if(i==0)
// 			{
// 				v[i*size+j] = 1;
// 			}
// 			if(j==size-1)
// 			{
// 				h[i*size+j] = -1;
// 			}
// 			if(i==size-1)
// 			{
// 				v[i*size+j] = -1;
// 			}
// 			if(i!=0 && j!=0 && i!=size-1 && j!=size-1)
// 			{
// 				h[i*size+j] = 0;
// 				v[i*size+j] = 0;
// 			}

// 		}	
// 	}


// 	uint8_t *imh = convolve(image, h, size, type);
// 	uint8_t *imv = convolve(image, v, size, type);
// 	double *res_p = new double[image.rows * image.cols];
// 	uint8_t *res = new uint8_t[image.rows * image.cols];

// 	for(int i = 0; i < image.rows; i++)
// 	{
// 		for (int j = 0; j < image.cols; j++)
// 		{
// 			res_p[i*image.cols+j] = sqrt(pow(imh[i*image.cols+j],2)+pow(imv[i*image.cols+j],2))/sqrt(2);
			
// 		}
// 	}

// 	for(int i=0;i<image.rows;i++)
// 	{
// 		for (int j = 0; j < image.cols; j++)
// 		{
// 			res[i*image.cols+j] = uint8_t(res_p[i*image.cols+j]);
// 		}
// 	}

// 	return res;
// }


void applyfilter(int fileid, int filterid, int cutoff, bool valid)
{	
	vector<string> imgs;
	ListDir("./", imgs);
	ListDir("./", imgs);

	vector<string> filters = {"Ideal-LPF", "Ideal-HPF", "Gaussian-LPF", "Gaussian-HPF", "Buuterworth-LPF", "Buuterworth-HPF"};

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

	ComplexFloat **dft = FFT2(image);
	Mat fft_res(n, m, CV_32F, Scalar(0));
	
	Complex2Mat(dft,fft_res,n,false,255);
	ComplexFloat** newimage = new ComplexFloat*[n];
	for(int i=0;i<n;i++)
	{
		newimage[i] = new ComplexFloat[m];
	}
	switch(filterid){
		case 0:
			newimage = applyideal_low(dft, fft_res, n, m, cutoff, 0);
			break;
		case 1:
			newimage = applyideal_high(dft, fft_res, n, m, cutoff, 1);
			break;
		case 2:
			newimage = applygaussian_low(dft, fft_res, n, m, cutoff, 2);
			break;

		// case 2:
		// 	newimage = applygaus_low(image, kernel, 2);
		// 	break;
		// case 3:
		// 	newimage = applygaus_high(image, kernel, 3);
		// 	break;
		// case 4:
		// 	newimage = applybutter_low(image, kernel, 4);
		// 	break;
		// case 5:
		// 	newimage = applybutter_high(image, kernel, 5);
		// 	break;	
		default:
			cout << "Invalid Filter" << endl;
			return;
	}


	Mat filtered_fft(n,m,CV_8UC1,Scalar(0));
	ComplexFloat **shift_new = FFTShift(newimage,n);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			filtered_fft.at<uint8_t>(i,j) = uint8_t(((shift_new[i][j]).magnitude())*255);
		}
	}
	

	ComplexFloat **result_ifft;
	result_ifft = IFFT2(newimage,n);
	ComplexFloat **result_c = result_ifft;

	Mat result_f(n, m, CV_32F, Scalar(0));
	Complex2Mat(result_c,result_f,n,false,255);

	Mat res(n,m,CV_8UC1,Scalar(0));
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			res.at<uint8_t>(i,j) = uint8_t((result_f.at<float>(i,j)));
		}
	}

	Mat result(Size((image.cols)*3,image.rows),CV_8UC1,Scalar::all(0));
	Mat mat_im = result(Rect(0,0,image.cols,image.rows));
	image.copyTo(mat_im);
	mat_im = result(Rect(image.cols,0,image.cols,image.rows));
	filtered_fft.copyTo(mat_im);
	mat_im = result(Rect(2*image.cols,0,image.cols,image.rows));
	res.copyTo(mat_im);
	imshow("Tracker", result);

}

void myFunc(int value, void *ud)
{
	userdata u = *static_cast<userdata*>(ud);

    //cout << *(u.file_id) << " " << *(u.filter_id) << " " << *(u.cutoff_size) << endl;
    applyfilter(*u.file_id,*u.filter_id,*u.cutoff_size,1);
}


int main()
{	

	int fname = 0;
	int filter_id = 0;
	int ksize = 0;
	userdata u;
	u.file_id = &fname;
	u.filter_id = &filter_id;
	u.cutoff_size = &ksize;
	int id;
	cout << "Enter any ID" << endl;
	cin >> id;
	namedWindow("Tracker", 1);
	vector<string> imgs;
	ListDir("./", imgs);
	ListDir("./", imgs);
	vector<string> filters = {"Ideal-LPF", "Ideal-HPF", "Gaussian-LPF", "Gaussian-HPF", "Buuterworth-LPF", "Buuterworth-HPF"};
	createTrackbar("File-ID", "Tracker", u.file_id, imgs.size() - 1, myFunc, &u);
	createTrackbar("Filter-ID", "Tracker", u.filter_id, filters.size() - 1, myFunc, &u);
	createTrackbar("Kernel_size", "Tracker", u.cutoff_size, 50, myFunc, &u);
	Mat image = imread(imgs[id], IMREAD_GRAYSCALE);
	Mat res_im( image.cols,image.rows, CV_8UC1, Scalar(255));
	cv::putText(res_im, //target image
        "Move the track bars for output!", //text
        cv::Point(10, 600 / 2), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        0.75,
        CV_RGB(0, 0, 0), //font color
        2);
	
	Mat res(Size(image.cols*3,image.rows),CV_8UC1,Scalar::all(0));
	Mat mat_im = res(Rect(0,0,image.cols,image.rows));
	image.copyTo(mat_im);
	mat_im = res(Rect(image.cols,0,image.cols,image.rows));
	res_im.copyTo(mat_im);
	mat_im = res(Rect(2*image.cols,0,image.cols,image.rows));
	res_im.copyTo(mat_im);
	imshow("Tracker", res);
	waitKey();


	return 0;
}