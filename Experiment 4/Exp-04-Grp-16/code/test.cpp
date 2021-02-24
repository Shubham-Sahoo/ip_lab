/*
Task: Frequency domain Filtering

Things to do:

1. Design frequency domain filters using C++
2. Implement sliders to select the image, filter, cutoff size
3. Apply the required algorithm according to outputs from the slider
4. Display the input image and the output image.



################ Slider Desgin ########################

1. To select the image:
2. To select the filter
3. To select the cutoff size

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
 int* type;
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
	//ComplexFloat **fft_im_shift = fft_im;//FFTShift<ComplexFloat>(fft_im,n);
	ComplexFloat **fft_im_shift = FFTShift<ComplexFloat>(fft_im,n);
	uint8_t *fil = new uint8_t[n*m];
	cutoff = cutoff*sqrt(n);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			float dist = sqrt((i-(float(n)/2))*(i-(float(n)/2)) + (j-(float(m)/2))*(j-(float(m)/2)));
			if((dist<=float(cutoff)))
			{
				fil[i*n+j] = 1;
				fft_im_shift[i][j] *= 1;
				//cout<<i<<" "<<j<<"\n";
			}
			else
			{
				fil[i*n+j] *= 0;
				fft_im_shift[i][j] *= 0;
			}
		}
	}
	
	fft_im_shift = FFTShift<ComplexFloat>(fft_im_shift,n);
	return fft_im_shift;

}

ComplexFloat **applyideal_high(ComplexFloat **fft_im, Mat dft, int n, int m, int cutoff, int type)
{
	Mat image(n, m, CV_32F, Scalar(0));
	Mat dft_shift = FFTShift(dft,n);
	ComplexFloat **fft_im_shift = FFTShift<ComplexFloat>(fft_im,n);
	uint8_t *fil = new uint8_t[n*m];
	cutoff = cutoff*sqrt(n);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			float dist = sqrt((i-(float(n)/2))*(i-(float(n)/2)) + (j-(float(m)/2))*(j-(float(m)/2)));
			
			if((dist<float(cutoff)))
			{
				fil[i*n+j] = 0;
				fft_im_shift[i][j] *= 0;
				//cout<<i<<" "<<j<<"\n";
			}
			else
			{
				fil[i*n+j] = 1;
				fft_im_shift[i][j] *= 1;
				// cout<<fft_im_shift[]
			}
		}
	}
	
	fft_im_shift = FFTShift<ComplexFloat>(fft_im_shift,n);
	return fft_im_shift;

}

ComplexFloat **applygaussian_low(ComplexFloat **fft_im, Mat dft, int n, int m, int cutoff, int type)
{
	Mat image(n, m, CV_32F, Scalar(0));
	Mat dft_shift = FFTShift(dft,n);
	ComplexFloat **fft_im_shift = FFTShift<ComplexFloat>(fft_im,n);
	uint8_t *fil = new uint8_t[n*m];
	cutoff = cutoff*sqrt(n)/6;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			float dist = (i-(float(n)/2))*(i-(float(n)/2)) + (j-(float(m)/2))*(j-(float(m)/2));
			float weight = exp(- ((dist) / (2 * cutoff * cutoff)));
			fft_im_shift[i][j] *= weight;
			fil[i*n + j] = weight;
			
		}
	}
	
	fft_im_shift = FFTShift<ComplexFloat>(fft_im_shift,n);
	return fft_im_shift;

}

ComplexFloat **applygaussian_high(ComplexFloat **fft_im, Mat dft, int n, int m, int cutoff, int type)
{
	Mat image(n, m, CV_32F, Scalar(0));
	Mat dft_shift = FFTShift(dft,n);
	ComplexFloat **fft_im_shift = FFTShift<ComplexFloat>(fft_im,n);
	uint8_t *fil = new uint8_t[n*m];
	cutoff = cutoff*sqrt(n)/6;
	// cout << cutoff << " ";
	for(int i=0;i<n;i++)
	{	float weight;
		float dist;
		for(int j=0;j<m;j++)
		{
			dist = (i-(float(n)/2))*(i-(float(n)/2)) + (j-(float(m)/2))*(j-(float(m)/2));
			weight = 1 - exp(-(dist/(2*float(cutoff)*float(cutoff))));    //(1/(sqrt(2*3.14)*float(cutoff)))
			fft_im_shift[i][j] *= weight;
			fil[i*n + j] = weight;
			
		}
		//cout<<weight<<" "<<dist<<" "<<flush;
	}
	
	fft_im_shift = FFTShift<ComplexFloat>(fft_im_shift,n);
	return fft_im_shift;

}

ComplexFloat **applybutter_low(ComplexFloat **fft_im, Mat dft, int n, int m, int cutoff, int type)
{
	Mat image(n, m, CV_32F, Scalar(0));
	Mat dft_shift = FFTShift(dft,n);
	ComplexFloat **fft_im_shift = FFTShift<ComplexFloat>(fft_im,n);
	uint8_t *fil = new uint8_t[n*m];
	cutoff = cutoff*sqrt(n);
	int order = 5;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			float dist = sqrt((i-(float(n)/2))*(i-(float(n)/2)) + (j-(float(m)/2))*(j-(float(m)/2)));
			float weight = 1/(1 + pow((dist/cutoff), 2 * order));
			fft_im_shift[i][j] *= weight;
			fil[i*n + j] = weight;
			
		}
	}
	
	fft_im_shift = FFTShift<ComplexFloat>(fft_im_shift,n);
	return fft_im_shift;

}

ComplexFloat **applybutter_high(ComplexFloat **fft_im, Mat dft, int n, int m, int cutoff, int type)
{
	Mat image(n, m, CV_32F, Scalar(0));
	Mat dft_shift = FFTShift(dft,n);
	ComplexFloat **fft_im_shift = FFTShift<ComplexFloat>(fft_im,n);
	uint8_t *fil = new uint8_t[n*m];
	cutoff = cutoff*sqrt(n);
	int order = 5;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			float dist = sqrt((i-(float(n)/2))*(i-(float(n)/2)) + (j-(float(m)/2))*(j-(float(m)/2)));
			float weight = 1/(1 + pow((cutoff/dist), 2 * order));
			fft_im_shift[i][j] *= weight;
			fil[i*n + j] = weight;
			
		}
	}
	
	fft_im_shift = FFTShift<ComplexFloat>(fft_im_shift,n);
	return fft_im_shift;

}



void applyfilter(int fileid, int filterid, int cutoff, bool valid)
{	
	vector<string> imgs;
	ListDir("./", imgs);
	ListDir("./", imgs);

	vector<string> filters = {"Ideal-LPF", "Ideal-HPF", "Gaussian-LPF", "Gaussian-HPF", "Buuterworth-LPF", "Buuterworth-HPF"};
	//cout << cutoff << " ";
	Mat image = imread(imgs[fileid], IMREAD_GRAYSCALE);

	// cout << cutoff << " ";
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
		case 3:
			newimage = applygaussian_high(dft, fft_res, n, m, cutoff, 3);
			break;
		case 4:
			newimage = applybutter_low(dft, fft_res, n, m, cutoff, 4);
			break;
		case 5:
			newimage = applybutter_high(dft, fft_res, n, m, cutoff, 5);
			break;	
		default:
			cout << "Invalid Filter" << endl;
			return;
	}


	Mat filtered_fft(n,m,CV_8UC1,Scalar(0));
	ComplexFloat **shift_new = FFTShift(newimage,n);

	float min_val = 1000000,max_val=0,sum_val=0;
	for (int i = 0; i < n; i++) 
	{
		for (int j = 0; j < m; j++) 
		{
			float m = shift_new[i][j].magnitude();
			if (m < min_val) {
				min_val = m;
			}
			if (m > max_val) {
				max_val = m;
			}
			sum_val += m;
		}
	}
	//cout<<max_val<<" "<<min_val<<"\n";

	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			filtered_fft.at<uint8_t>(i,j) = uint8_t(((shift_new[i][j]).magnitude())*float(255)/(n));
		}
	}
	
	if(valid==0)
	{
		for(int i=0;i<n;i++)
		{
			for(int j=0;j<m;j++)
			{
				filtered_fft.at<uint8_t>(i,j) = uint8_t(((shift_new[i][j]).magnitude())*float(255)/(1));
			}
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

	cv::resize(result,result, cv::Size(), 0.75, 0.75);
	imshow("Tracker", result);

}

void myFunc(int value, void *ud)
{
	userdata u = *static_cast<userdata*>(ud);
	//cout<<*(u.type)<<" "<<flush;
    applyfilter(*u.file_id,*u.filter_id,*u.cutoff_size,*(u.type));
}


int main()
{	

	int fname = 0;
	int filter_id = 0;
	int ksize = 0;
	int type_val = 0;
	userdata u;
	u.file_id = &fname;
	u.filter_id = &filter_id;
	u.cutoff_size = &ksize;
	u.type = &type_val;
	int id;
	int type;
	cout << "Enter type (separation{0} or variation{1}) :" << endl;
	cin >> type;
	namedWindow("Tracker", 1);
	vector<string> imgs;
	ListDir("./", imgs);
	ListDir("./", imgs);
	vector<string> filters = {"Ideal-LPF", "Ideal-HPF", "Gaussian-LPF", "Gaussian-HPF", "Buuterworth-LPF", "Buuterworth-HPF"};
	createTrackbar("File-ID", "Tracker", u.file_id, imgs.size() - 1, myFunc, &u);
	createTrackbar("Filter-ID", "Tracker", u.filter_id, filters.size() - 1, myFunc, &u);
	createTrackbar("Cutoff_size", "Tracker", u.cutoff_size, 20, myFunc, &u);
	createTrackbar("Type", "Tracker", u.type, 1, myFunc, &u);

	Mat image = imread(imgs[0], IMREAD_GRAYSCALE);
	Mat res_im( image.cols,image.rows, CV_8UC1, Scalar(255));
	cv::putText(res_im, 
        "Move the track bars for output!", 
        cv::Point(10, 600 / 2), 
        cv::FONT_HERSHEY_DUPLEX,
        0.75,
        CV_RGB(0, 0, 0),
        2);
	
	Mat res(Size(image.cols*3,image.rows),CV_8UC1,Scalar::all(0));
	Mat mat_im = res(Rect(0,0,image.cols,image.rows));
	image.copyTo(mat_im);
	mat_im = res(Rect(image.cols,0,image.cols,image.rows));
	res_im.copyTo(mat_im);
	mat_im = res(Rect(2*image.cols,0,image.cols,image.rows));
	res_im.copyTo(mat_im);

	cv::resize(res,res, cv::Size(), 0.75, 0.75);

	imshow("Tracker", res);
	waitKey();


	return 0;
}