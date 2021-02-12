#include <iostream>
#include <cstdio>

#include <dirent.h>
#include <opencv2/opencv.hpp>

#define delay 1000
using namespace std;
using namespace cv;

/*
Tasks to be done:
1. Reading an image - Done
2. Obtaining histogram of the image. Like how is it defined.. depending on number of levels in the image - Done
3. Create an array for each level and record the CDF. - Done
4. Transformed histogram hist = (L - 1) * CDF - Done
5. Map the levels in the image to new levels - Done
6. Display the original histogram, new histogram and so o

*/

void plothist(uint8_t* pixels, int n, int m, string f, string name)
{
	vector<float>hist(256,0);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			int val = pixels[i*m + j];
			hist[val]++;
		}
	}
	int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double) hist_w/256);
    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));
    float histmax = -1;
    for(int i = 0;i<256;i++)
    	histmax = max(histmax, hist[i]);
    for(int i=0;i<256;i++)
    {
    	hist[i] = ((double)hist[i]/histmax)*histImage.rows;
    	// cout << hist[i] << " ";
    }
    // cout << endl;
    // cout << histmax << endl;
    line(histImage, Point(0, hist_h - 30), Point(hist_w, hist_h - 30), Scalar(0, 0, 0), 2, 8, 0);
    line(histImage, Point(0, hist_h - 20), Point(0, hist_h - 40), Scalar(0, 0, 0), 2, 8, 0);
    putText(histImage, "0", Point(0, hist_h-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 1, cv::LINE_AA);

    for(int i = 1; i < 256; i++)
    {
        // line(histImage, Point(bin_w*(i), hist_h), Point(bin_w*(i), hist_h - hist[i]),Scalar(0,0,0), 1, 8, 0);
        line(histImage, Point((i-1)*bin_w, hist_h - 30 - hist[i-1]), 
            Point(i*bin_w, hist_h - 30 - hist[i]), Scalar(0, 0, 255), 2, 8, 0);
        if (i % 20 == 0){
            string text = to_string(i);
            line(histImage, Point(i*bin_w, hist_h - 20), Point(i*bin_w, hist_h - 40), Scalar(0, 0, 0), 2, 8, 0);
            putText(histImage, text, Point(i*bin_w, hist_h-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 1, cv::LINE_AA);
        }
    }
    string save_loc = "./output_2/" + name + "_ " + f + ".jpg";
    imwrite(save_loc, histImage);
    namedWindow("Histogram", CV_WINDOW_AUTOSIZE);
    imshow("Histogram", histImage);
    waitKey(delay);
	destroyWindow("Histogram");  
}

map<int,int> cal_hist(uint8_t* pixel, int n,int m)
{
	vector<int> v(256,0);
	// PDF calculation
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			// cout << (int)pixel[i*m + j] << endl;
			v[(int)pixel[i*m + j]]++;
		}
	}
	// CDF
	map<int,int> mapping;
	for(int i=1;i<v.size();i++)
	{
		v[i] += v[i-1];
	}
	for(int i=0;i<v.size();i++)
	{
		mapping[i] = floor((float(v[i])/(n*m)) * 255);
	}
	// for(auto j=mapping.begin();j!=mapping.end();j++)
	// 	cout << j->first << " " << j->second << endl;
	
	return mapping;
}

uint8_t *hist_match( map<int,int> histeq1, map<int,int> histeq2, uint8_t *pixel, int n, int m, int color)
{
	int min;
	int ind;
	for(int i=0;i<256;i++)
	{	
		min = abs(histeq1[i]-histeq2[0]);
		for(int j=1;j<256;j++)
		{
			if(abs(histeq1[i]-histeq2[j])<min)
			{
				min = abs(histeq1[i]-histeq2[j]);
				ind = j;
			}
		}
		histeq1[i] = ind;
	}

	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{	
			if(color>1)
			{
				for(int k=0;k<color;k++)
				{
					pixel[(color*i)*m+color*j+k] = histeq1[pixel[(color*i)*m+color*j+k]];
				}
			}
			else
			{
				pixel[i*m+j] = histeq1[pixel[i*m+j]];	
			}
			
		}
	}
	return pixel;
}

uint8_t *convert2gray(uint8_t* img, int n,int m,int colors)
{
	uint8_t* res = new uint8_t[n*m];
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			res[i*m + j] = (floor(0.3*img[(3*i)*m+3*j+2]+0.59*img[(3*i)*m+3*j+1]+0.11*img[(3*i)*m+3*j]));
		}
	}
	return res;
}

bool endsWith(std::string str, std::string suffix)
{
   return str.find(suffix, str.size() - suffix.size()) != string::npos;
}

void process(string ifile, string tfile, string source, string target)
{
	Mat image1 = imread(ifile);
	Mat image2 = imread(tfile);
	if (image1.empty() | image2.empty()) 
	{
		cout << "Invalid File Format" << endl;
		return;
	}
	uint8_t* pixel1 = (uint8_t*)image1.data;
	int n1 = image1.rows;
	int m1 = image1.cols;

	uint8_t* pixel2 = (uint8_t*)image2.data;
	int n2 = image2.rows;
	int m2 = image2.cols;

	int colors1 = image1.channels();
	int colors2 = image2.channels();

	cout << "Height of Input Image" << " " << n1 << endl;
	cout << "Width of Input Image" << " " << m1 << endl;
	cout << "Channels of Input Image" << " " << colors1 << endl;

	cout << "Height of Target Image" << " " << n2 << endl;
	cout << "Width of Target Image" << " " << m2 << endl;
	cout << "Channels of Target Image" << " " << colors2 << endl;

	map<int,int> histeq1;
	map<int,int> histeq2;

	if(colors1 > 1)
	{
		uint8_t *res_g = convert2gray(pixel1, n1, m1, colors1);
		histeq1 = cal_hist(res_g, n1, m1);
	}
	else
	{
		uint8_t *res_g = convert2gray(pixel1, n1, m1, colors1);
		histeq1 = cal_hist(res_g, n1, m1);
	}
	if(colors2 > 1)
	{
		uint8_t *res_g  = convert2gray(pixel2, n2, m2, colors2);
		histeq2 = cal_hist(res_g, n2, m2);
	}
	else
	{
		uint8_t *res_g = convert2gray(pixel2, n2, m2, colors2);
		histeq2 = cal_hist(res_g, n2, m2);
	}

	
	


	Mat res1(n1, m1, CV_8UC1, Scalar(0));
	Mat res2(n2, m2, CV_8UC1, Scalar(0));
	
	Mat res4(n1, m1, CV_8UC3, Scalar(0,0,0));

	Mat res3(n1, m1, CV_8UC1, Scalar(0));

	if(colors1>1)
	{
		Mat res1(n1, m1, CV_8UC3, Scalar(0,0,0));
		string f1 = "Before Matching input";
		namedWindow(f1);
		res1.data = pixel1;
		imshow(f1, res1);
		waitKey(delay);
		destroyWindow(f1);
		string save_loc = "./output_2/" + source + "_source.jpg";
		imwrite(save_loc,res1);
		uint8_t *res_g = convert2gray(res1.data, n1, m1, colors1);
		plothist(res_g, n1, m1, "source_before", source);

	}
	else
	{
		string f1 = "Before Matching input";
		namedWindow(f1);
		res1.data = pixel1;
		imshow(f1, res1);
		waitKey(delay);
		destroyWindow(f1);
		string save_loc = "./output_2/" + source + "_source.jpg";
		imwrite(save_loc,res1);
		plothist(res1.data, n1, m1, "source_before", source);
	}
	
	if(colors2>1)
	{	
		Mat res2(n2, m2, CV_8UC3, Scalar(0,0,0));
		string f2 = "Before Matching target";
		namedWindow(f2);
		res2.data = pixel2;
		imshow(f2, res2);
		waitKey(delay);
		destroyWindow(f2);
		string save_loc = "./output_2/" + target + "_target.jpg";
		imwrite(save_loc,res2);
		uint8_t *res_g = convert2gray(res2.data, n2, m2, colors2);
		plothist(res2.data, n2, m2, "target_before", target);
	}

	else
	{
		string f2 = "Before Matching target";
		namedWindow(f2);
		res2.data = pixel2;
		imshow(f2, res2);
		waitKey(delay);
		destroyWindow(f2);
		string save_loc = "./output_2/" + target + "_target.jpg";
		imwrite(save_loc,res2);

		plothist(res2.data, n2, m2, "target_before", target);
	}
	

	uint8_t *hist_ch =  hist_match(histeq1,histeq2,pixel1,n1,m1,colors1);
	if(colors1>1)
		res4.data = hist_ch;
	else
		res3.data = hist_ch;
	string f3 = "After Matching";
	namedWindow(f3);
	if(colors1>1)
	{
		imshow(f3, res4);
		waitKey(delay);
		destroyWindow(f3);
		string save_loc = "./output_2/" + source + "_matched.jpg";
		imwrite(save_loc,res4);
		uint8_t *res_g = convert2gray(res4.data, n1, m1, colors1);
		plothist(res_g, n1, m1, "source_matched", source);
	}
	else
	{
		imshow(f3, res3);
		waitKey(delay);
		destroyWindow(f3);
		string save_loc = "./output_2/" + source + "_matched.jpg";
		imwrite(save_loc,res3);
		plothist(res3.data, n1, m1, "source_matched", source);
	}
	 
	
}

int ListDir(const std::string& path, string input, string target) {
  struct dirent *entry;
  DIR *dp;

  dp = ::opendir(path.c_str());
  if (dp == NULL) {
    perror("opendir: Path does not exist or could not be read.");
    return -1;
  }

	process("./images/" + input, "./images/" + target, input, target);

  ::closedir(dp);
  return 0;
}

int main()
{	
	string input,target;
	cout<<"Enter input image \n";
	getline(cin,input);
	cout<<" "<<flush;
	cout<<"Enter Target Image \n";
	getline(cin,target); 

	ListDir("./images/",input,target);
	return 0;
}
