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
    string fname = name.substr(0,name.size() - 4);
    string save_loc = "./output/" + fname +  + "_" + f + ".jpg";
    bool check = imwrite(save_loc, histImage);
    namedWindow(f, CV_WINDOW_AUTOSIZE);
    imshow(f, histImage);
    waitKey(delay);
	destroyWindow(f);  
}

uint8_t* equalise(uint8_t* pixel, uint8_t* col_pixel, int n,int m, int color)
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
	uint8_t* res = new uint8_t[n*m];
	uint8_t* res_c = new uint8_t[n*m*color];
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{		
			if(color>1)
			{
				for(int k=0;k<color;k++)
				{	
					int src = (int)col_pixel[(color*i)*m+color*j+k];
					res_c[(color*i)*m+color*j+k] = mapping[src];
				}

			}	
			else
			{
				int src = (int)pixel[i*m + j];
				int dest = mapping[src];
				res[i*m + j] = dest;
			}
			
		}
	}
	if(color>1)
		return res_c;
	else
		return res;
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

void process(string file, string name)
{
	Mat image = imread(file);
	if (image.empty()) 
	{
		cout << "Invalid File Format" << endl;
		return;
	}
	string fname = name.substr(0,name.size() - 4);
	uint8_t* pixel = (uint8_t*)image.data;
	int n = image.rows;
	int m = image.cols;
	int colors = image.channels();
	cout << "Height of Image" << " " << n << endl;
	cout << "Width of Image" << " " << m << endl;
	cout << "Channels" << " " << colors << endl;
	if(colors > 1)
	{	
		
		uint8_t *res_g = convert2gray(pixel, n, m, colors);
		Mat res_c(n, m, CV_8UC3, Scalar(0,0,0));
		res_c.data = image.data;
		uint8_t* pix_col = (uint8_t*)res_c.data;

		uint8_t *histeq = equalise(res_g, pix_col, n, m, colors);

		Mat res(n, m, CV_8UC3, Scalar(0,0,0));	
		string f = "Before Equalisation";
		namedWindow(f);
		res.data = pix_col;
		imshow(f, res);
		waitKey(delay);
		destroyWindow(f);
		string save_loc = "./output/" + fname + "_before_image.jpg";
		imwrite(save_loc,res);
		plothist(res_g, n, m, f,name);
		res.data = histeq;
		string f1 = "After Equalisation";
		namedWindow(f1);
		imshow(f1, res);
		waitKey(delay);
		destroyWindow(f1);
		save_loc = "./output/" + fname + "_after_image.jpg";
		res_g = convert2gray(histeq, n, m, colors);
		imwrite(save_loc,res); 
		plothist(res_g, n, m, f1, name);
		return;
		
	}
	uint8_t *histeq = equalise(pixel, pixel, n, m, 1);
	Mat res(n, m, CV_8UC1, Scalar(0));	
	string f = "Before Equalisation";
	namedWindow(f);
	res.data = pixel;
	imshow(f, res);
	waitKey(delay);
	destroyWindow(f);
	string save_loc = "./output/" + fname + "_before_image.jpg";
	imwrite(save_loc,res);
	plothist(res.data, n, m, f, name);
	res.data = histeq;
	string f1 = "After equalisation";
	namedWindow(f1);
	imshow(f1, res);
	waitKey(delay);
	destroyWindow(f1); 
	save_loc = "./output/" + fname + "_after_image.jpg";
	imwrite(save_loc,res);
	plothist(res.data, n, m, f1, name);
	return; 

}

int ListDir(const std::string& path) {
  struct dirent *entry;
  DIR *dp;

  dp = ::opendir(path.c_str());
  if (dp == NULL) {
    perror("opendir: Path does not exist or could not be read.");
    return -1;
  }

  while ((entry = ::readdir(dp))) {
  	string file = entry->d_name;
  	if(endsWith(file,"tif") or endsWith(file,"tiff") or endsWith(file,"jpg") or endsWith(file, "bmp"))
  	{
	  	cout << "Filename" << " " << file << endl;
	    process("./images/" + file, file);
	}
	else continue;
  }

  ::closedir(dp);
  return 0;
}

int main()
{
	ListDir("./images/");
	return 0;
}
