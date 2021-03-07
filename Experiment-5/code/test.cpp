#include<iostream>
#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include<vector>
#include<math.h>
#include <cstdio>
#include <math.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef unsigned int uint32_t;
typedef unsigned short int uint16_t;
typedef unsigned char uint8_t;
typedef int int32_t;
#define PI 3.14159265

typedef struct {
 int* file_id;
 int* st_id;
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


void erosion(uint8_t *imarray, Mat st)
{
	uint8_t ***rdata;

	rdata = new uint8_t**[height];
	for(int i=0;i<height;i++)
	{
		rdata[i] = new uint8_t*[width];
		for(int j=0;j<width;j++)
		{
			rdata[i][j] = new uint8_t[colors];
		}
	}
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			for(int k=0;k<colors;k++)
				rdata[i][j][k] = 255;
		}
	}

	rdata = convert1Dto3D(imarray,width, height, colors, rdata);
	
	int ox = height/2, oy = width/2;

	uint8_t ***newdata;
	newdata = new uint8_t**[height];
	for(int i=0;i<height;i++)
	{
		newdata[i] = new uint8_t*[width];
		for(int j=0;j<width;j++)
		{
			newdata[i][j] = new uint8_t[colors];
		}
	}
	
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			for(int k=0;k<colors;k++)
				newdata[i][j][k] = 255;
		}
	}
			
	
	uint8_t *res;
	res = new uint8_t[n*(m*colors)];

	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{	
			for(int k=0;k<colors;k++)
			{
				res[i*m*colors + j*colors +k] = newdata[i][j][k];
			}
		}
	}

}

void applyfilter(int fileid, int st_id, int type)
{	
	vector<string> imgs;
	ListDir("./", imgs);
	ListDir("./", imgs);

	Mat st_1(1, 2, CV_8UC1, Scalar(0));
	st_1.at<uint8_t>(0,0) = 1;
	st_1.at<uint8_t>(0,1) = 1;

	Mat st_2(3, 3, CV_8UC1, Scalar(1));
	Mat st_3(3, 3, CV_8UC1, Scalar(1));
	st_3.at<uint8_t>(0,0) = 0;
	st_3.at<uint8_t>(0,2) = 0;
	st_3.at<uint8_t>(2,0) = 0;
	st_3.at<uint8_t>(2,2) = 0;

	Mat st_4(9, 9, CV_8UC1, Scalar(1));
	Mat st_5(15, 15, CV_8UC1, Scalar(1));

	vector< Mat > structures;
	structures.push_back(st_1);
	structures.push_back(st_2);
	structures.push_back(st_3);
	structures.push_back(st_4);
	structures.push_back(st_5);
	//cout << cutoff << " ";
	Mat image = imread(imgs[fileid], IMREAD_GRAYSCALE);

	// cout << cutoff << " ";
	int n = image.rows;
	int m = image.cols;

	Mat newimage(n,m,CV_8UC1,Scalar(0));
	switch(filterid)
	{
		case 0:
			newimage = erosion(image,structures[st_id]);
			break;
		case 1:
			newimage = dilation(image,structures[st_id]);
			break;
		case 2:
			newimage = opening(image,structures[st_id]);
			break;
		case 3:
			newimage = closing(image,structures[st_id]);
			break;
		default:
			cout << "Invalid Filter" << endl;
			return;
	}

	//cout<<max_val<<" "<<min_val<<"\n";

	Mat res(n,m,CV_8UC1,Scalar(0));
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			res.at<uint8_t>(i,j) = 1;
		}
	}

	Mat st_res(n,m,CV_8UC1,Scalar(0));
	for(int i=0;i<structures[st_id].rows;i++)
	{
		for(int j=0;j<structures[st_id].cols;j++)
		{
			st_res.at<uint8_t>(i+n/2,j+m/2) = structures[st_id].at<uint8_t>(i,j)*255;
		}
	}

	Mat result(Size((image.cols)*3,image.rows),CV_8UC1,Scalar::all(0));
	Mat mat_im = result(Rect(0,0,image.cols,image.rows));
	image.copyTo(mat_im);
	mat_im = result(Rect(image.cols,0,image.cols,image.rows));
	res.copyTo(mat_im);
	mat_im = result(Rect(2*image.cols,0,image.cols,image.rows));
	st_res.copyTo(mat_im);

	cv::resize(result,result, cv::Size(), 0.75, 0.75);

	imshow("Tracker", result);

}

void myFunc(int value, void *ud)
{
	userdata u = *static_cast<userdata*>(ud);
	//cout<<*(u.type)<<" "<<flush;
    applyfilter(*u.file_id,*u.st_id,*(u.type));
}



int main()
{
	// string file;
	// cout << "Enter File Name" << endl;
	// cin >> file; 
	// string filename = file+".bmp";
	// int ind;
	// cout << "Enter the structuring element number : "<<endl;
	// cin >> ind;
	// Mat image = imread(filename, IMREAD_GRAYSCALE);

	int fname = 0;
	int st_id = 0;
	int type_val = 0;
	userdata u;
	u.file_id = &fname;
	u.st_id = &st_id;
	u.type = &type_val;

	namedWindow("Tracker", 1);
	vector<string> imgs;
	ListDir("./", imgs);
	ListDir("./", imgs);
	vector<string> structures = {"0","1","2","3","4"};
	vector<string> types = {"Erosion","Dilation","Opening","Closing"};

	createTrackbar("File-ID", "Tracker", u.file_id, imgs.size() - 1, myFunc, &u);
	createTrackbar("Filter-ID", "Tracker", u.st_id, structures.size() - 1, myFunc, &u);
	createTrackbar("Type", "Tracker", u.type, types.size(), myFunc, &u);

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
	//obj.erosion(file,imarray);

	
}	