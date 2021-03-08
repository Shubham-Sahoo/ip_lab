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


Mat erosion(Mat image, Mat st)
{

	int n = image.rows;
	int m = image.cols;

	Mat res(n,m,CV_8UC1,Scalar(0));

	int st_r = (st.rows)/2;
	int st_c = (st.cols)/2;
	int en_r = (st.rows)/2;
	int en_c = (st.cols)/2;
	if(st.rows == 1)
	{
		st_r = 0;	
	}
	

	for(int i=st_r;i<n-en_r;i++)
	{
		for(int j=st_c;j<m-en_c;j++)
		{
			int flag = 0;
			for(int k=0;k<st.rows;k++)
			{
				for(int l=0;l<st.cols;l++)
				{	
					int ref_i = i-st_r+k;
					int ref_j = j-st_c+l;
					if(image.at<uint8_t>(ref_i,ref_j) == 0 && st.at<uint8_t>(k,l)==1)
					{
						flag = 1;
						break;
					}
					// if(image.at<uint8_t>(ref_i,ref_j) == 255 && st.at<uint8_t>(k,l)==0)
					// {
					// 	flag = 1;
					// 	break;
					// }
				}
				if(flag==1)
				{
					break;
				}
			}
			if(flag==0)
			{
				res.at<uint8_t>(i,j) = 255;
			}
		}
	}
	return res;
}

Mat dilation(Mat image, Mat st)
{

	int n = image.rows;
	int m = image.cols;

	Mat res(n,m,CV_8UC1,Scalar(0));

	int st_r = (st.rows)/2;
	int st_c = (st.cols)/2;
	int en_r = (st.rows)/2;
	int en_c = (st.cols)/2;
	if(st.rows == 1)
	{
		st_r = 0;	
	}
	

	for(int i=st_r;i<n-en_r;i++)
	{
		for(int j=st_c;j<m-en_c;j++)
		{
			for(int k=0;k<st.rows;k++)
			{
				for(int l=0;l<st.cols;l++)
				{	
					int ref_i = i-st_r+k;
					int ref_j = j-st_c+l;
					if(st.at<uint8_t>(k,l)==1 && image.at<uint8_t>(i,j)==255)
					{
						res.at<uint8_t>(ref_i,ref_j) = 255;
					}
					
				}
				
			}
		}
	}
	return res;
}

Mat opening(Mat image, Mat st)
{	
	int n = image.rows;
	int m = image.cols;

	Mat res_er(n,m,CV_8UC1,Scalar(0));
	res_er = erosion(image,st);
	Mat res(n,m,CV_8UC1,Scalar(0));
	res = dilation(res_er,st);
	
	return res;
}

Mat closing(Mat image, Mat st)
{
	int n = image.rows;
	int m = image.cols;

	Mat res_dl(n,m,CV_8UC1,Scalar(0));
	res_dl = dilation(image,st);
	Mat res(n,m,CV_8UC1,Scalar(0));
	res = erosion(res_dl,st);
	
	return res;
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
	switch(type)
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
	newimage.copyTo(mat_im);
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
	createTrackbar("Structure-ID", "Tracker", u.st_id, structures.size() - 1, myFunc, &u);
	createTrackbar("Filter-Type", "Tracker", u.type, types.size() - 1, myFunc, &u);

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