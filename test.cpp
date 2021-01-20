#include<iostream>
#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include<vector>
#include<math.h>
using namespace std;

typedef unsigned int uint32_t;
typedef unsigned short int uint16_t;
typedef unsigned char uint8_t;
typedef int int32_t;

#pragma pack(push, 1)

typedef struct BITMAPIMAGE
{
	uint16_t signature;
	uint32_t fsize;
	uint32_t reserve;
	uint32_t offset;

	uint32_t header_size;
	uint32_t width;
	uint32_t height;
	uint16_t planes;
	uint16_t bitwidth;
	uint32_t compression;
	uint32_t imsize;
	uint32_t Xresol;
	uint32_t Yresol;
	uint32_t Ucolors;
	uint32_t Icolors;	
}BITMAP;

#pragma pack(pop)

BITMAP readbmp(char* s,vector<uint8_t>& data)
{
	int i;
	FILE* f = fopen(s, "rb");
	FILE* ifile = fopen(s, "rb");
	BITMAP A;
	if(f==NULL)
	{
		cout << "Error in Opening File" << endl;
		return A;
	}
	unsigned char fileheader[54];
    fread(fileheader, sizeof(unsigned char), 54, f);
    string filestruct = "";
  	for(uint8_t i=0;i<sizeof(fileheader);i++)
  	{
  		if(i<2)
  			filestruct += fileheader[i];
  		else break;
  	}
  	if(filestruct != "BM")
  	{
  		throw std::runtime_error("Incorrect Format");
  		return A;
  	}
  	int width, height, filesize, bitwidth, offset, reserve, headsize, planes, compression, imsize, Xresol, Yresol, Ucolors, Icolors;
  	filesize = *(int *)&fileheader[2];
  	reserve = *(int *)&fileheader[6];
  	offset = *(int *)&fileheader[10];
  	headsize = *(int *)&fileheader[14];
  	width = *(int *)&fileheader[18];
  	height = *(int *)&fileheader[22];
  	planes = *(uint16_t *)&fileheader[26];
  	bitwidth = *(uint16_t *)&fileheader[28];
  	compression = *(int *)&fileheader[30];
  	imsize = *(int *)&fileheader[34];
  	Xresol = *(int *)&fileheader[38];
  	Yresol = *(int *)&fileheader[42];
  	Ucolors = *(int *)&fileheader[46];
  	Icolors = *(int *)&fileheader[50];

  	cout << "Width of Image" << " " << width << endl;
  	cout << "Height of Image" << " " << height << endl;
  	cout << "Bit-width of Image" << " " << bitwidth << endl;
  	cout << "File-Size in Bytes" << " " << filesize << endl;
  	cout << "Offset size" << " " << offset << endl;

 	A.width = width;
 	A.height = height;
 	A.fsize = filesize;
 	A.bitwidth = bitwidth;
 	A.offset = offset;
 	A.header_size = headsize;
 	A.planes = planes;
 	A.compression = compression;
 	A.imsize = imsize;
 	A.Xresol = Xresol;
 	A.Yresol = Yresol;
 	A.Ucolors = Ucolors;
 	A.Icolors = Icolors;

 	int colors = bitwidth/8;
	int unpadded_row_size = width * colors;
	int padded_row_size = (int)(4 * ceil((float)(width/4.0f))*colors);
	int totalsize = unpadded_row_size * height;
	data.resize(totalsize);
	i = 0;
	auto currentpointer = data.data() +  (height - 1)*unpadded_row_size;
	for(int i=0;i<height;i++)
	{
		fseek(ifile, offset + (i * padded_row_size), SEEK_SET);
		fread(currentpointer, sizeof(unsigned char), unpadded_row_size, ifile);
		currentpointer -= unpadded_row_size;
	}
	cout << "Printing Image Array" << endl;
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
			cout << (int)data[i*width + j] << " ";
		cout << endl;
	}
	fclose(ifile);
	fclose(f);
	return A;
}


void writebmp(char* out, BITMAP A, vector<uint8_t> data)
{
	FILE* outfile = fopen(out, "wb");
	char *format = "BM";
	int width = A.width;
	int height = A.height;
	uint16_t bitwidth = A.bitwidth;
	int colors = A.bitwidth/8;
	int offset = A.offset;
	fwrite(&format[0], 1, 1, outfile);
	fwrite(&format[1], 1, 1, outfile);
	int padded_row_size = (int )(4 * ceil((float)(width)/4.0f))*colors;
	uint32_t filesize = padded_row_size * height + 54;
	fwrite(&filesize, 4, 1, outfile);
	int reserve = 0;
	fwrite(&reserve, 4, 1, outfile);
	int dataoffset = 54;
	fwrite(&dataoffset, 4, 1, outfile);

	int headsize = 40;
	uint16_t planes = 1;
	int compression = A.compression;
	int Xresol = A.Xresol; //300 dpi
    int Yresol = A.Yresol; //300 dpi
	int Ucolors = A.Ucolors;
	int Icolors = A.Icolors;
	int imsize = width * height * colors;
	
	fwrite(&headsize, 4, 1, outfile);
	fwrite(&width, 4, 1, outfile);
	fwrite(&height, 4, 1, outfile);
	fwrite(&planes, 2, 1, outfile);
	fwrite(&bitwidth, 2, 1, outfile);
	fwrite(&compression, 4, 1, outfile);
	fwrite(&imsize, 4, 1, outfile);
	fwrite(&Xresol, 4, 1, outfile);
	fwrite(&Yresol, 4, 1, outfile);
	fwrite(&Ucolors, 4, 1, outfile);
	fwrite(&Icolors, 4, 1, outfile);
	int i = 0;
	int unpadded_row_size = width*colors;
	for(int i=0;i<height;i++)
	{
		int pixeloffset = ((height - i - 1) * unpadded_row_size);
		fwrite(&data[pixeloffset], 1, padded_row_size, outfile);
	}
	fclose(outfile);
}


int main()
{
	// vector<uint8_t> data1;
	// BITMAP A1 = readbmp("lena_colored_256.bmp", data1);
	// writebmp("lena_test2.bmp",A1,data1);

	vector<uint8_t> data2;
	BITMAP A2 = readbmp("cameraman.bmp", data2);
	writebmp("cameraman_test4.bmp",A2,data2);
}