#include <string>
#include <string.h>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <functional>
#include <cctype>
#include <locale>
#include <cmath>



bool fexists(const char *filename)
{
  std::ifstream ifile(filename);
  return ifile;
}

void destroy_arr_valid(void** ptr)
{
	if(*ptr!= NULL)
	{
		delete[] *ptr;
		*ptr = 0;
	}
}

// trim from start
std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

// trim from end
std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

// trim from both ends
std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
}


void ReadFortranRecord(FILE* IO, void* data_ptr) {
        unsigned int file_record_size = 0;
        //Read the first 4 bytes
        fread(&file_record_size, 4, 1, IO);
        //Read the output to the data pointer
        fread(data_ptr, file_record_size, 1, IO);
        //Read last 4 bytes of record
        fread(&file_record_size, 4, 1, IO);
}

void assertdouble(double & d1, double & d2, double tol)
{
	assert(abs(d1-d2) < tol);

}

