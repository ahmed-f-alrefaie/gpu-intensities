#include "common.h"
#include "cuda_objects.cuh"
#include "cuda_host.cuh"
#include "dipole_kernals.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "fields.h"
#include "trove_functions.h"
#include <omp.h>
#include <sys/time.h>
#include <ctime>


typedef long int int64;
typedef unsigned long int uint64;

/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. */

int64 GetTimeMs64()
{

 /* Linux */
 struct timeval tv;

 gettimeofday(&tv, NULL);

 uint64 ret = tv.tv_usec;
 /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
 ret /= 1000;

 /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
 ret += (tv.tv_sec * 1000);

 return ret;

};



int main(int argc,char** argv)
{
	////////----------THIS IS TESTED ON AN EMPTY 8 GPU NODE ON EMERALD
	//printf("%12.6f\n",three_j(2,3,1,1,-1,0));
	FintensityJob test_intensity;
	//get_cuda_info(test_intensity);
	//exit(0);
	dipole_initialise_cpu(&test_intensity);
	//dipole_initialise(&test_intensity);
	//dipol_do_intensities(test_intensity);
	//dipole_do_intensities_async(test_intensity,0);
	//Set number of threads
	char* gpu_env = getenv("NUM_GPUS");
	int num_gpu = 1;	
	if(gpu_env!=NULL){
		num_gpu = atoi(gpu_env);
	}
	omp_set_dynamic(0);
	omp_set_num_threads(num_gpu);
	//Parallel region here
	
	double time = GetTimeMs64();

	#pragma omp parallel default(shared) shared(test_intensity)
	{
		int device = omp_get_thread_num();		
		dipole_do_intensities_async_omp(test_intensity,device,num_gpu);
	}

	time = GetTimeMs64() - time;
	printf("\ndone\n");
	printf("\ndone in %.fs\n",time/1000.0);
	exit(0);


	return 0;

}
