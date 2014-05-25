#include "common.h"
#include "cuda_objects.cuh"
#include "cuda_host.cuh"
#include "dipole_kernals.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "fields.h"
#include "Util.h"
#include "trove_functions.h"
#include "test.cuh"
#include <omp.h>






/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. */





int main(int argc,char** argv)
{
	////////----------THIS IS TESTED ON AN EMPTY 8 GPU NODE ON EMERALD
	//printf("%12.6f\n",three_j(2,3,1,1,-1,0));
	FintensityJob test_intensity;
	//float test = 3.6;
	//int i = 2;
	//test = 3.6*i;
	//printf("%11.2f\n",test); 
	//get_cuda_info(test_intensity);
	//exit(0);
	
	//
	char* gpu_env = getenv("NUM_GPUS");
	int num_gpu = 1;	
	
	if(gpu_env!=NULL){
		num_gpu = atoi(gpu_env);
	}

	int free_gpus = count_free_devices();
	printf("Found %i free devices\n",free_gpus);
	//exit(0);

	
	//dipol_do_intensities(test_intensity);
	//dipole_do_intensities_async(test_intensity,0);
	//Set number of threads
	dipole_initialise_cpu(&test_intensity);

	//num_gpu = 1;
	omp_set_dynamic(0);
	omp_set_num_threads(num_gpu);
	//Parallel region here
	
	double time = GetTimeMs64();

	#pragma omp parallel default(shared) shared(test_intensity)
	{
		
		int device= omp_get_thread_num();		
		dipole_do_intensities_async_omp(test_intensity,device,num_gpu);
				
	}

	time = GetTimeMs64() - time;
	printf("\ndone\n");
	printf("\ndone in %.fs\n",time/1000.0);

	//dipole_initialise(&test_intensity);
	//benchmark_half_ls(test_intensity,1);

	
	exit(0);

	return 0;

}
