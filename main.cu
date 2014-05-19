#include "common.h"
#include "cuda_objects.cuh"
#include "cuda_host.cuh"
#include "dipole_kernals.cuh"
#include <cstdio>
#include <cmath>
#include "fields.h"
#include "trove_functions.h"
#include <omp.h>





int main(int argc,char** argv)
{
	////////----------THIS IS TESTED ON AN EMPTY 8 GPU NODE ON EMERALD
	printf("%12.6f\n",three_j(2,3,1,1,-1,0));
	FintensityJob test_intensity;
	//get_cuda_info(test_intensity);
	//exit(0);
	dipole_initialise_cpu(&test_intensity);
	//dipole_initialise(&test_intensity);
	//dipol_do_intensities(test_intensity);
	//dipole_do_intensities_async(test_intensity,0);
	//Set number of threads
	omp_set_dynamic(0);
	omp_set_num_threads(1);
	//Parallel region here
	
	#pragma omp parallel default(shared) shared(test_intensity)
	{
		int device = omp_get_thread_num();		
		dipole_do_intensities_async_omp(test_intensity,device,1);
	}
	exit(0);


	return 0;

}
