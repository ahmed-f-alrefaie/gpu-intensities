#include "trove_objects.h"
#include "cuda_objects.cuh"
#include "cuda_host.cuh"
#include "cublas_v2.h"

#include <cstdio>
#include <cstdlib>

double pi = 4.0 * atan2(1.0,1.0);
double A_coef_s_1 = 64.0*pow(10,-36) * pow(pi,4)  / (3.0 * 6.62606896*pow(10,-27));
double planck = 6.62606896*pow(10,-27);
double avogno = 6.0221415*pow(10,23);
double vellgt = 2.99792458*pow(10,10);
double intens_cm_mol  = 8.0*pow(10,-36) * pow(pi,3)*avogno/(3.0*planck*vellgt);
double boltz = 1.380658*pow(10,-16);
    //beta = planck * vellgt / (boltz * intensity%temperature)

void CheckCudaError(const char* tag){
  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("[%s] CUDA error: %s\n", tag,cudaGetErrorString(error));
    exit(-1);
  }
}


#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

void get_cuda_info(FintensityJob & intensity){
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }    // Iterate through devices

}


__host__ void copy_array_to_gpu(void* arr,void** arr_gpu,size_t arr_size,const char* arr_name)
{

		//Malloc dipole
		if(cudaSuccess != cudaMalloc(arr_gpu,arr_size))
		{
			fprintf(stderr,"[copy_array_to_gpu]: couldn't malloc for %s \n",arr_name);
			exit(0);			
		}
	if(cudaSuccess != cudaMemcpy((*arr_gpu),arr,arr_size,cudaMemcpyHostToDevice))
	{
		fprintf(stderr,"[copy_array_to_gpu]: error copying %s \n",arr_name);
		exit(0);
	}
};


//Copies relevant information needed to do intensity calculations onto the gpu
//Arguments p1: The bset_contr to copy p2: A device memory pointer to copy to
__host__ void copy_bset_contr_to_gpu(TO_bset_contrT* bset_contr,cuda_bset_contrT* bset_gptr,int* ijterms,int sym_nrepres,int*sym_degen)
{
	printf("Copying bset_contr for J=%i to gpu........",bset_contr->jval);
	//construct a gpu_bset_contr
	cuda_bset_contrT to_gpu_bset;
	printf("copy easy part\n");
	//Copy easy stuff
	to_gpu_bset.jval = bset_contr->jval;
	to_gpu_bset.Maxsymcoeffs = bset_contr->Maxsymcoeffs;
	to_gpu_bset.max_deg_size = bset_contr->max_deg_size;
	to_gpu_bset.Maxcontracts = bset_contr->Maxcontracts;
	to_gpu_bset.Nclasses = bset_contr->Nclasses;
	
	printf("copy icontr\n");
	//GPU pointer to icontr2icase///////////////////////////////////////
	int* icontr_gptr;
	//Malloc in the gpu
	if(cudaSuccess != cudaMalloc(&icontr_gptr,sizeof(int)*bset_contr->Maxcontracts*2))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for icontr2icase for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}
	//give the pointer to the cuda object
	to_gpu_bset.icontr2icase = icontr_gptr;
	
	//Copy over
	if(cudaSuccess != cudaMemcpy(icontr_gptr,bset_contr->icontr2icase,sizeof(int)*bset_contr->Maxcontracts*2,cudaMemcpyHostToDevice))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't copy icontr2icase to gpu for J=%i\n",to_gpu_bset.jval);
	}
	////////////////////////////////////////////////////////////////////////
	printf("copy iroot\n");
	////////////////////////////////Same for iroot_correlat_j0///////////////////////////////////////////////////
	int* iroot_corr_gptr;
	//Malloc in the gpu
	if(cudaSuccess != cudaMalloc (&iroot_corr_gptr , sizeof(int)*bset_contr->Maxcontracts ) )
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for iroot_correlat_j0 for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}
	//give the pointer to the cuda object
	to_gpu_bset.iroot_correlat_j0 = iroot_corr_gptr;
	
	//Copy over
	if(cudaSuccess != cudaMemcpy(iroot_corr_gptr,bset_contr->iroot_correlat_j0,sizeof(int)*bset_contr->Maxcontracts,cudaMemcpyHostToDevice))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't copy iroot_correlat_j0 to gpu for J=%i\n",to_gpu_bset.jval);
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////		K		////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("copy K\n");
	int* k_gptr;
	//Malloc in the gpu
	if(cudaSuccess != cudaMalloc(&k_gptr,sizeof(int)*bset_contr->Maxcontracts))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for K for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}
	//give the pointer to the cuda object
	to_gpu_bset.k = k_gptr;
	
	//Copy over
	if(cudaSuccess != cudaMemcpy(k_gptr,bset_contr->k,sizeof(int)*bset_contr->Maxcontracts,cudaMemcpyHostToDevice))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't copy k to gpu for J=%i\n",to_gpu_bset.jval);
	}	

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////		KTau		////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("copy Ktau\n");	
	int* kt_gptr;
	//Malloc in the gpu
	if(cudaSuccess != cudaMalloc(&kt_gptr,sizeof(int)*bset_contr->Maxcontracts))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for Ktau for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}
	//give the pointer to the cuda object
	to_gpu_bset.ktau = kt_gptr;
	
	//Copy over
	if(cudaSuccess != cudaMemcpy(kt_gptr,bset_contr->ktau,sizeof(int)*bset_contr->Maxcontracts,cudaMemcpyHostToDevice))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't copy ktau to gpu for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}	
	
	///////////////////////////////////////////////N///////////////////////////////////////////////////////////////////
	printf("copy N\n");
	int* N_gptr;
	if(cudaSuccess != cudaMalloc(&N_gptr,sizeof(int)*sym_nrepres*bset_contr->Maxsymcoeffs))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for N for J=%i\n",to_gpu_bset.jval);
	}
	to_gpu_bset.N = N_gptr;
	printf("Malloc\n");
	int* Ncopy = (int*)malloc(sizeof(int)*sym_nrepres*bset_contr->Maxsymcoeffs);
	printf("Make copy\n");
	for(int i = 0; i < sym_nrepres; i++)
		for(int j = 0; j < bset_contr->Maxsymcoeffs; j++)
		{
			Ncopy[ i + (j*sym_nrepres)] = bset_contr->irr[i].N[j];
			//printf("N[%i,%i] = %i %i\n",i,j,Ncopy[ i + (j*sym_nrepres)],bset_contr->irr[i].N[j]);
		}
	printf("Copy\n");		
	cudaMemcpy(N_gptr,Ncopy,sizeof(int)*sym_nrepres*bset_contr->Maxsymcoeffs,cudaMemcpyHostToDevice);
	
	to_gpu_bset.N = N_gptr;
	
	free(Ncopy);
	////////////////////////////////////////////////////////////////////////////////////////
	printf("copy Ntotal\n");
	//////////////////////////////N total////////////////////////////////////////////////////////
	int* Ntot_gptr;
	copy_array_to_gpu((void*)bset_contr->Ntotal,(void**)&Ntot_gptr,sizeof(int)*sym_nrepres,"Ntotal");
	to_gpu_bset.Ntotal = Ntot_gptr;
	///////////////////////////////////////////
	printf("copy irr_repres\n");
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////		irre_repres		////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	double** irr_gptr;
	if(cudaSuccess != cudaMalloc(&irr_gptr,sizeof(double*)*sym_nrepres))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for irreducible representation for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}	
	
	to_gpu_bset.irr_repres = irr_gptr;
	
	//Hold pointers to doubles
	double** d_ptr = (double**)malloc(sizeof(double*)*sym_nrepres);
	
	for(int i =0; i < sym_nrepres; i++)
	{
		
		if(cudaSuccess != cudaMalloc(&d_ptr[i],sizeof(double)*bset_contr->Ntotal[i]*sym_degen[i]*bset_contr->mat_size))
		{
			fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for irreducible representation for J=%i\n",to_gpu_bset.jval);
			exit(0);
		}
		//copy repres to irr_repres
		cudaMemcpy(d_ptr[i],bset_contr->irr[i].repres,sizeof(double)*bset_contr->Ntotal[i]*sym_degen[i]*bset_contr->mat_size,cudaMemcpyHostToDevice);
	}
	
	//copy pointerlist to irr_gptr;
	cudaMemcpy(irr_gptr,d_ptr,sizeof(double*)*sym_nrepres,cudaMemcpyHostToDevice);	
	free(d_ptr); //clear memory and pointer
	d_ptr = 0;
	
	printf("copy ijterms size = %i\n",bset_contr->Maxsymcoeffs*sym_nrepres);
	//Copy ijterms
	copy_array_to_gpu((void*)ijterms,(void**)&(to_gpu_bset.ijterms),sizeof(int)*bset_contr->Maxsymcoeffs*sym_nrepres,"ijterms");
	
	printf("copy final bset\n");
	/////////////////////////////////copy object over////////////////////////////////
	cudaMemcpy(bset_gptr,&to_gpu_bset,sizeof(cuda_bset_contrT),cudaMemcpyHostToDevice);
	
	printf(".....done!\n");

};

__host__ void create_and_copy_bset_contr_to_gpu(TO_bset_contrT* bset_contr,cuda_bset_contrT** bset_gptr,int* ijterms,int sym_nrepres,int*sym_degen)
{
	if(cudaSuccess != cudaMalloc(bset_gptr,sizeof(cuda_bset_contrT) ) )
	{
		fprintf(stderr,"[create_and_copy_bset_contr_to_gpu]: Couldn't allocate memory for bset\n");
		exit(0);
	}
	copy_bset_contr_to_gpu( bset_contr,*bset_gptr,ijterms,sym_nrepres,sym_degen);
}

//Copy threej
__host__ void copy_threej_to_gpu(double* threej,double** threej_gptr, int jmax)
{
	copy_array_to_gpu((void*)threej,(void**) threej_gptr, (jmax+1)*(jmax+1)*3*3*sizeof(double),"three_j");
	
};


///////////Dipole stuff now

__host__ void dipole_initialise(FintensityJob* intensity){
	printf("Begin Input\n");
	read_fields(intensity);
	printf("End Input\n");

	//Wake up the gpu//
	printf("Wake up gpu\n");
	cudaFree(0);
	printf("....Done!\n");
	
	int jmax = max(intensity->jvals[0],intensity->jvals[1]);

	//Now create the bset_contrs
	bset_contr_factory(&(intensity->bset_contr[0]),0,intensity->molec.sym_degen,intensity->molec.sym_nrepres);
	bset_contr_factory(&(intensity->bset_contr[1]),intensity->jvals[0],intensity->molec.sym_degen,intensity->molec.sym_nrepres);
	bset_contr_factory(&(intensity->bset_contr[2]),intensity->jvals[1],intensity->molec.sym_degen,intensity->molec.sym_nrepres);

	//Correlate them 
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[0]);
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[1]);
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[2]);
	
	printf("Reading dipole\n");
	//Read the dipole
	read_dipole(intensity->bset_contr[0],&(intensity->dipole_me),intensity->dip_size);
	printf("Computing threej\n");
	//Compute threej
	precompute_threej(&(intensity->threej),jmax);
	//ijterms
	printf("Computing ijerms\n");
	compute_ijterms((intensity->bset_contr[1]),&(intensity->bset_contr[1].ijterms),intensity->molec.sym_nrepres);
	compute_ijterms((intensity->bset_contr[2]),&(intensity->bset_contr[2].ijterms),intensity->molec.sym_nrepres);
		
	//Read eigenvalues
	read_eigenvalues((*intensity));

	intensity->dimenmax = 0;
	intensity->nsizemax = 0;
	//Find nsize
	for(int i =0; i < intensity->molec.sym_nrepres; i++){
		if(intensity->isym_do[i]){
			intensity->nsizemax= max(intensity->bset_contr[1].nsize[i],intensity->nsizemax);
			intensity->nsizemax = max(intensity->bset_contr[2].nsize[i],intensity->nsizemax);
		}
	}

	printf("Biggest vector dimensiton is %i \n",intensity->nsizemax);
	intensity->dimenmax = max(intensity->bset_contr[1].Maxcontracts,intensity->dimenmax);
	intensity->dimenmax = max(intensity->bset_contr[2].Maxcontracts,intensity->dimenmax);
	printf("Biggest max contraction is is %i \n",intensity->dimenmax);
	printf("Find igamma pairs\n");
	find_igamma_pair((*intensity));
	printf("done!\n");
	//Begin GPU related initalisation////////////////////////////////////////////////////////
	intensity_info int_gpu;
	//Copy over constants to GPU
	int_gpu.sym_nrepres = intensity->molec.sym_nrepres;
	int_gpu.jmax = jmax;
	int_gpu.dip_stride_1 = intensity->bset_contr[0].Maxcontracts;
	int_gpu.dip_stride_2 = intensity->bset_contr[0].Maxcontracts*intensity->bset_contr[0].Maxcontracts;
	int_gpu.dimenmax = intensity->dimenmax;
	int_gpu.sq2 = 1.0/sqrt(2.0);

	copy_array_to_gpu((void*)intensity->molec.sym_degen,(void**)&int_gpu.sym_degen,sizeof(int)*intensity->molec.sym_nrepres,"sym_degen");

	CheckCudaError("Pre-initial");
	printf("Copy intensity information\n");	
	copy_intensity_info(&int_gpu);
	printf("done\n");
	CheckCudaError("Post-initial");
	printf("Copying bset_contrs to GPU...\n");
	intensity->g_ptrs.bset_contr = new cuda_bset_contrT*[2];
	create_and_copy_bset_contr_to_gpu(&intensity->bset_contr[1],&(intensity->g_ptrs.bset_contr[0]),intensity->bset_contr[1].ijterms,intensity->molec.sym_nrepres,intensity->molec.sym_degen);
	create_and_copy_bset_contr_to_gpu(&intensity->bset_contr[2],&(intensity->g_ptrs.bset_contr[1]),intensity->bset_contr[2].ijterms,intensity->molec.sym_nrepres,intensity->molec.sym_degen);

	printf("Done\n");
	
	printf("Copying threej...\n");
	copy_threej_to_gpu(intensity->threej,&(intensity->g_ptrs.threej), jmax);
	printf("done\n");

	printf("Copying dipole\n");
	copy_array_to_gpu((void*)intensity->dipole_me,(void**)&(intensity->g_ptrs.dipole_me),intensity->dip_size,"dipole_me");
	printf("Done..");
	//exit(0);
	//Will improve
	intensity->gpu_memory = 1l*1024l*1024l*1024l;
	intensity->cpu_memory = 1l*1024l*1024l*1024l;
	
};

__host__ void dipole_do_intensities(FintensityJob & intensity){

	//Prinf get available cpu memory
	unsigned long available_cpu_memory = intensity.cpu_memory;
	unsigned long available_gpu_memory = intensity.gpu_memory;

	//Compute how many inital state vectors and final state vectors
	unsigned long no_final_states_cpu = ((available_cpu_memory)/8l - long(2*intensity.dimenmax))/(3l*intensity.dimenmax);//(Initial + vec_cor + half_ls)*dimen_max
	unsigned long no_final_states_gpu = ((available_gpu_memory)/8l - long(2*intensity.dimenmax))/(3l*intensity.dimenmax);//(Initial + vec_cor + half_ls)*dimen_max
	printf("No of final states in gpu_memory: %d  cpu memory: %d\n",no_final_states_gpu,no_final_states_cpu);

	//The intial state vector
	double* initial_vec = new double[intensity.dimenmax];

	double* gpu_initial_vec=NULL;

	copy_array_to_gpu((void*)initial_vec,(void**)&(gpu_initial_vec),sizeof(double)*intensity.dimenmax,"gpu_initial_vec");
	printf("%p\n",gpu_initial_vec);

	double* final_vec = new double[intensity.dimenmax];
	double* gpu_final_vec=NULL;
	
	copy_array_to_gpu((void*)final_vec,(void**)&(gpu_final_vec),sizeof(double)*intensity.dimenmax,"gpu_final_vec");

	double* corr_vec = new double[intensity.dimenmax];
	double* gpu_corr_vec=NULL;

	copy_array_to_gpu((void*)corr_vec,(void**)&(gpu_corr_vec),sizeof(double)*intensity.dimenmax,"gpu_corr_vec");

	double* half_ls = new double[intensity.dimenmax];
	double** gpu_half_ls=new double*[2];

	copy_array_to_gpu((void*)half_ls,(void**)&(gpu_half_ls[0]),sizeof(double)*intensity.dimenmax,"gpu_half_ls1");
	copy_array_to_gpu((void*)half_ls,(void**)&(gpu_half_ls[1]),sizeof(double)*intensity.dimenmax,"gpu_half_ls2");
	double line_str =0.0;


	char filename[1024];
	//Get the filename
	printf("Open vector unit\n");
	FILE** eigenvec_unit = new FILE*[2*intensity.molec.sym_nrepres];
	for(int i =0; i< 2; i++){
		for(int j = 0; j < intensity.molec.sym_nrepres; j++)
		{

			sprintf(filename,j0eigen_vector_gamma_filebase,intensity.jvals[i],j+1);
			printf("Reading %s\n",filename);
			eigenvec_unit[i + j*2] = fopen(filename,"r");
			if(eigenvec_unit[i + j*2] == NULL)
			{
				printf("error opening %s \n",filename);
				exit(0);
			}
		}
	}
	
	//Opened all units, now lets start compuing
	
	//Initialise cublas
	cublasHandle_t handle;
	cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
		return;
	}
	
	CheckCudaError("Initialisation");

			    // Number of threads in each thread block
    	int blockSize =256;
 
    	// Number of thread blocks in grid
    	int gridSize = (int)ceil((float)intensity.dimenmax/blockSize);

			printf("Nu_if\tJf Kf quantaF\t <-- \tJI KI tauI quantaI\t Ein_A\tLine_str\n");
	//Run
	for(int ilevelI = 0; ilevelI < intensity.Neigenlevels; ilevelI++){
	
			    //  ! start measuring time per line
	   //   !
	      int indI = intensity.eigen[ilevelI].jind;
	  //    !
	  //    !dimension of the bases for the initial states
	  //    !
	     int dimenI = intensity.bset_contr[indI+1].Maxcontracts;
	   //   !
	    //  !energy, quanta, and gedeneracy order of the initial state
	    //  !
	      int jI = intensity.eigen[ilevelI].jval;
	      double energyI = intensity.eigen[ilevelI].energy;
	      int igammaI  = intensity.eigen[ilevelI].igamma;
	      int * quantaI = intensity.eigen[ilevelI].quanta;
	      int * normalI = intensity.eigen[ilevelI].normal;
	      int ndegI   = intensity.eigen[ilevelI].ndeg;
	      int nsizeI = intensity.bset_contr[indI+1].nsize[igammaI];

	      FILE* unitI = eigenvec_unit[ indI + (igammaI)*2]; 
	    //   printf("Ilevel = %i\n",ilevelI);

	      if(!energy_filter_lower(intensity,jI,energyI,quantaI)) continue;
	      fseek(unitI,(intensity.eigen[ilevelI].irec[0]-1)*nsizeI*sizeof(double),SEEK_SET);


		//Read vector from file
	    //  printf("Read vector\n");
	     int tread =  fread(initial_vec,sizeof(double),nsizeI,unitI);

		//for(int i=0; i< nsizeI; i++){
		//	printf("vec[%i]=%16.8e\n",i,initial_vec[i]);}
		//printf("read = %i\n",tread);
		//Transfer it to the GPU
	//	printf("Transfer vector\n");
	        stat = cublasSetVector(intensity.dimenmax, sizeof(double),initial_vec, 1, gpu_initial_vec, 1);
		CheckCudaError("Set Vector I");

		cudaDeviceSynchronize();

	  //    printf("Correlating vectors\n");
		//for(int ideg = 0; ideg < ndegI; ideg++){
		//host_correlate_vectors(&intensity.bset_contr[indI+1],0,igammaI,intensity.bset_contr[indI+1].ijterms,intensity.molec.sym_degen,initial_vec,corr_vec);
	      	
		device_correlate_vectors<<<gridSize,blockSize>>>(intensity.g_ptrs.bset_contr[indI],0,igammaI, gpu_initial_vec,gpu_corr_vec);
			CheckCudaError("device correlate I");
			cudaDeviceSynchronize();

//
		//printf("Done\n");
		printf("J= %i energy = %11.4f\n",jI,energyI);

		

		printf("----------------------------------\n");
	     for(int indF=0; indF <2; indF++){
	     	device_compute_1st_half_ls<<<gridSize,blockSize>>>(intensity.g_ptrs.bset_contr[indI],intensity.g_ptrs.bset_contr[indF],intensity.g_ptrs.dipole_me,igammaI,0,gpu_corr_vec,intensity.g_ptrs.threej,gpu_half_ls[indF]);
			//CheckCudaError("compute half ls I");
			//cudaDeviceSynchronize();
			//cublasGetVector(dimenI, sizeof(double),gpu_half_ls[indF], 1, half_ls, 1);
			//for(int i=0; i< dimenI; i++){
			//  printf("half_ls[%i]=%16.8e\n",i,half_ls[i]);}
			//printf("----------------------------------\n");

		}
	
		


			
	
		//Final states
		for(int ilevelF = 0; ilevelF < intensity.Neigenlevels; ilevelF++){

					    //  ! start measuring time per line
		   //   !
		      int indF = intensity.eigen[ilevelF].jind;
		  //    !
			//printf("indF=%i",indF);
		  //    !dimension of the bases for the initial states
		  //    !
		     int dimenF = intensity.bset_contr[indF+1].Maxcontracts;
		   //   !
		    //  !energy, quanta, and gedeneracy order of the initial state
		    //  !
		      int jF = intensity.eigen[ilevelF].jval;
		      double energyF = intensity.eigen[ilevelF].energy;
		      int igammaF  = intensity.eigen[ilevelF].igamma;
		      int * quantaF = intensity.eigen[ilevelF].quanta;
		      int * normalF = intensity.eigen[ilevelF].normal;
		      int ndegF   = intensity.eigen[ilevelF].ndeg;
		      int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];

			FILE* unitF = eigenvec_unit[ indF + (igammaF)*2]; 

		      if(!energy_filter_upper(intensity,jF,energyF,quantaF)) continue;

			for(int i = 0; i < intensity.dimenmax; i++){
				final_vec[i]=0.0;
			}

		      fseek(unitF,(intensity.eigen[ilevelF].irec[0]-1)*nsizeF*sizeof(double),SEEK_SET);
			//Read vector from file
		      fread(final_vec,sizeof(double),nsizeF,unitF);
		
			//for(int i=0; i< dimenF; i++){
			//	printf("ivec[%i]=%16.8e\n",i,final_vec[i]);}

			if(!intensity_filter(intensity,jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF)) continue;
			//device_clear_vector<<<gridSize,blockSize>>>(gpu_final_vec);
			//Transfer it to the GPU
		      stat = cublasSetVector(intensity.dimenmax, sizeof(double),final_vec, 1, gpu_final_vec, 1);

			if (stat != CUBLAS_STATUS_SUCCESS) {
				printf ("CUBLAS SetVector F failed\n");
				printf ("Error code: %s\n",_cudaGetErrorEnum(stat));
				return;
			}

			double nu_if = energyF - energyI; 
			//for(int ideg = 0; ideg < ndegF; ideg++){
		        device_correlate_vectors<<<gridSize,blockSize>>>(intensity.g_ptrs.bset_contr[indF],0,igammaF, gpu_final_vec,gpu_corr_vec);
			CheckCudaError("correlate final vector");
		        cudaDeviceSynchronize();
			
			//cublasGetVector(dimenF, sizeof(double),gpu_corr_vec, 1, corr_vec, 1);
			//for(int i=0; i< dimenF; i++){
			//	printf("ivec[%i]=%16.8e\n",i,corr_vec[i]);}
			
			//}

//
			cudaDeviceSynchronize();
			//Compute ls
		//	for(int i = 0; i < dimenF; i++)
		//			printf("%11.4e\n",corr_vec[i]);
		//	//exit(0);
			line_str = 0;
			//cublasDdot (handle,intensity.dimenmax,gpu_half_ls[indF], 1,gpu_corr_vec, 1,&line_str);
			cublasDdot (handle, intensity.dimenmax, gpu_corr_vec, 1, gpu_half_ls[indF], 1, &line_str);
			//cublasDdot (handle, intensity.dimenmax, gpu_half_ls[indF], 1, gpu_half_ls[indF], 1, &line_str);
			double orig_ls = line_str;
			//Print intensitys
			line_str *= line_str;
			//printf("line_str %11.4e\n",line_str);
			double A_einst = A_coef_s_1*double((2*jI)+1)*line_str*pow(abs(nu_if),3);
			 line_str = line_str * intensity.gns[igammaI] * double( (2*jI + 1)*(2 * jF + 1) );

			//if(line_str < 0.00000000001) continue;
			/*

               write(out, "( (i4, 1x, a4, 3x),'<-', (i4, 1x, a4, 3x),a1,&
                            &(2x, f11.4,1x),'<-',(1x, f11.4,1x),f11.4,2x,&
                            &'(',1x,a3,x,i3,1x,')',1x,'(',1x,<nclasses>(x,a3),1x,<nmodes>(1x, i3),1x,')',1x,'<- ',   &
                            &'(',1x,a3,x,i3,1x,')',1x,'(',1x,<nclasses>(x,a3),1x,<nmodes>(1x, i3),1x,')',1x,   &
                            & 3(1x, es16.8),2x,(1x,i6,1x),'<-',(1x,i6,1x),i8,1x,i8,&
                            1x,'(',1x,<nmodes>(1x, i3),1x,')',1x,'<- ',1x,'(',1x,<nmodes>(1x, i3),1x,')',1x,& 
                            <nformat>(1x, es16.8))")  &
                            !
                            jF,sym%label(igammaF),jI,sym%label(igammaI),branch, &
                            energyF-intensity%ZPE,energyI-intensity%ZPE,nu_if,                 &
                            eigen(ilevelF)%cgamma(0),eigen(ilevelF)%krot,&
                            eigen(ilevelF)%cgamma(1:nclasses),eigen(ilevelF)%quanta(1:nmodes), &
                            eigen(ilevelI)%cgamma(0),eigen(ilevelI)%krot,&
                            eigen(ilevelI)%cgamma(1:nclasses),eigen(ilevelI)%quanta(1:nmodes), &
                            linestr,A_einst,absorption_int,&
                            eigen(ilevelF)%ilevel,eigen(ilevelI)%ilevel,&
                            itransit,istored(ilevelF),normalF(1:nmodes),normalI(1:nmodes),&
                            linestr_deg(1:ndegI,1:ndegF)
             endif

			*/


			printf("%11.4f\t(%i %i ) ( ",nu_if,jF,intensity.eigen[ilevelF].krot);
			for(int i = 0; i < intensity.molec.nmodes+1; i++)
				printf("%i ",quantaF[i]);
			printf(")\t <-- \t(%i %i ) ",jI,intensity.eigen[ilevelI].krot);
			for(int i = 0; i < intensity.molec.nmodes+1; i++)
				printf("%i ",quantaI[i]);	
			printf("\t %16.8e    %16.8e %16.8e\n",A_einst,line_str,orig_ls);

			//exit(0);		



			

			
		}
		
		
		


	}


	

		
	
}

__host__ void dipole_do_intensities_async(FintensityJob & intensity,int device_id){
	//Prinf get available cpu memory
	unsigned long available_cpu_memory = intensity.cpu_memory;
	unsigned long available_gpu_memory = intensity.gpu_memory;

	int nJ = 2;

	//Compute how many inital state vectors and final state vectors
	unsigned long no_final_states_cpu = ((available_cpu_memory)/8l - long(2*intensity.dimenmax))/(3l*intensity.dimenmax);//(Initial + vec_cor + half_ls)*dimen_max
	unsigned long no_final_states_gpu = ((available_gpu_memory)/8l - long(2*intensity.dimenmax))/(3l*intensity.dimenmax);//(Initial + vec_cor + half_ls)*dimen_max

	no_final_states_gpu = min((unsigned int )intensity.Neigenlevels,(unsigned int )no_final_states_gpu)/2l;//half are the eigenvectors, the other half are the correlations

	printf("No of final states in gpu_memory: %d  cpu memory: %d\n",no_final_states_gpu,no_final_states_cpu);



	//Half linestrength related variable
	cudaStream_t* st_half_ls = new cudaStream_t[nJ*intensity.molec.sym_maxdegen]; 	//Concurrently run half_ls computations on this many of the half_ls's
	double* half_ls = new double[intensity.dimenmax*nJ*intensity.molec.sym_maxdegen]; // half_ls(dimen,indF,ndeg)
	double* gpu_half_ls;
	//Create initial vector holding point
	double* initial_vector = new double[intensity.dimenmax];
	double* gpu_initial_vector;
	


	//Final vectors
	//Streams for each final vector computation
	cudaStream_t* st_ddot_vectors = new cudaStream_t[no_final_states_gpu];
	cudaStream_t intial_f_memcpy;
	double* final_vectors;
	cudaMallocHost(&final_vectors,sizeof(double)*intensity.dimenmax*no_final_states_gpu);
	//= new double[intensity.dimenmax*no_final_states_gpu]; //Pin this memory in final build
	//int* vec_ilevelF = new int[no_final_states_gpu];

	double* gpu_corr_vectors;
	double* gpu_final_vectors;

	int** vec_ilevel_buff = new int*[2];
	vec_ilevel_buff[0] = new int[no_final_states_gpu];
	vec_ilevel_buff[1] = new int[no_final_states_gpu];

	double* line_str = new double[no_final_states_gpu*intensity.molec.sym_maxdegen*intensity.molec.sym_maxdegen];
	double* gpu_line_str;
	//Track which vectors we are using
	int vector_idx=0;
	int vector_count=0;	
	int ilevel_total=0;
	int ilevelF=0,start_ilevelF=0;

	//Copy them to the gpu
	copy_array_to_gpu((void*)initial_vector,(void**)&(gpu_initial_vector),sizeof(double)*intensity.dimenmax,"gpu_initial_vector");
	copy_array_to_gpu((void*)final_vectors,(void**)&(gpu_final_vectors),sizeof(double)*intensity.dimenmax*no_final_states_gpu,"gpu_final_vectors");
	copy_array_to_gpu((void*)final_vectors,(void**)&(gpu_corr_vectors),sizeof(double)*intensity.dimenmax*no_final_states_gpu,"gpu_corr_vectors");
	copy_array_to_gpu((void*)half_ls,(void**)&(gpu_half_ls),sizeof(double)*intensity.dimenmax*nJ*intensity.molec.sym_maxdegen,"gpu_half_ls");
	//copy_array_to_gpu((void*)line_str,(void**)&(gpu_line_str),sizeof(double)*intensity.dimenmax*nJ*intensity.molec.sym_maxdegen,"gpu_line_str");
	//Open the eigenvector units
	char filename[1024];

	//Get the filename1552 bytes stack frame, 24 bytes spill stores, 24 bytes spill loads

	printf("Open vector units\n");
	FILE** eigenvec_unit = new FILE*[2*intensity.molec.sym_nrepres];
	for(int i =0; i< 2; i++){
		for(int j = 0; j < intensity.molec.sym_nrepres; j++)
		{

			sprintf(filename,j0eigen_vector_gamma_filebase,intensity.jvals[i],j+1);
			printf("Reading %s\n",filename);
			eigenvec_unit[i + j*2] = fopen(filename,"r");
			if(eigenvec_unit[i + j*2] == NULL)
			{
				printf("error opening %s \n",filename);
				exit(0);
			}
		}
	}
	
	//Initialise cublas
	cublasHandle_t handle;
	cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
		return;
	}	
	
	//Create the streams
	//Intial state
	for(int i = 0; i < intensity.molec.sym_maxdegen; i++)
		for(int j=0; j < nJ; j++)
			cudaStreamCreate(&st_half_ls[j + i*nJ]);

	//Final states
	cudaStreamCreate(&intial_f_memcpy);
	for(int i = 0; i < no_final_states_gpu; i++)
		cudaStreamCreate(&st_ddot_vectors[i]);


	int last_ilevelF= 0;
	// Number of threads in each thread block
    	int blockSize =768;

 
    	// Number of thread blocks in grid
    	int gridSize = (int)ceil((float)intensity.dimenmax/blockSize);
	

	//////Begin the computation//////////////////////////////////////////////////////////////////////////////
	CheckCudaError("Initialisation");

//(/t4'J',t6'Gamma <-',t17'J',t19'Gamma',t25'Typ',t35'Ef',t42'<-',t50'Ei',t62'nu_if',&
//                   &t85,<nclasses>(4x),1x,<nmodes>(4x),3x,'<-',14x,<nclasses>(4x),1x,<nmodes>(4x),&
//                   &8x,'S(f<-i)',10x,'A(if)',12x,'I(f<-i)',12x,'Ni',8x,'Nf',8x,'N')"

	printf("Nu_if\tJf Kf quantaF\t <-- \tJI KI tauI quantaI\t Ein_A\tLine_str\n");

	//Run
	for(int ilevelI = 0; ilevelI < intensity.Neigenlevels; ilevelI++){
		//printf("new I level!\n");
		//Get the basic infor we need
	      int indI = intensity.eigen[ilevelI].jind;

	      int dimenI = intensity.bset_contr[indI+1].Maxcontracts;

	      int jI = intensity.eigen[ilevelI].jval;
	      double energyI = intensity.eigen[ilevelI].energy;
	      int igammaI  = intensity.eigen[ilevelI].igamma;
	      int * quantaI = intensity.eigen[ilevelI].quanta;
	      int * normalI = intensity.eigen[ilevelI].normal;
	      int ndegI   = intensity.eigen[ilevelI].ndeg;
	      int nsizeI = intensity.bset_contr[indI+1].nsize[igammaI];

	      FILE* unitI = eigenvec_unit[ indI + (igammaI)*2]; 
		//Check filters
		
	      if(!energy_filter_lower(intensity,jI,energyI,quantaI)) continue;
		//If success then read
	      fseek(unitI,(intensity.eigen[ilevelI].irec[0]-1)*nsizeI*sizeof(double),SEEK_SET);
	      fread(initial_vector,sizeof(double),nsizeI,unitI);

	      stat = cublasSetVector(intensity.dimenmax, sizeof(double),initial_vector, 1, gpu_initial_vector, 1);
	      CheckCudaError("Set Vector I");			


	      printf("State J = %i Energy = %11.4f igammaI = %i ilevelI = %i\n",jI,energyI,igammaI,ilevelI);

    	      blockSize =768;

 
    	// Number of thread blocks in grid
    	      gridSize = (int)ceil((float)dimenI/blockSize);
              //We have the vector now we compute the half_ls Asynchronously
	      for(int ideg=0; ideg < ndegI; ideg++){
			//printf("ideg=%i ndegI=%i\n",ideg,ndegI);
			device_correlate_vectors<<<gridSize,blockSize,0,st_half_ls[ideg]>>>(intensity.g_ptrs.bset_contr[indI],ideg,igammaI, gpu_initial_vector,gpu_corr_vectors + intensity.dimenmax*ideg); //This will have priority
	      
		}
		//printf("Correlate");		
	      cudaDeviceSynchronize();
	      for(int ideg=0; ideg < ndegI; ideg++){
			for(int indF =0; indF < nJ; indF++){
				//These will execute asychronously
	     			device_compute_1st_half_ls<<<gridSize,blockSize,0,st_half_ls[indF + ideg*nJ]>>>(intensity.g_ptrs.bset_contr[indI],intensity.g_ptrs.bset_contr[indF],
								   intensity.g_ptrs.dipole_me,igammaI,ideg,gpu_corr_vectors + intensity.dimenmax*ideg,intensity.g_ptrs.threej,
								   gpu_half_ls + indF*intensity.dimenmax + ideg*intensity.dimenmax*nJ);				
			}

			 //wait for the next batch
			
	      }


				vector_idx=0;
		ilevelF=0;
		int current_buff = 0;
		//printf("First half_ls");
		//While the half_ls is being computed, lets load up some final state vectors
		while(vector_idx < no_final_states_gpu && ilevelF < intensity.Neigenlevels)
		{
					   //   !
		      	int indF = intensity.eigen[ilevelF].jind;
		  //    !
			//printf("indF=%i",indF);
		  //    !dimension of the bases for the initial states
		  //    !
		   //   !
		      //!energy, quanta, and gedeneracy order of the initial state
		     // !
		      	int jF = intensity.eigen[ilevelF].jval;
		      	double energyF = intensity.eigen[ilevelF].energy;
		      	int igammaF  = intensity.eigen[ilevelF].igamma;
		      	int * quantaF = intensity.eigen[ilevelF].quanta;
		      	int * normalF = intensity.eigen[ilevelF].normal;
		     	 int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];
			int irec = intensity.eigen[ilevelF].irec[0]-1;
			FILE* unitF = eigenvec_unit[ indF + (igammaF)*2]; 			

			ilevelF++;
			if(!energy_filter_upper(intensity,jF,energyF,quantaF)) {continue;}
			if(!intensity_filter(intensity,jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF)) continue;
 			// store the level
			vec_ilevel_buff[0][vector_idx] = ilevelF-1;
			//printf("ilevelF=%i\n",vec_ilevel_buff[0][vector_idx]);
			//Otherwise load the vector to a free slot
			fseek(unitF,irec*nsizeF*sizeof(double),SEEK_SET);
			fread(final_vectors + vector_idx*intensity.dimenmax,sizeof(double),nsizeF,unitF);
			//Increment
			vector_idx++;
		}
		vector_count = vector_idx;
		

		//printf("memcopy");
		//Memcopy it in one go
		cudaMemcpyAsync(gpu_final_vectors,final_vectors,sizeof(double)*intensity.dimenmax*vector_count,cudaMemcpyHostToDevice,intial_f_memcpy) 	;


		cudaDeviceSynchronize(); //Wait till we're set up

		CheckCudaError("Batch final vectors");	
		//printf("vector_count = %i\n",vector_count);

		while(vector_count != 0)
		{
			for(int i = 0; i < vector_count; i++){
				ilevelF = vec_ilevel_buff[int(current_buff)][i];
				//printf("ilevelF=%i\n",ilevelF);
				int indF = intensity.eigen[ilevelF].jind;
			      	int jF = intensity.eigen[ilevelF].jval;
			      	double energyF = intensity.eigen[ilevelF].energy;
			      	int igammaF  = intensity.eigen[ilevelF].igamma;
			      	int * quantaF = intensity.eigen[ilevelF].quanta;
			      	int * normalF = intensity.eigen[ilevelF].normal;
			     	 int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];
				int irec = intensity.eigen[ilevelF].irec[0]-1;
				int dimenF = intensity.bset_contr[indF+1].Maxcontracts;

				int idegF = 0;
				int idegI = 0;
				//for(int i = 0; i < ndeg
				//Correlate the vectors
				device_correlate_vectors<<<gridSize,blockSize,0,st_ddot_vectors[i]>>>(intensity.g_ptrs.bset_contr[indF],0,igammaF, (gpu_final_vectors + i*intensity.dimenmax),gpu_corr_vectors + intensity.dimenmax*i);
				cublasSetStream(handle,st_ddot_vectors[i]);
				cublasDdot (handle, dimenF,gpu_corr_vectors + intensity.dimenmax*i, 1, gpu_half_ls + indF*intensity.dimenmax + idegI*intensity.dimenmax*nJ, 1, &line_str[i + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen]);
				//cublasDDot(handle,, dimenF,gpu_corr_vectors + intensity.dimenmax*i, 1, gpu_half_ls + indF*intensity.dimenmax + idegI*intensity.dimenmax*nJ, 1, line_str + i + idegF*no_final_states_gpu + idegI*no_final_states_gpu*intensity.molec.sym_maxdegen);
			
			}
			current_buff = 1-current_buff;
			vector_idx = 0;
			ilevelF++;
			//While the line_Strength is being computed, lets load up some final state vectors
			while(vector_idx < no_final_states_gpu && ilevelF < intensity.Neigenlevels)
			{
						   //   !
			      	int indF = intensity.eigen[ilevelF].jind;
			  //    !
				//printf("indF=%i",indF);
			  //    !dimension of the bases for the initial states
			  //    !
			   //   !
			      //!energy, quanta, and gedeneracy order of the initial state
			     // !
			      	int jF = intensity.eigen[ilevelF].jval;
			      	double energyF = intensity.eigen[ilevelF].energy;
			      	int igammaF  = intensity.eigen[ilevelF].igamma;
			      	int * quantaF = intensity.eigen[ilevelF].quanta;
			      	int * normalF = intensity.eigen[ilevelF].normal;
			     	 int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];
				int irec = intensity.eigen[ilevelF].irec[0]-1;
				FILE* unitF = eigenvec_unit[ indF + (igammaF)*2]; 			

				ilevelF++;
				if(!energy_filter_upper(intensity,jF,energyF,quantaF)) {continue;}
				if(!intensity_filter(intensity,jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF)) continue;
				 // store the level				
				vec_ilevel_buff[current_buff][vector_idx] = ilevelF-1;
				//load the vector to a free slot
				fseek(unitF,irec*nsizeF*sizeof(double),SEEK_SET);
				fread(final_vectors + vector_idx*intensity.dimenmax,sizeof(double),nsizeF,unitF);
				//cudaMemcpyAsync(gpu_final_vectors,final_vectors,sizeof(double)*intensity.dimenmax*vector_count,cudaMemcpyHostToDevice,st_ddot_vectors[vector_idx]) ;
				//Increment
				vector_idx++;
			}
			last_ilevelF=ilevelF;

			cudaDeviceSynchronize();
			cudaMemcpyAsync(gpu_final_vectors,final_vectors,sizeof(double)*intensity.dimenmax*vector_count,cudaMemcpyHostToDevice,intial_f_memcpy) ;
			//We'e done now lets output
			for(int i = 0; i < vector_count; i++)
			{
				ilevelF = vec_ilevel_buff[1-current_buff][i];
				//printf("ilevelF=%i\n",ilevelF);
				int indF = intensity.eigen[ilevelF].jind;
			      	int jF = intensity.eigen[ilevelF].jval;
			      	double energyF = intensity.eigen[ilevelF].energy;
			      	int igammaF  = intensity.eigen[ilevelF].igamma;
			      	int * quantaF = intensity.eigen[ilevelF].quanta;
			      	int * normalF = intensity.eigen[ilevelF].normal;
				cudaStreamSynchronize(st_ddot_vectors[i]);
				double orig_ls = line_str[i];
				double final_ls = line_str[i];
				double nu_if = energyF - energyI; 
				//Print intensitys
				final_ls *= final_ls;
				//printf("line_str %11.4e\n",line_str);
				double A_einst = A_coef_s_1*double((2*jI)+1)*final_ls*pow(abs(nu_if),3);
				 final_ls = final_ls * intensity.gns[igammaI] * double( (2*jI + 1)*(2 * jF + 1) );
				//if(final_ls < intensity.thresh_linestrength) continue;
	/*
				printf("%11.4f\t(%i %i ) ( ",nu_if,jF,intensity.eigen[ilevelF].krot);

				for(int i = 0; i < intensity.molec.nmodes+1; i++)
					printf("%i ",quantaF[i]);

				printf(")\t <-- \t(%i %i ) ",jI,intensity.eigen[ilevelI].krot);

				for(int i = 0; i < intensity.molec.nmodes+1; i++)
					printf("%i ",quantaI[i]);	

				printf("\t %16.8e    %16.8e %16.8e\n",A_einst,final_ls,orig_ls);	
*/

			/*	               write(out, "( (i4, 1x, a4, 3x),'<-', (i4, 1x, a4, 3x),a1,&
                            &(2x, f11.4,1x),'<-',(1x, f11.4,1x),f11.4,2x,&
                            &'(',1x,a3,x,i3,1x,')',1x,'(',1x,<nclasses>(x,a3),1x,<nmodes>(1x, i3),1x,')',1x,'<- ',   &
                            &'(',1x,a3,x,i3,1x,')',1x,'(',1x,<nclasses>(x,a3),1x,<nmodes>(1x, i3),1x,')',1x,   &
                            & 3(1x, es16.8),2x,(1x,i6,1x),'<-',(1x,i6,1x),i8,1x,i8,&
                            1x,'(',1x,<nmodes>(1x, i3),1x,')',1x,'<- ',1x,'(',1x,<nmodes>(1x, i3),1x,')',1x,& 
                            <nformat>(1x, es16.8))")  &
                            !
                            jF,sym%label(igammaF),jI,sym%label(igammaI),branch, &
                            energyF-intensity%ZPE,energyI-intensity%ZPE,nu_if,                 &
                            eigen(ilevelF)%cgamma(0),eigen(ilevelF)%krot,&
                            eigen(ilevelF)%cgamma(1:nclasses),eigen(ilevelF)%quanta(1:nmodes), &
                            eigen(ilevelI)%cgamma(0),eigen(ilevelI)%krot,&
                            eigen(ilevelI)%cgamma(1:nclasses),eigen(ilevelI)%quanta(1:nmodes), &
                            linestr,A_einst,absorption_int,&
                            eigen(ilevelF)%ilevel,eigen(ilevelI)%ilevel,&
                            itransit,istored(ilevelF),normalF(1:nmodes),normalI(1:nmodes),&
                            linestr_deg(1:ndegI,1:ndegF)
			*/	
			   printf("%4i %4s   <-%4i %4s   %1s  %11.4f <- %11.4f %11.4f  ( %3s %3i ) ( ",jF,"BA",jI,"BS",branch(jF,jI),energyF-intensity.ZPE,energyI-intensity.ZPE,abs(nu_if),"GA",intensity.eigen[ilevelF].krot);
			   for(int i = 0; i < intensity.molec.nclasses; i++)
					printf(" %3s","X");
			   printf(" ");
			   for(int i = 1; i <= intensity.molec.nmodes; i++)
					printf(" %3i",intensity.eigen[ilevelF].quanta[i]);
			   printf(" ) <- ( %3s %3i ) ( ","X",intensity.eigen[ilevelI].krot);

			   for(int i = 0; i < intensity.molec.nclasses; i++)
					printf(" %3s","X");
			   printf(" ");
			   for(int i = 1; i <= intensity.molec.nmodes; i++)
					printf(" %3i",intensity.eigen[ilevelI].quanta[i]);
			   printf(")  %16.8e %16.8e %16.8e  \n",final_ls,A_einst);


			}			 
			ilevelF=last_ilevelF+1;
			//Save the new vector_count
			vector_count = vector_idx;
			//printf("Vector_count=%i, ilevelF=%i!\n",vector_count,ilevelF);
			//Copy the bunch of new vectors to the gpu
			
			//printf("Done!\n");
			CheckCudaError("Compute final vectors");
			cudaDeviceSynchronize();
		}
		//cudaDeviceReset();
		//cudaDeviceSynchronize();
		//exit(0);
		
	

	}

	

}

/////////////////////////////--------------------Multi-threaded verstions--------------------///////////////////////////////////////////////

__host__ void dipole_initialise_cpu(FintensityJob* intensity){
	printf("Begin Input\n");
	read_fields(intensity);
	printf("End Input\n");
	
	int jmax = max(intensity->jvals[0],intensity->jvals[1]);

	//Now create the bset_contrs
	bset_contr_factory(&(intensity->bset_contr[0]),0,intensity->molec.sym_degen,intensity->molec.sym_nrepres);
	bset_contr_factory(&(intensity->bset_contr[1]),intensity->jvals[0],intensity->molec.sym_degen,intensity->molec.sym_nrepres);
	bset_contr_factory(&(intensity->bset_contr[2]),intensity->jvals[1],intensity->molec.sym_degen,intensity->molec.sym_nrepres);

	//Correlate them 
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[0]);
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[1]);
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[2]);
	
	printf("Reading dipole\n");
	//Read the dipole
	read_dipole(intensity->bset_contr[0],&(intensity->dipole_me),intensity->dip_size);
	printf("Computing threej\n");
	//Compute threej
	precompute_threej(&(intensity->threej),jmax);
	//ijterms
	printf("Computing ijerms\n");
	compute_ijterms((intensity->bset_contr[1]),&(intensity->bset_contr[1].ijterms),intensity->molec.sym_nrepres);
	compute_ijterms((intensity->bset_contr[2]),&(intensity->bset_contr[2].ijterms),intensity->molec.sym_nrepres);
		
	//Read eigenvalues
	read_eigenvalues((*intensity));

	intensity->dimenmax = 0;
	intensity->nsizemax = 0;
	//Find nsize
	for(int i =0; i < intensity->molec.sym_nrepres; i++){
		if(intensity->isym_do[i]){
			intensity->nsizemax= max(intensity->bset_contr[1].nsize[i],intensity->nsizemax);
			intensity->nsizemax = max(intensity->bset_contr[2].nsize[i],intensity->nsizemax);
		}
	}

	printf("Biggest vector dimensiton is %i \n",intensity->nsizemax);
	intensity->dimenmax = max(intensity->bset_contr[1].Maxcontracts,intensity->dimenmax);
	intensity->dimenmax = max(intensity->bset_contr[2].Maxcontracts,intensity->dimenmax);
	printf("Biggest max contraction is is %i \n",intensity->dimenmax);
	printf("Find igamma pairs\n");
	find_igamma_pair((*intensity));
	printf("done!\n");

	
};

__host__ void dipole_initialise_gpu(FintensityJob * intensity, FGPU_ptrs & g_ptrs,int device_id){
	int jmax = max(intensity->jvals[0],intensity->jvals[1]);

	//Get available memory
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, device_id);
	g_ptrs.avail_mem = size_t(double(devProp.totalGlobalMem)*0.95);
	printf("Available gpu memory = %2.4f GB",float(g_ptrs.avail_mem)/(1024.0f*1024.0f*1024.0f));
	//Begin GPU related initalisation////////////////////////////////////////////////////////
	intensity_info int_gpu;
	//Copy over constants to GPU
	int_gpu.sym_nrepres = intensity->molec.sym_nrepres;
	int_gpu.jmax = jmax;
	int_gpu.dip_stride_1 = intensity->bset_contr[0].Maxcontracts;
	int_gpu.dip_stride_2 = intensity->bset_contr[0].Maxcontracts*intensity->bset_contr[0].Maxcontracts;
	int_gpu.dimenmax = intensity->dimenmax;
	int_gpu.sq2 = 1.0/sqrt(2.0);

	copy_array_to_gpu((void*)intensity->molec.sym_degen,(void**)&int_gpu.sym_degen,sizeof(int)*intensity->molec.sym_nrepres,"sym_degen");
	g_ptrs.avail_mem -= sizeof(int)*intensity->molec.sym_nrepres;

	CheckCudaError("Pre-initial");
	printf("Copy intensity information...");	
	copy_intensity_info(&int_gpu);
	printf("done...");
	CheckCudaError("Post-initial");
	printf("Copying bset_contrs to GPU...");
	g_ptrs.bset_contr = new cuda_bset_contrT*[2];
	create_and_copy_bset_contr_to_gpu(&intensity->bset_contr[1],&(g_ptrs.bset_contr[0]),intensity->bset_contr[1].ijterms,intensity->molec.sym_nrepres,intensity->molec.sym_degen);
	create_and_copy_bset_contr_to_gpu(&intensity->bset_contr[2],&(g_ptrs.bset_contr[1]),intensity->bset_contr[2].ijterms,intensity->molec.sym_nrepres,intensity->molec.sym_degen);

	printf("Done..");
	
	printf("Copying threej...");
	copy_threej_to_gpu(intensity->threej,&(g_ptrs.threej), jmax);
	printf("done..");

	printf("Copying dipole...");
	copy_array_to_gpu((void*)intensity->dipole_me,(void**)&(g_ptrs.dipole_me),intensity->dip_size,"dipole_me");
	printf("Done\n");
	g_ptrs.avail_mem -=intensity->dip_size;
}

__host__ void dipole_do_intensities_async_omp(FintensityJob & intensity,int device_id,int num_devices){

	cudaThreadExit(); // clears all the runtime state for the current thread
	cudaSetDevice(device_id); //Set the device name
	//Wake up the gpu//
	//printf("Wake up gpu\n");
	cudaFree(0);
	//printf("....Done!\n");



	int nJ = 2;
	//Setup the gpu pointers
	FGPU_ptrs g_ptrs;
	dipole_initialise_gpu(&intensity,g_ptrs,device_id); // Initialise the gpu pointers
	//Prinf get available cpu memory
	//unsigned long available_cpu_memory = intensity.cpu_memory;
	size_t available_gpu_memory = g_ptrs.avail_mem;
	printf("Available gpu memory = %2.4f GB\n",float(available_gpu_memory)/(1024.0f*1024.0f*1024.0f));
	//Compute how many inital state vectors and final state vectors
	//unsigned long no_final_states_cpu = ((available_cpu_memory)/8l - long(2*intensity.dimenmax))/(3l*intensity.dimenmax);//(Initial + vec_cor + half_ls)*dimen_max
	size_t no_final_states_gpu = available_gpu_memory/sizeof(double);
	no_final_states_gpu -=	size_t(intensity.dimenmax)*( 1+ nJ*intensity.molec.sym_maxdegen);
	no_final_states_gpu /= ( 2l * size_t(intensity.dimenmax) );
	no_final_states_gpu /=2;
	printf("%d\n",no_final_states_gpu);
	no_final_states_gpu = min((unsigned int )intensity.Neigenlevels,(unsigned int )no_final_states_gpu);

	printf("%d\n",no_final_states_gpu);
	//printf("Memory=%.f KB\n",float(no_final_states_gpu*sizeof(double))/float(1024));
	//no_final_states_gpu/=sizeof(double);
	//printf("%d\n",no_final_states_gpu);

	//exit(0);


	//Print out the header
	// write(out,"(/t4'J',t6'Gamma <-',t17'J',t19'Gamma',t25'Typ',t35'Ei',t42'<-',t50'Ef',t62'nu_if',&
       //           &t85,<nclasses>(4x),1x,<nmodes>(4x),3x,'<-',14x,<nclasses>(4x),1x,<nmodes>(4x),&
        //          &8x,'S(f<-i)',10x,'A(if)',12x,'I(f<-i)',12x,'Ni',8x,'Nf',8x,'N')")



	



//	exit(0);
	//half are the eigenvectors, the other half are the correlations

	//printf("No of final states in gpu_memory: %d  cpu memory: %d\n",no_final_states_gpu,no_final_states_cpu);



	//Half linestrength related variable
	cudaStream_t* st_half_ls = new cudaStream_t[nJ*intensity.molec.sym_maxdegen]; 	//Concurrently run half_ls computations on this many of the half_ls's
	double* half_ls = new double[intensity.dimenmax*nJ*intensity.molec.sym_maxdegen]; // half_ls(dimen,indF,ndeg)
	double* gpu_half_ls;
	//Create initial vector holding point
	double* initial_vector = new double[intensity.dimenmax];
	double* gpu_initial_vector;
	


	//Final vectors
	//Streams for each final vector computation
	cudaStream_t* st_ddot_vectors = new cudaStream_t[no_final_states_gpu];
	cudaStream_t intial_f_memcpy;
	double* final_vectors;
	cudaMallocHost(&final_vectors,sizeof(double)*intensity.dimenmax*no_final_states_gpu);
	//= new double[intensity.dimenmax*no_final_states_gpu]; //Pin this memory in final build
	//int* vec_ilevelF = new int[no_final_states_gpu];

	double* gpu_corr_vectors;
	double* gpu_final_vectors;

	int** vec_ilevel_buff = new int*[2];
	vec_ilevel_buff[0] = new int[no_final_states_gpu];
	vec_ilevel_buff[1] = new int[no_final_states_gpu];

	double* line_str = new double[no_final_states_gpu*intensity.molec.sym_maxdegen*intensity.molec.sym_maxdegen];
	double* gpu_line_str;
	//Track which vectors we are using
	int vector_idx=0;
	int vector_count=0;	
	int ilevel_total=0;
	int ilevelF=0,start_ilevelF=0;

	//Copy them to the gpu
	copy_array_to_gpu((void*)initial_vector,(void**)&(gpu_initial_vector),sizeof(double)*intensity.dimenmax,"gpu_initial_vector");
	available_gpu_memory -= sizeof(double)*intensity.dimenmax;

	copy_array_to_gpu((void*)final_vectors,(void**)&(gpu_final_vectors),sizeof(double)*intensity.dimenmax*no_final_states_gpu,"gpu_final_vectors");
	CheckCudaError("gpu_final_vectors");
	available_gpu_memory -= sizeof(double)*intensity.dimenmax*no_final_states_gpu;
	copy_array_to_gpu((void*)final_vectors,(void**)&(gpu_corr_vectors),sizeof(double)*intensity.dimenmax*no_final_states_gpu,"gpu_corr_vectors");
	CheckCudaError("gpu_corr_vectors");
	available_gpu_memory -= sizeof(double)*intensity.dimenmax*no_final_states_gpu;
	copy_array_to_gpu((void*)half_ls,(void**)&(gpu_half_ls),sizeof(double)*intensity.dimenmax*nJ*intensity.molec.sym_maxdegen,"gpu_half_ls");
	available_gpu_memory -= sizeof(double)*intensity.dimenmax*nJ*intensity.molec.sym_maxdegen;
	CheckCudaError("gpu_half_ls");
	printf("Available gpu memory = %.f Number of states = %d\n",float(available_gpu_memory)/float((1024l*1024l*1024l)),available_gpu_memory);
	//copy_array_to_gpu((void*)line_str,(void**)&(gpu_line_str),sizeof(double)*intensity.dimenmax*nJ*intensity.molec.sym_maxdegen,"gpu_line_str");
	//Open the eigenvector units
	char filename[1024];

	//Get the filename1552 bytes stack frame, 24 bytes spill stores, 24 bytes spill loads

	printf("Open vector units\n");
	FILE** eigenvec_unit = new FILE*[2*intensity.molec.sym_nrepres];
	for(int i =0; i< 2; i++){
		for(int j = 0; j < intensity.molec.sym_nrepres; j++)
		{

			sprintf(filename,j0eigen_vector_gamma_filebase,intensity.jvals[i],j+1);
			printf("Reading %s\n",filename);
			eigenvec_unit[i + j*2] = fopen(filename,"r");
			if(eigenvec_unit[i + j*2] == NULL)
			{
				printf("error opening %s \n",filename);
				exit(0);
			}
		}
	}
	
	//Initialise cublas
	cublasHandle_t handle;
	cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
		return;
	}	
	
	//Create the streams
	//Intial state
	for(int i = 0; i < intensity.molec.sym_maxdegen; i++)
		for(int j=0; j < nJ; j++)
			cudaStreamCreate(&st_half_ls[j + i*nJ]);

	//Final states
	cudaStreamCreate(&intial_f_memcpy);
	for(int i = 0; i < no_final_states_gpu; i++)
		cudaStreamCreate(&st_ddot_vectors[i]);


	int last_ilevelF= 0;
	// Number of threads in each thread block
    	int blockSize =768;

 
    	// Number of thread blocks in grid
    	int gridSize = (int)ceil((float)intensity.dimenmax/blockSize);
	

	//////Begin the computation//////////////////////////////////////////////////////////////////////////////
	CheckCudaError("Initialisation");

//(/t4'J',t6'Gamma <-',t17'J',t19'Gamma',t25'Typ',t35'Ef',t42'<-',t50'Ei',t62'nu_if',&
//                   &t85,<nclasses>(4x),1x,<nmodes>(4x),3x,'<-',14x,<nclasses>(4x),1x,<nmodes>(4x),&
//                   &8x,'S(f<-i)',10x,'A(if)',12x,'I(f<-i)',12x,'Ni',8x,'Nf',8x,'N')"

	//printf("Nu_if\tJf Kf quantaF\t <-- \tJI KI tauI quantaI\t Ein_A\tLine_str\n");

	//If zero then itll progress normally otherwise with 4 devices it will go like this
	//Thread 0 = 0 4 8 12
	//Thread 1 = 1 5 9 13
	//Thread 2 = 2 6 10 14
	//Thread 3 = 3 7 11 15
	//Run

	#pragma omp barrier
	if(device_id==0){
		printf("Linestrength S(f<-i) [Debye**2], Transition moments [Debye],Einstein coefficient A(if) [1/s],and Intensities [cm/mol]\n\n\n");
	}

	#pragma omp barrier
	//constants
	double beta = planck * vellgt / (boltz * intensity.temperature);
	double boltz_fc=0.0;
	double absorption_int = 0.0;

	for(int ilevelI = device_id; ilevelI < intensity.Neigenlevels; ilevelI+=num_devices){
		//printf("new I level!\n");
		//Get the basic infor we need
	      int indI = intensity.eigen[ilevelI].jind;

	      int dimenI = intensity.bset_contr[indI+1].Maxcontracts;

	      int jI = intensity.eigen[ilevelI].jval;
	      double energyI = intensity.eigen[ilevelI].energy;
	      int igammaI  = intensity.eigen[ilevelI].igamma;
	      int * quantaI = intensity.eigen[ilevelI].quanta;
	      int * normalI = intensity.eigen[ilevelI].normal;
	      int ndegI   = intensity.eigen[ilevelI].ndeg;
	      int nsizeI = intensity.bset_contr[indI+1].nsize[igammaI];

	      FILE* unitI = eigenvec_unit[ indI + (igammaI)*2]; 
		//Check filters
		
	      if(!energy_filter_lower(intensity,jI,energyI,quantaI)) continue;
		//If success then read
	      fseek(unitI,(intensity.eigen[ilevelI].irec[0]-1)*nsizeI*sizeof(double),SEEK_SET);
	      fread(initial_vector,sizeof(double),nsizeI,unitI);

	      stat = cublasSetVector(intensity.dimenmax, sizeof(double),initial_vector, 1, gpu_initial_vector, 1);
	      CheckCudaError("Set Vector I");			


	      //printf("State J = %i Energy = %11.4f igammaI = %i ilevelI = %i\n",jI,energyI,igammaI,ilevelI);

    	      blockSize =768;

 
    	// Number of thread blocks in grid
    	      gridSize = (int)ceil((float)dimenI/blockSize);
              //We have the vector now we compute the half_ls Asynchronously
	      for(int ideg=0; ideg < ndegI; ideg++){
			//printf("ideg=%i ndegI=%i\n",ideg,ndegI);
			device_correlate_vectors<<<gridSize,blockSize,0,st_half_ls[ideg]>>>(g_ptrs.bset_contr[indI],ideg,igammaI, gpu_initial_vector,gpu_corr_vectors + intensity.dimenmax*ideg); //This will have priority
	      
		}
		//printf("Correlate");		
	      cudaDeviceSynchronize();
	      for(int ideg=0; ideg < ndegI; ideg++){
			for(int indF =0; indF < nJ; indF++){
				//These will execute asychronously
	     			device_compute_1st_half_ls<<<gridSize,blockSize,0,st_half_ls[indF + ideg*nJ]>>>(g_ptrs.bset_contr[indI],g_ptrs.bset_contr[indF],
								   g_ptrs.dipole_me,igammaI,ideg,gpu_corr_vectors + intensity.dimenmax*ideg,g_ptrs.threej,
								   gpu_half_ls + indF*intensity.dimenmax + ideg*intensity.dimenmax*nJ);				
			}

			 //wait for the next batch
			
	      }


				vector_idx=0;
		ilevelF=0;
		int current_buff = 0;
		//printf("First half_ls");
		//While the half_ls is being computed, lets load up some final state vectors
		while(vector_idx < no_final_states_gpu && ilevelF < intensity.Neigenlevels)
		{
					   //   !
		      	int indF = intensity.eigen[ilevelF].jind;
		  //    !
			//printf("indF=%i",indF);
		  //    !dimension of the bases for the initial states
		  //    !
		   //   !
		      //!energy, quanta, and gedeneracy order of the initial state
		     // !
		      	int jF = intensity.eigen[ilevelF].jval;
		      	double energyF = intensity.eigen[ilevelF].energy;
		      	int igammaF  = intensity.eigen[ilevelF].igamma;
		      	int * quantaF = intensity.eigen[ilevelF].quanta;
		      	int * normalF = intensity.eigen[ilevelF].normal;
		     	 int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];
			int irec = intensity.eigen[ilevelF].irec[0]-1;
			FILE* unitF = eigenvec_unit[ indF + (igammaF)*2]; 			

			ilevelF++;
			if(!energy_filter_upper(intensity,jF,energyF,quantaF)) {continue;}
			if(!intensity_filter(intensity,jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF)) continue;
 			// store the level
			vec_ilevel_buff[0][vector_idx] = ilevelF-1;
			//printf("ilevelF=%i\n",vec_ilevel_buff[0][vector_idx]);
			//Otherwise load the vector to a free slot
			fseek(unitF,irec*nsizeF*sizeof(double),SEEK_SET);
			fread(final_vectors + vector_idx*intensity.dimenmax,sizeof(double),nsizeF,unitF);
			//Increment
			vector_idx++;
		}
		vector_count = vector_idx;
		
	
		//printf("memcopy");
		//Memcopy it in one go
		cudaMemcpyAsync(gpu_final_vectors,final_vectors,sizeof(double)*intensity.dimenmax*vector_count,cudaMemcpyHostToDevice,intial_f_memcpy) 	;


		cudaDeviceSynchronize(); //Wait till we're set up

		CheckCudaError("Batch final vectors");	
		//printf("vector_count = %i\n",vector_count);

		while(vector_count != 0)
		{
			for(int i = 0; i < vector_count; i++){
				ilevelF = vec_ilevel_buff[int(current_buff)][i];
				//printf("ilevelF=%i\n",ilevelF);
				int indF = intensity.eigen[ilevelF].jind;
			      	int jF = intensity.eigen[ilevelF].jval;
			      	double energyF = intensity.eigen[ilevelF].energy;
			      	int igammaF  = intensity.eigen[ilevelF].igamma;
			      	int * quantaF = intensity.eigen[ilevelF].quanta;
			      	int * normalF = intensity.eigen[ilevelF].normal;
			     	 int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];
				int irec = intensity.eigen[ilevelF].irec[0]-1;
				int dimenF = intensity.bset_contr[indF+1].Maxcontracts;
				int ndegF   = intensity.eigen[ilevelF].ndeg;
				int idegF = 0;
				int idegI = 0;
				//for(int i = 0; i < ndeg
				//Correlate the vectors
				for(idegF = 0; idegF < ndegF; idegF++){
					device_correlate_vectors<<<gridSize,blockSize,0,st_ddot_vectors[i]>>>(g_ptrs.bset_contr[indF],idegF,igammaF, (gpu_final_vectors + i*intensity.dimenmax),gpu_corr_vectors + intensity.dimenmax*i);
					for(idegI=0; idegI < ndegI; idegI++)
						cublasDdot (handle, dimenF,gpu_corr_vectors + intensity.dimenmax*i, 1, gpu_half_ls + indF*intensity.dimenmax + idegI*intensity.dimenmax*nJ, 1, 
														&line_str[i + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen]);
				}
			
			}
			current_buff = 1-current_buff;
			vector_idx = 0;
			ilevelF++;
			//While the line_Strength is being computed, lets load up some final state vectors
			while(vector_idx < no_final_states_gpu && ilevelF < intensity.Neigenlevels)
			{
						   //   !
			      	int indF = intensity.eigen[ilevelF].jind;
			  //    !
				//printf("indF=%i",indF);
			  //    !dimension of the bases for the initial states
			  //    !
			   //   !
			      //!energy, quanta, and gedeneracy order of the initial state
			     // !
			      	int jF = intensity.eigen[ilevelF].jval;
			      	double energyF = intensity.eigen[ilevelF].energy;
			      	int igammaF  = intensity.eigen[ilevelF].igamma;
			      	int * quantaF = intensity.eigen[ilevelF].quanta;
			      	int * normalF = intensity.eigen[ilevelF].normal;
			     	 int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];
				int irec = intensity.eigen[ilevelF].irec[0]-1;
				
				FILE* unitF = eigenvec_unit[ indF + (igammaF)*2]; 			

				ilevelF++;
				if(!energy_filter_upper(intensity,jF,energyF,quantaF)) {continue;}
				if(!intensity_filter(intensity,jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF)) continue;
				 // store the level				
				vec_ilevel_buff[current_buff][vector_idx] = ilevelF-1;
				//load the vector to a free slot
				fseek(unitF,irec*nsizeF*sizeof(double),SEEK_SET);
				fread(final_vectors + vector_idx*intensity.dimenmax,sizeof(double),nsizeF,unitF);
				//cudaMemcpyAsync(gpu_final_vectors,final_vectors,sizeof(double)*intensity.dimenmax*vector_count,cudaMemcpyHostToDevice,st_ddot_vectors[vector_idx]) ;
				//Increment
				vector_idx++;
			}
			last_ilevelF=ilevelF;

			cudaDeviceSynchronize();
			cudaMemcpyAsync(gpu_final_vectors,final_vectors,sizeof(double)*intensity.dimenmax*vector_count,cudaMemcpyHostToDevice,intial_f_memcpy) ;
			//We'e done now lets output
			for(int ivec = 0; ivec < vector_count; ivec++)
			{
				ilevelF = vec_ilevel_buff[1-current_buff][ivec];
				//printf("ilevelF=%i\n",ilevelF);
				int indF = intensity.eigen[ilevelF].jind;
			      	int jF = intensity.eigen[ilevelF].jval;
			      	double energyF = intensity.eigen[ilevelF].energy;
			      	int igammaF  = intensity.eigen[ilevelF].igamma;
			      	int * quantaF = intensity.eigen[ilevelF].quanta;
			      	int * normalF = intensity.eigen[ilevelF].normal;
				int ndegF   = intensity.eigen[ilevelF].ndeg;
				double ls=0.0;
			        for(int idegF=0; idegF < ndegF; idegF++){
				      for(int idegI=0; idegI < ndegI; idegI++){
						ls +=line_str[ivec + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen]*line_str[ivec + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen];
						
					}
				}
				ls /= double(ndegI);
				double final_ls = ls;
				double nu_if = energyF - energyI; 
             			boltz_fc = abs(nu_if) * exp(-(energyI-intensity.ZPE) * beta) * (1.0 - exp(-abs(nu_if) * beta))/ intensity.q_stat;
				//Print intensitys
				//printf("line_str %11.4e\n",line_str);
				double A_einst = A_coef_s_1*double((2*jI)+1)*final_ls*pow(abs(nu_if),3);
				final_ls = final_ls * intensity.gns[igammaI] * double( (2*jI + 1)*(2 * jF + 1) );
				absorption_int = final_ls * intens_cm_mol * boltz_fc;
				//if(final_ls < intensity.thresh_linestrength) continue;
	/*
				printf("%11.4f\t(%i %i ) ( ",nu_if,jF,intensity.eigen[ilevelF].krot);

				for(int i = 0; i < intensity.molec.nmodes+1; i++)
					printf("%i ",quantaF[i]);

				printf(")\t <-- \t(%i %i ) ",jI,intensity.eigen[ilevelI].krot);

				for(int i = 0; i < intensity.molec.nmodes+1; i++)
					printf("%i ",quantaI[i]);	

				printf("\t %16.8e    %16.8e %16.8e\n",A_einst,final_ls,orig_ls);	
*/

			/*	               write(out, "( (i4, 1x, a4, 3x),'<-', (i4, 1x, a4, 3x),a1,&
                            &(2x, f11.4,1x),'<-',(1x, f11.4,1x),f11.4,2x,&
                            &'(',1x,a3,x,i3,1x,')',1x,'(',1x,<nclasses>(x,a3),1x,<nmodes>(1x, i3),1x,')',1x,'<- ',   &
                            &'(',1x,a3,x,i3,1x,')',1x,'(',1x,<nclasses>(x,a3),1x,<nmodes>(1x, i3),1x,')',1x,   &
                            & 3(1x, es16.8),2x,(1x,i6,1x),'<-',(1x,i6,1x),i8,1x,i8,&
                            1x,'(',1x,<nmodes>(1x, i3),1x,')',1x,'<- ',1x,'(',1x,<nmodes>(1x, i3),1x,')',1x,& 
                            <nformat>(1x, es16.8))")  &
                            !
                            jF,sym%label(igammaF),jI,sym%label(igammaI),branch, &
                            energyF-intensity%ZPE,energyI-intensity%ZPE,nu_if,                 &
                            eigen(ilevelF)%cgamma(0),eigen(ilevelF)%krot,&
                            eigen(ilevelF)%cgamma(1:nclasses),eigen(ilevelF)%quanta(1:nmodes), &
                            eigen(ilevelI)%cgamma(0),eigen(ilevelI)%krot,&
                            eigen(ilevelI)%cgamma(1:nclasses),eigen(ilevelI)%quanta(1:nmodes), &
                            linestr,A_einst,absorption_int,&
                            eigen(ilevelF)%ilevel,eigen(ilevelI)%ilevel,&
                            itransit,istored(ilevelF),normalF(1:nmodes),normalI(1:nmodes),&
                            linestr_deg(1:ndegI,1:ndegF)
			*/	
			   #pragma omp critical(output_ls)
			   {
			   printf("%4i %4s   <-%4i %4s   %1s  %11.4f <- %11.4f %11.4f  ( %3s %3i ) ( ",jF,intensity.molec.c_sym[igammaF],jI,intensity.molec.c_sym[igammaI],branch(jF,jI),energyF-intensity.ZPE,energyI-intensity.ZPE,abs(nu_if),intensity.eigen[ilevelF].cgamma[0],intensity.eigen[ilevelF].krot);
			   for(int i = 1; i <= intensity.molec.nclasses; i++)
					printf(" %3s",intensity.eigen[ilevelF].cgamma[i]);
			   printf(" ");
			   for(int i = 1; i <= intensity.molec.nmodes; i++)
					printf(" %3i",quantaF[i]);
			   printf(" ) <- ( %3s %3i ) ( ",intensity.eigen[ilevelI].cgamma[0],intensity.eigen[ilevelI].krot);

			   for(int i = 1; i <= intensity.molec.nclasses; i++)
					printf(" %3s",intensity.eigen[ilevelI].cgamma[i]);
			   printf(" ");
			   for(int i = 1; i <= intensity.molec.nmodes; i++)
					printf(" %3i",quantaI[i]);
			   printf(")  %16.8e %16.8e %16.8e   %6i <- %6i %8i %8i ( ",final_ls,A_einst,absorption_int,intensity.eigen[ilevelF].ilevel,intensity.eigen[ilevelI].ilevel,0,0);
			   
			   for(int i = 1; i <= intensity.molec.nmodes; i++)
					printf(" %3i",normalF[i]);
			   printf(" ) <-  ( ");

			   for(int i = 1; i <= intensity.molec.nmodes; i++)
					printf(" %3i",normalI[i]);
			   printf(" ) ");
			   //printf(" )  %16.9e\n",1.23456789);	
			   for(int idegF=0; idegF < ndegF; idegF++){
				 for(int idegI=0; idegI < ndegI; idegI++){
						printf(" %16.9e",line_str[ivec + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen]);
						
					}
				}		
			    printf("\n");
			   }


			}			 
			ilevelF=last_ilevelF+1;
			//Save the new vector_count
			vector_count = vector_idx;
			//printf("Vector_count=%i, ilevelF=%i!\n",vector_count,ilevelF);
			//Copy the bunch of new vectors to the gpu
			
			//printf("Done!\n");
			CheckCudaError("Compute final vectors");
			cudaDeviceSynchronize();
		}
		//cudaDeviceReset();
		//cudaDeviceSynchronize();
		//exit(0);
		
	

	}
	cudaDeviceReset();
	cudaFreeHost(&final_vectors);

	

}

