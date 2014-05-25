#include "test.cuh"


__host__ void benchmark_half_ls(FintensityJob & intensity,int no_initial_states){


	dipole_initialise(&intensity);
	int nJ=2;
	//The intial state vector
	double* initial_vec = new double[intensity.dimenmax];

	double* gpu_initial_vec=NULL;

	copy_array_to_gpu((void*)initial_vec,(void**)&(gpu_initial_vec),sizeof(double)*intensity.dimenmax,"gpu_initial_vec");
	printf("%p\n",gpu_initial_vec);


	double* corr_vec = new double[intensity.dimenmax*2*intensity.molec.sym_maxdegen];
	double* gpu_corr_vec=NULL;

	copy_array_to_gpu((void*)corr_vec,(void**)&(gpu_corr_vec),sizeof(double)*intensity.dimenmax*2*intensity.molec.sym_maxdegen,"gpu_corr_vec");

	double* half_ls = new double[intensity.dimenmax*2*intensity.molec.sym_maxdegen];
	double* gpu_half_ls;

	copy_array_to_gpu((void*)half_ls,(void**)&(gpu_half_ls),sizeof(double)*intensity.dimenmax*2*intensity.molec.sym_maxdegen,"gpu_half_ls1");


	char filename[1024];
	//Get the filename
	printf("Open vector unit\n");
	FILE** eigenvec_unit = new FILE*[2*intensity.molec.sym_nrepres];
	for(int i =0; i< 2; i++){
		for(int j = 0; j < intensity.molec.sym_nrepres; j++)
		{
			if(intensity.isym_do[j] == false) continue;
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
    	int blockSize =384;
 
    	// Number of thread blocks in grid
    	int gridSize = (int)ceil((float)intensity.dimenmax/blockSize);


	//Testing variables
	double time=0.0,half_ls_time=0,flipped_half_ls_time=0;
	int states_done = 0;
	cudaStream_t* stream = new cudaStream_t[2*intensity.molec.sym_maxdegen];
	for(int i =0; i < intensity.molec.sym_maxdegen; i++){
		cudaStreamCreate(&stream[i*2]);
		cudaStreamCreate(&stream[1 + i*2]);
		}
	printf("Nu_if\tJf Kf quantaF\t <-- \tJI KI tauI quantaI\t Ein_A\tLine_str\n");
	int half_ls_runs =0;
	//Run
	for(int ilevelI = 0; ilevelI < intensity.Neigenlevels; ilevelI++){
	
			    //  ! start measuring time per line

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


 
    	      // Number of thread blocks in grid
              //We have the vector now we compute the half_ls

		device_correlate_vectors<<<gridSize,blockSize>>>(intensity.g_ptrs.bset_contr[indI],0,igammaI, gpu_initial_vec,gpu_corr_vec);

		time = GetTimeMs64();
	       	cudaDeviceSynchronize();
		int jF;
			for(int indF =0; indF < nJ; indF++){
				jF=intensity.jvals[indF];
				
				/*					do ilevelF = 1, nlevels
				  !
				  if (eigen(ilevelF)%jval/=jF) cycle 
				  !
				  energyF = eigen(ilevelF)%energy
				  igammaF = eigen(ilevelF)%igamma        
				  quantaF(0:nmodes) = eigen(ilevelF)%quanta(0:nmodes) 
				  !
				  call intens_filter(jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF,igamma_pair,passed)
				  !
				  if (passed) exit
				  !
				  !passed = passed_
				  !
				enddo
				*/
				if(!indF_filter(intensity,jI,jF,energyI,igammaI,quantaI))continue;
     	
			for(int ideg=0; ideg < ndegI; ideg++){
					half_ls_runs++;
				do_1st_half_ls(intensity.g_ptrs.bset_contr[indI],intensity.g_ptrs.bset_contr[indF],intensity.dimenmax,ideg,igammaI,intensity.g_ptrs.dipole_me
							, gpu_initial_vec,gpu_corr_vec+ indF*intensity.dimenmax + ideg*intensity.dimenmax*nJ,intensity.g_ptrs.threej,gpu_half_ls + indF*intensity.dimenmax + ideg*intensity.dimenmax*nJ,stream[indF + ideg*nJ]);				
			}
	      	}
		cudaDeviceSynchronize();
		CheckCudaError("Flipped half ls");

		time = GetTimeMs64()-time;
		flipped_half_ls_time += time/1000.0;	
		printf("%i - Flipped half_ls done in: %11.4fs\n",states_done,time/1000.0);
/*		
		time = GetTimeMs64();

	       cudaDeviceSynchronize();
	       for(int ideg=0; ideg < ndegI; ideg++){
			for(int indF =0; indF < nJ; indF++){
		
				do_1st_half_ls_branch(intensity.g_ptrs.bset_contr[indI],intensity.g_ptrs.bset_contr[indF],intensity.dimenmax,ideg,igammaI,intensity.g_ptrs.dipole_me
							, gpu_initial_vec,gpu_corr_vec+ indF*intensity.dimenmax + ideg*intensity.dimenmax*nJ,intensity.g_ptrs.threej,gpu_half_ls + indF*intensity.dimenmax + ideg*intensity.dimenmax*nJ,stream[indF + ideg*nJ]);				
			}
	      }
		cudaDeviceSynchronize();
		CheckCudaError("Branch");

		time = GetTimeMs64()-time;
		printf("%i - Branch half_ls done in: %11.4fs\n",states_done,time/1000.0);
		half_ls_time += time/1000.0;
*/		
		

		CheckCudaError("First run");
		states_done++;
		if(states_done >= no_initial_states) break;

		
		
		


	}
		printf("State stats-  largest dimension: %i number of degeneracies: %i\n",intensity.dimenmax,intensity.molec.sym_maxdegen) ;
		printf("-----------------Time results---------------------\n");
		printf("Branch - Total time: %11.4fs Average Time per state: %11.4fs\n",half_ls_time,half_ls_time/double(no_initial_states));
		printf("Flipped - Total time: %11.4fs Average Time per state: %11.4fs Average time per call = %11.4f\n",flipped_half_ls_time,flipped_half_ls_time/double(no_initial_states) ,flipped_half_ls_time/(double(no_initial_states)*half_ls_runs));
		printf("Total states completed: %i\n",states_done);
	
		cudaDeviceReset();

	

		
	
};

