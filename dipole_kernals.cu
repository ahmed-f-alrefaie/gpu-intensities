#include "cuda_objects.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>
__constant__ intensity_info int_info;


__host__ void copy_intensity_info(intensity_info* int_inf)
{
	//void* ptr;
	//cudaGetSymbolAddress ( &ptr, int_info );

	cudaMemcpyToSymbol(int_info, (void*)int_inf, sizeof(intensity_info),0,cudaMemcpyHostToDevice);
};

__global__ void device_clear_vector(double* vec,int N){
	int irootI = blockIdx.x * blockDim.x + threadIdx.x;
	if(irootI < N)
		vec[irootI] = 0.0;
}

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}



__global__ void device_correlate_vectors(cuda_bset_contrT* bset_contr,int idegI,int igammaI,const double* vecI,double* vec)
{
	int irootI = blockIdx.x * blockDim.x + threadIdx.x;
	
	int dimenI = bset_contr->Maxcontracts;
	
	if(irootI < dimenI)
	{
	
		int irow,ib,iterm,nelem,isrootI,Ntot,sdeg;
		double dtemp0 = 0.0;
		irow = bset_contr->icontr2icase[irootI];
		ib = bset_contr->icontr2icase[irootI + bset_contr->Maxcontracts];
	
		iterm = bset_contr->ijterms[irow + igammaI*bset_contr->Maxsymcoeffs];
	
		nelem = bset_contr->N[igammaI + irow*int_info.sym_nrepres];
	
		Ntot = bset_contr->Ntotal[igammaI];
		sdeg = int_info.sym_degen[igammaI];
		double* irr = bset_contr->irr_repres[igammaI];
		
		for(int i = 0; i < nelem; i++)
		{
			isrootI = iterm+i;
			dtemp0 +=  vecI[isrootI]*irr[isrootI + idegI*Ntot + ib*sdeg*Ntot];
		}
	
		vec[irootI] = dtemp0;
	
	}


}
__global__ void device_compute_1st_half_ls(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,int igammaI,double* vector,double* threej,double* half_ls)
{

	const int irootF = blockIdx.x * blockDim.x + threadIdx.x;
	//double sq2 = 1.0/sqrt(2.0);
	
	int dimenI,icontrF, icontrI,kF, kI, tauI,tauF,sigmaF, sigmaI, ktau,dipole_idx,jI,jF;
	//These are o remove if statements
	bool kI_kF_diff,kI_kF_eq,tauF_tauI_neq,kI_kF_zero;
	double ls = 0.0,f3j=0.0,final_half_ls;
	
	dimenI = bset_contrI->Maxcontracts;
	jI = bset_contrI->jval;
	jF = bset_contrF->jval;
	
	if(irootF < bset_contrF->Maxcontracts)
	{
		final_half_ls = 0.0;
	
		//If we are out of range the we always acces the zeroth element
		icontrF = bset_contrF->iroot_correlat_j0[irootF];
	
		tauF  =  bset_contrF->ktau[irootF] & 1;//fmodf((float),2.0f);
		kF = bset_contrF->k[irootF];

		sigmaF = (kF % 3)*tauF;
		//Possible remove this for loop all together
		for(int irootI = 0; irootI < dimenI; irootI++)
		{

		        
		        kI = bset_contrI->k[irootI];
		        kI_kF_diff = fabsf(kI - kF) <= 1.0f;
		        
		        icontrI = bset_contrI->iroot_correlat_j0[irootI];
		        tauI = bset_contrI->ktau[irootI] & 1;
		        
		         sigmaI = (kI % 3)*tauI;
		
			f3j  =  threej[jI + kI*(int_info.jmax+1) + (jF - jI + 1)*(int_info.jmax+1)*(int_info.jmax+1) + (kF - kI +1)*kI_kF_diff*(int_info.jmax+1)*(int_info.jmax+1)*3];  //this is big and unwieldy
		
			
			
		          //Evaluate all conditions without branching
		          kI_kF_eq = (kF==kI); // 1 or 0
		          
		          tauF_tauI_neq = (tauF!=tauI); // 1 or  0
		          
		          kI_kF_zero = ((kI*kF) != 0); // 1 or zero       //If evaluated with CMP branch instruction then implement my own
		          
			  dipole_idx=2*(kI_kF_eq)+(!kI_kF_eq)*(tauF_tauI_neq)*0 + (!kI_kF_eq)*(!tauF_tauI_neq)*1;
			

			  ls = double(tauF-tauI)*double(kI_kF_eq) + (tauF-tauI)*(kF-kI)*(!kI_kF_eq)*( tauF_tauI_neq) + -1.0*(!kI_kF_eq)*(!tauF_tauI_neq);


		          ls*=dipole_me[icontrI + icontrF*int_info.dip_stride_1 + dipole_idx*int_info.dip_stride_2]*(1.0 + (int_info.sq2 - 1.0)*double(kI_kF_zero)*(!kI_kF_eq));
//
		

			  
		          //Only contribue if in range
			// final_half_ls+=pow(-1.0,double(sigmaI+kI))*ls*f3j*vector[irootI]*double(kI_kF_diff);
			final_half_ls+=double(2*((sigmaI+kI) & 1)-1)*ls*f3j*vector[irootI]*double(kI_kF_diff);
			 // half_ls[irootF] +=dipole_idx;//(double)kI_kF_diff;//(double)dipole_idx;
			  
/*
			if(kF==kI)
			   ls = double(tauF-tauI) * dipole_me[icontrI + icontrF*int_info.dip_stride_1 + 2*int_info.dip_stride_2];
			else if(tauF != tauI){
			    ls = double((tauF-tauI)*(kF-kI)) * dipole_me[icontrI + icontrF*int_info.dip_stride_1 + 0*int_info.dip_stride_2];
			    if(kI*kF != 0) ls*=sq2;
			}else if(tauF == tauI){
				ls = -dipole_me[icontrI + icontrF*int_info.dip_stride_1 + 1*int_info.dip_stride_2];
				 if(kI*kF != 0) ls*=sq2;
			}
			half_ls[irootF] +=pow(-1.0,double(sigmaI+kI))*ls*f3j*vector[irootI]*double(kI_kF_diff);
*/				
				

		}
	
		//
		//final_half_ls *= pow(	-1.0	, double(sigmaF) );
		final_half_ls *= double(2*((sigmaF) & 1)-1);//pow(	-1.0	, double(sigmaF) );
		half_ls[irootF] = final_half_ls;
	}
		
}

__global__ void device_compute_1st_half_ls_flipped_dipole(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,double* vector,double* threej,double* half_ls)
{

	//const int irootF = blockIdx.x * blockDim.x + threadIdx.x;
	//double sq2 = 1.0/sqrt(2.0);
	const int dimenI = bset_contrI->Maxcontracts;
	
	int icontrF,icontrI,kF, kI, tauI,tauF,sigmaF, sigmaI, jI,jF,dipole_idx;
	//These are o remove if statements
	bool kI_kF_diff,kI_kF_eq,tauF_tauI_neq,kI_kF_zero;
	double ls = 0.0,f3j=0.0,final_half_ls;
	
	
	jI = bset_contrI->jval;
	jF = bset_contrF->jval;
	
	
	for(int irootF=blockIdx.x * blockDim.x + threadIdx.x; irootF < bset_contrF->Maxcontracts; irootF+=blockDim.x*gridDim.x)
	{
		final_half_ls = 0.0;
	
		//If we are out of range the we always acces the zeroth element
		icontrF = bset_contrF->iroot_correlat_j0[irootF];
	
		tauF  =  bset_contrF->ktau[irootF] & 1;
		kF = bset_contrF->k[irootF];

		sigmaF = (kF % 3)*tauF;

		for(int irootI = 0; irootI < dimenI; irootI++)
		{
			//All non-dipole global accesses
		        icontrI = bset_contrI->iroot_correlat_j0[irootI];
		        kI = bset_contrI->k[irootI]; 
			tauI = bset_contrI->ktau[irootI] & 1;

		        kI_kF_diff = abs(kI-kF) <=1;

		        sigmaI = (kI % 3)*tauI;
			sigmaI = 2*(!(sigmaI+kI) & 1)-1;
		
			f3j  =  threej[jI + kI*(int_info.jmax) + (jF - jI + 1)*(int_info.jmax)*(int_info.jmax) + (kF - kI +1)*kI_kF_diff*(int_info.jmax)*(int_info.jmax)*3];  //this is big and unwieldy
			//if(fabsf(f3j) < 0.00000000000000001) continue;
		        //Evaluate all conditions without branching
		        kI_kF_eq = (kF==kI); // 1 or 0
		          
		        tauF_tauI_neq = (tauF!=tauI); // 1 or  0
		          
		        kI_kF_zero = ((kI*kF) != 0); // 1 or zero       //If evaluated with CMP branch instruction then implement my own

		        dipole_idx=2*(kI_kF_eq)+(!kI_kF_eq)*(tauF_tauI_neq)*0 + (!kI_kF_eq)*(!tauF_tauI_neq)*1;
			  // These accesses should be coalesed and therefore significantly faster
			ls = dipole_me[icontrF + icontrI*int_info.dip_stride_1 + dipole_idx*int_info.dip_stride_2];

			ls *= double(tauF-tauI)*double(kI_kF_eq) + (tauF-tauI)*(kF-kI)*(!kI_kF_eq)*( tauF_tauI_neq) + -1.0*(!kI_kF_eq)*(!tauF_tauI_neq);
			  
		          //Only contribue if in range
			final_half_ls+=double(sigmaI)*ls*f3j*vector[irootI]*double(kI_kF_diff)*(1.0 + (int_info.sq2 - 1.0)*double(kI_kF_zero)*(!kI_kF_eq));	
							
		}
	
		final_half_ls *= double(2*(!(sigmaF) & 1)-1);
		half_ls[irootF] = final_half_ls;
	}
		
}

__global__ void device_compute_1st_half_ls_flipped_dipole_blocks(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int startF,int endF,double* dipole_me,double* vector,double* threej,double* half_ls)
{

	//const int irootF = blockIdx.x * blockDim.x + threadIdx.x;
	//double sq2 = 1.0/sqrt(2.0);
	const int dimenI = bset_contrI->Maxcontracts;
	
	int icontrF,icontrI,kF, kI, tauI,tauF,sigmaF, sigmaI, jI,jF,dipole_idx;
	//These are o remove if statements
	bool kI_kF_diff,kI_kF_eq,tauF_tauI_neq,kI_kF_zero;
	double ls = 0.0,f3j=0.0,final_half_ls;
	
	
	jI = bset_contrI->jval;
	jF = bset_contrF->jval;
	
	
	for(int irootF=blockIdx.x * blockDim.x + threadIdx.x + startF; irootF < endF; irootF+=blockDim.x*gridDim.x)
	{
		final_half_ls = 0.0;
	
		//If we are out of range the we always acces the zeroth element
		icontrF = bset_contrF->iroot_correlat_j0[irootF];
	
		tauF  =  bset_contrF->ktau[irootF] & 1;
		kF = bset_contrF->k[irootF];

		sigmaF = (kF % 3)*tauF;

		for(int irootI = 0; irootI < dimenI; irootI++)
		{
			//All non-dipole global accesses
		        icontrI = bset_contrI->iroot_correlat_j0[irootI];
		        kI = bset_contrI->k[irootI]; 
			tauI = bset_contrI->ktau[irootI] & 1;

		        kI_kF_diff = abs(kI-kF) <=1;

		        sigmaI = (kI % 3)*tauI;
			sigmaI = 2*(!(sigmaI+kI) & 1)-1;
		
			f3j  =  threej[jI + kI*(int_info.jmax) + (jF - jI + 1)*(int_info.jmax)*(int_info.jmax) + (kF - kI +1)*kI_kF_diff*(int_info.jmax)*(int_info.jmax)*3];  //this is big and unwieldy
			//if(fabsf(f3j) < 0.00000000000000001) continue;
		        //Evaluate all conditions without branching
		        kI_kF_eq = (kF==kI); // 1 or 0
		          
		        tauF_tauI_neq = (tauF!=tauI); // 1 or  0
		          
		        kI_kF_zero = ((kI*kF) != 0); // 1 or zero       //If evaluated with CMP branch instruction then implement my own

		        dipole_idx=2*(kI_kF_eq)+(!kI_kF_eq)*(tauF_tauI_neq)*0 + (!kI_kF_eq)*(!tauF_tauI_neq)*1;
			  // These accesses should be coalesed and therefore significantly faster
			ls = dipole_me[icontrF + icontrI*(int_info.dip_stride_1/2) + dipole_idx*(int_info.dip_stride_1/2)*int_info.dip_stride_1];

			ls *= double(tauF-tauI)*double(kI_kF_eq) + (tauF-tauI)*(kF-kI)*(!kI_kF_eq)*( tauF_tauI_neq) + -1.0*(!kI_kF_eq)*(!tauF_tauI_neq);
			  
		          //Only contribue if in range
			final_half_ls+=double(sigmaI)*ls*f3j*vector[irootI]*double(kI_kF_diff)*(1.0 + (int_info.sq2 - 1.0)*double(kI_kF_zero)*(!kI_kF_eq));	
							
		}
	
		final_half_ls *= double(2*(!(sigmaF) & 1)-1);
		half_ls[irootF] = final_half_ls;

	}
		
}


__global__ void device_compute_1st_half_ls_flipped_dipole_branch(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,double* vector,double* threej,double* half_ls)
{

	//const int irootF = blockIdx.x * blockDim.x + threadIdx.x;
	//double sq2 = 1.0/sqrt(2.0);
	const int dimenI = bset_contrI->Maxcontracts;
	
	int icontrF,icontrI,kF, kI, tauI,tauF,sigmaF, sigmaI, jI,jF,dipole_idx;
	//These are o remove if statements
	bool kI_kF_diff,kI_kF_eq,tauF_tauI_neq,kI_kF_zero;
	double ls = 0.0,f3j=0.0,final_half_ls;
	
	
	jI = bset_contrI->jval;
	jF = bset_contrF->jval;
	
	
	for(int irootF=blockIdx.x * blockDim.x + threadIdx.x; irootF < bset_contrF->Maxcontracts; irootF+=blockDim.x*gridDim.x)
	{
		final_half_ls = 0.0;
	
		//If we are out of range the we always acces the zeroth element
		icontrF = bset_contrF->iroot_correlat_j0[irootF];
	
		tauF  =  bset_contrF->ktau[irootF] & 1;
		kF = bset_contrF->k[irootF];

		sigmaF = (kF % 3)*tauF;

		for(int irootI = 0; irootI < dimenI; irootI++)
		{
			//All non-dipole global accesses
		        icontrI = bset_contrI->iroot_correlat_j0[irootI];
		        kI = bset_contrI->k[irootI]; 
			tauI = bset_contrI->ktau[irootI] & 1;

		        kI_kF_diff = (((kI - kF)^(kI-kF)>>31) - ( (kI-kF) >> 31 ) ) <= 1;

		        sigmaI = (kI % 3)*tauI;
			sigmaI = 2*(!(sigmaI+kI) & 1)-1;
		
			f3j  =  threej[jI + kI*(int_info.jmax) + (jF - jI + 1)*(int_info.jmax)*(int_info.jmax) + (kF - kI +1)*kI_kF_diff*(int_info.jmax)*(int_info.jmax)*3];  //this is big and unwieldy
			//if(fabsf(f3j) < 0.00000000000000001) continue;
		        //Evaluate all conditions without branching
			if(kF==kI)
			   ls = double(tauF-tauI) * dipole_me[icontrF + icontrI*int_info.dip_stride_1 + 2*int_info.dip_stride_2];
			else if(tauF != tauI){
			    ls = double((tauF-tauI)*(kF-kI)) * dipole_me[icontrF + icontrI*int_info.dip_stride_1 + 0*int_info.dip_stride_2];
			}else if(tauF == tauI){
			    ls = -dipole_me[icontrF + icontrF*int_info.dip_stride_1 + 1*int_info.dip_stride_2];
			}
			if(kF!=kI && kF*kI!=0) ls*=int_info.sq2;
			  
		          //Only contribue if in range
			final_half_ls+=double(sigmaI)*ls*f3j*vector[irootI]*double(kI_kF_diff);	
							
		}
	
		final_half_ls *= double(2*(!(sigmaF) & 1)-1);
		half_ls[irootF] = final_half_ls;
	}
		
}



__global__ void device_compute_1st_half_ls_2D(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,int igammaI,double* vector,double* threej,double* half_ls)
{
	extern __shared__ double s[];
	int irootF = blockIdx.x * blockDim.x + threadIdx.x;
	int irootI = blockIdx.y * blockDim.y + threadIdx.y;
	double result;
	//double sq2 = 1.0/sqrt(2.0);
	
	int dimenI,dimenF,  icontrF, icontrI,kF, kI, tauF, tauI,sigmaF, sigmaI, ktau,dipole_idx,jI,jF;
	//These are o remove if statements
	bool kI_kF_diff,kI_kF_eq,tauF_tauI_neq,kI_kF_zero;
	double ls = 0.0,f3j=0.0;
	
	dimenI = bset_contrI->Maxcontracts;
	dimenF = bset_contrF->Maxcontracts;
	jI = bset_contrI->jval;
	jF = bset_contrF->jval;
	s[blockDim.x * threadIdx.y + threadIdx.x] = 0.0;
	if(irootF < dimenF & irootI < dimenI)
	{
	
	
		//If we are out of range the we always acces the zeroth element
		icontrF = bset_contrF->iroot_correlat_j0[irootF];
	
		ktau = bset_contrF->ktau[irootF];
		tauF  =  fmodf((float)ktau,2.0f);
		kF = bset_contrF->k[irootF];

		        
		kI = bset_contrI->k[irootI];
		kI_kF_diff = fabsf(kI - kF) <= 1.0f;
		        
		icontrI = bset_contrI->iroot_correlat_j0[irootI];
		ktau = bset_contrI->ktau[irootI];
		tauI = fmodf((float)ktau,2.0f);
		        
		sigmaI = fmodf((float)kI, 3.0f)*tauI;

		f3j  =  threej[jI + kI*(int_info.jmax+1) + (jF - jI + 1)*(int_info.jmax+1)*(int_info.jmax+1) + (kF - kI +1)*kI_kF_diff*(int_info.jmax+1)*(int_info.jmax+1)*3];  //this is big and unwieldy
		
			
			
		          //Evaluate all conditions without branching
		kI_kF_eq = (kF==kI); // 1 or 0
		          
		tauF_tauI_neq = (tauF!=tauI); // 1 or  0
		          
		kI_kF_zero = ((kI*kF) != 0); // 1 or zero       //If evaluated with CMP branch instruction then implement my own
		          
		dipole_idx=2*(kI_kF_eq)+(!kI_kF_eq)*(tauF_tauI_neq)*0 + (!kI_kF_eq)*(!tauF_tauI_neq)*1;
			

		ls = double(tauF-tauI)*double(kI_kF_eq) + (tauF-tauI)*(kF-kI)*(!kI_kF_eq)*( tauF_tauI_neq) + -1.0*(!kI_kF_eq)*(!tauF_tauI_neq);


		ls*=dipole_me[icontrI + icontrF*int_info.dip_stride_1 + dipole_idx*int_info.dip_stride_2]*(1.0 + (int_info.sq2 - 1.0)*double(kI_kF_zero)*(!kI_kF_eq));
		s[blockDim.x * threadIdx.y + threadIdx.x] = half_ls[irootF]+=pow(-1.0,double(sigmaI+kI))*ls*f3j*vector[irootI]*double(kI_kF_diff);
		
		__syncthreads();
			//Reduction part//////////////////////////////////Use a better method/////////////////////
		if(threadIdx.x==0 && threadIdx.y==0){	
			for(int y = 0; y < blockDim.y; y++){
				result = 0;
				for(int x = 0; x < blockDim.x; x++)
					result+=s[blockDim.x*y + x];
				atomicAdd(&half_ls[irootF],result);	//Add the result of the block
			}
		}
		
	}


		//
//	half_ls[irootF] *= pow(	-1.0	, double(sigmaF) );
		
}


////////////////////////////////////////////////////////////NEWER VERSION WHERE WE COMPUTE PER ROOTF

__global__ void device_compute_1st_half_ls_dimenF(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,int irootF,int igammaI,int idegI,double* vector,double* threej,double* half_ls)
{
	extern __shared__  double s[];
	int irootI = blockIdx.x * blockDim.x + threadIdx.x;
	s[threadIdx.x]=0.0;
	//double sq2 = 1.0/sqrt(2.0);
	
	int dimenI,dimenF,  icontrF, icontrI,kF, kI, tauF, tauI,sigmaF, sigmaI, ktau,dipole_idx,jI,jF;
	//These are o remove if statements
	bool kI_kF_diff,kI_kF_eq,tauF_tauI_neq,kI_kF_zero;
	double ls = 0.0,f3j=0.0;
	
	dimenI = bset_contrI->Maxcontracts;
	dimenF = bset_contrF->Maxcontracts;
	jI = bset_contrI->jval;
	jF = bset_contrF->jval;
	if(irootI < dimenI){
	
			//If we are out of range the we always acces the zeroth element
		icontrF = bset_contrF->iroot_correlat_j0[irootF];
	
		ktau = bset_contrF->ktau[irootF];
		tauF  =  fmodf((float)ktau,2.0f);
		kF = bset_contrF->k[irootF];

		sigmaF = fmodf((float)kF, 3.0f)*tauF;

				
		kI = bset_contrI->k[irootI];
		kI_kF_diff = fabsf(kI - kF) <= 1.0f;
				
		icontrI = bset_contrI->iroot_correlat_j0[irootI];
		ktau = bset_contrI->ktau[irootI];
		tauI = fmodf((float)ktau,2.0f);
				
		sigmaI = fmodf((float)kI, 3.0f)*tauI;
		
		f3j  =  threej[jI + kI*(int_info.jmax+1) + (jF - jI + 1)*(int_info.jmax+1)*(int_info.jmax+1) + (kF - kI +1)*kI_kF_diff*(int_info.jmax+1)*(int_info.jmax+1)*3];  //this is big and unwieldy
		
			
			
				  //Evaluate all conditions without branching
		kI_kF_eq = (kF==kI); // 1 or 0
				  
		tauF_tauI_neq = (tauF!=tauI); // 1 or  0
				  
		kI_kF_zero = ((kI*kF) != 0); // 1 or zero       //If evaluated with CMP branch instruction then implement my own
				  
		dipole_idx=2*(kI_kF_eq)+(!kI_kF_eq)*(tauF_tauI_neq)*0 + (!kI_kF_eq)*(!tauF_tauI_neq)*1;
			

		ls = double(tauF-tauI)*double(kI_kF_eq) + (tauF-tauI)*(kF-kI)*(!kI_kF_eq)*( tauF_tauI_neq) + -1.0*(!kI_kF_eq)*(!tauF_tauI_neq);


		ls*=dipole_me[icontrI + icontrF*int_info.dip_stride_1 + dipole_idx*int_info.dip_stride_2]*(1.0 + (int_info.sq2 - 1.0)*double(kI_kF_zero)*(!kI_kF_eq));
	//
		

				  
				  //Only contribue if in range
		//half_ls[irootF]+=pow(-1.0,double(sigmaI+kI))*ls*f3j*vector[irootI]*double(kI_kF_diff);
		s[threadIdx.x] = pow(-1.0,double(sigmaI+kI))*ls*f3j*vector[irootI]*double(kI_kF_diff);
	
		__syncthreads();
		//Reduction part//////////////////////////////////Use a better method/////////////////////
		if(threadIdx.x==0){
			for(int i = 1; i < blockDim.x; i++){
				s[threadIdx.x]+=s[i];
			}
			atomicAdd(&half_ls[irootF],s[threadIdx.x]);
		}
	}
		
}

__global__ void device_complete_half_ls_dimenF(cuda_bset_contrT* bset_contrF,double* half_ls){
	int irootF = blockIdx.x * blockDim.x + threadIdx.x;
	int dimenF = bset_contrF->Maxcontracts;
	if(irootF < dimenF){
		int ktau = bset_contrF->ktau[irootF];
		int tauF  =  fmodf((float)ktau,2.0f);
		int kF = bset_contrF->k[irootF];
		int sigmaF = fmodf((float)kF, 3.0f)*tauF;
		half_ls[irootF] *= pow(	-1.0	, double(sigmaF) );
	}
}
