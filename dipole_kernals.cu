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

__global__ void device_clear_vector(double* vec){
	int irootI = blockIdx.x * blockDim.x + threadIdx.x;
	if(irootI < int_info.dimenmax)
		vec[irootI] = 0.0;
}

__global__ void device_correlate_vectors(cuda_bset_contrT* bset_contr,int idegI,int igammaI,const double* vecI,double* vec)
{
/*
              do irootI = 1,dimenI
                 !
                 irow = bset_contr(indI)%icontr2icase(irootI,1)
                 ib   = bset_contr(indI)%icontr2icase(irootI,2)
                 !
                 iterm = ijterm(indI)%kmat(irow,igammaI)
                 !
                 dtemp0 = 0
                 !
                 nelem = bset_contr(indI)%irr(igammaI)%N(irow)
                 !
                 do ielem = 1,nelem
                    !
                    isrootI = iterm+ielem 
                    !
                    dtemp0 = dtemp0 + vecI(isrootI)*bset_contr(indI)%irr(igammaI)%repres(isrootI,idegI,ib)
                    !
                 enddo
                 !
                 vec(irootI) = dtemp0
                 !
              enddo
*/

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
__global__ void device_compute_1st_half_ls(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,int igammaI,int idegI,double* vector,double* threej,double* half_ls)
{

	int irootF = blockIdx.x * blockDim.x + threadIdx.x;
	//double sq2 = 1.0/sqrt(2.0);
	
	int dimenI,dimenF,  icontrF, icontrI,kF, kI, tauF, tauI,sigmaF, sigmaI, ktau,dipole_idx,jI,jF;
	//These are o remove if statements
	bool kI_kF_diff,kI_kF_eq,tauF_tauI_neq,kI_kF_zero;
	double ls = 0.0,f3j=0.0;
	
	dimenI = bset_contrI->Maxcontracts;
	dimenF = bset_contrF->Maxcontracts;
	jI = bset_contrI->jval;
	jF = bset_contrF->jval;
	
	if(irootF < dimenF)
	{
		half_ls[irootF] = 0.0;
	
		//If we are out of range the we always acces the zeroth element
		icontrF = bset_contrF->iroot_correlat_j0[irootF];
	
		ktau = bset_contrF->ktau[irootF];
		tauF  =  fmodf((float)ktau,2.0f);
		kF = bset_contrF->k[irootF];

		sigmaF = fmodf((float)kF, 3.0f)*tauF;
		//Possible remove this for loop all together
		for(int irootI = 0; irootI < dimenI; irootI++)
		{

		        
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
		          //dipole_idx= 0;

		          //dipole_idx+=2*(kI_kF_eq);
			  //dipole_idx+=(!kI_kF_eq)*(tauF_tauI_neq)*0;
			  //dipole_idx+=(!kI_kF_eq)*(!tauF_tauI_neq)*1;
		          
			  dipole_idx=2*(kI_kF_eq)+(!kI_kF_eq)*(tauF_tauI_neq)*0 + (!kI_kF_eq)*(!tauF_tauI_neq)*1;
			

		          //ls = 0.0;
			    ls = double(tauF-tauI)*double(kI_kF_eq) + (tauF-tauI)*(kF-kI)*(!kI_kF_eq)*( tauF_tauI_neq) + -1*(!kI_kF_eq)*(!tauF_tauI_neq);
		      //    ls+=double(tauF-tauI)*double(kI_kF_eq);			 
			//  ls+=(tauF-tauI)*(kF-kI)*(!kI_kF_eq)*( tauF_tauI_neq);  
			//  ls+=-1*(!kI_kF_eq)*(!tauF_tauI_neq);

		          ls*=dipole_me[icontrI + icontrF*int_info.dip_stride_1 + dipole_idx*int_info.dip_stride_2]*(1.0 + (int_info.sq2 - 1.0)*double(kI_kF_zero)*(!kI_kF_eq));
//
		          //ls*=1.0 + (sq2 - 1.0)*double(kI_kF_zero)*(!kI_kF_eq);
		

			  
		          //Only contribue if in range
			  //half_ls[irootF] +=pow(-1.0,double(sigmaI+kI))*
			  half_ls[irootF]+=pow(-1.0,double(sigmaI+kI))*ls*f3j*vector[irootI]*double(kI_kF_diff);
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
		half_ls[irootF] *= pow(	-1.0	, double(sigmaF) );
	}
		
}
