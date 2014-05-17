#include "cuda_objects.cuh"
#pragma once

__host__ void copy_intensity_info(intensity_info* int_inf);
__global__ void device_correlate_vectors(cuda_bset_contrT* bset_contr,int idegI,int igammaI,const double* vecI,double* vec);
__global__ void device_compute_1st_half_ls(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,int igammaI,int idegI,double* vector,double* threej,double* half_ls);
__global__ void device_clear_vector(double* vec);