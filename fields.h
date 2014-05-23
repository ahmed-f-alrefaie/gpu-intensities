#include "trove_objects.h"
#include "cuda_objects.cuh"
#include <cstring>
#pragma once


struct Fmole_type{
	//number of modes
	int nmodes;
	int nclasses;

	//Other info
	int sym_nrepres;
	int* sym_degen;
	int sym_maxdegen;
	char** c_sym;
};	//Intensity job information

struct FGPU_ptrs{
	double* dipole_me;
	double* threej;
	cuda_bset_contrT** bset_contr;
	size_t avail_mem;
	
};


struct FintensityJob{		
	double ZPE;
	double erange[2];
	double erange_lower[2];
	double erange_upper[2];
	double freq_window[2];
	int jvals[2];
	double q_stat;
	double temperature;
	bool* isym_do;
	int* isym_pairs;
	double* gns;
	Fmole_type molec;
	TO_bset_contrT bset_contr[3];
	TO_PTeigen* eigen;
	int Neigenlevels, Neigenroots;
	double* dipole_me;
	size_t dip_size;
	double* threej;
	FGPU_ptrs g_ptrs;
	int* igamma_pair;
	unsigned int dimenmax;
	double thresh_linestrength;
	unsigned int nsizemax;
	int** quanta_lower;
	int** quanta_upper;
	bool host_dipole;
	//Will b replaced with something better
	unsigned long cpu_memory;
	unsigned long gpu_memory;

};

void read_fields(FintensityJob* intensity);
