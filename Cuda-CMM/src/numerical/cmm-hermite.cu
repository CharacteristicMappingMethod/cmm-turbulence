/******************************************************************************************************************************
*
*	This script is part of the code for the characteristic mapping method in 2D with particle flow
*	written in C++ (C) using Nvidia CUDA on Linux.
*
*   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
*   and distribute verbatim copies of this license document, but changing it is not allowed.
*
*   Documentation and further information can be taken from the GitHub page, located at:
*   https://github.com/Arcadia197/cmm-turbulence
*
******************************************************************************************************************************/

#include "cmm-hermite.h"
#include "stdio.h"
#include "../grid/cmm-grid2d.h"

/*******************************************************************
*						  Hermite interpolation					   *
*******************************************************************/
// this computation is used very often, using it as a function greatL.y reduces code-redundancy
// however, here we have 8 scattered memory accesses where I[0] - I[1] and I[2] - I[3] are paired most of the time (in theory)

// for reference in 2D:
//return b[0][0]* H[    I[0]] + b[0][1]* H[    I[1]] + b[1][0]* H[    I[2]] + b[1][1]* H[    I[3]] // Point interpolation
//	    + (b[0][2]* H[1*N+I[0]] + b[0][3]* H[1*N+I[1]] + b[1][2]* H[1*N+I[2]] + b[1][3]* H[1*N+I[3]]) * (h)  // dx
//	    + (b[2][0]* H[2*N+I[0]] + b[2][1]* H[2*N+I[1]] + b[3][0]* H[2*N+I[2]] + b[3][1]* H[2*N+I[3]]) * (h)  // dy
//	    + (b[2][2]* H[3*N+I[0]] + b[2][3]* H[3*N+I[1]] + b[3][2]* H[3*N+I[2]] + b[3][3]* H[3*N+I[3]]) * (h*h);  // dx dy

template<typename T>
__device__ T device_hermite_mult_2D(T *H, T b[][4], int I[], long long int N, T hx, T hy)
{
	T h_out = (T)0.0;
	for (int i_ord = 0; i_ord < 4; ++i_ord) {
		T h_temp = (T)0.0;
		int2 ind_ord = make_int2(2 * ((i_ord >> 1) & 1), 2*(i_ord & 1));  // (i/2)%2 and i%2
		for (int i_p = 0; i_p < 4; ++i_p) {
			h_temp += b[((i_p >> 1) & 1) + ind_ord.x][(i_p & 1) + ind_ord.y] * H[i_ord*N + I[i_p]];
		}
		if ((i_ord & 1) == 1) 		 h_temp *= hx;
		if (((i_ord >> 1) & 1) == 1) h_temp *= hy;
		h_out += h_temp;
	}
	return h_out;
}
template<typename T>
__device__ T device_hermite_mult_3D(T *H, T b[][4][4], int I[], long long int N, T hx, T hy, T hz)
{
	T h_out = (T)0.0;
	for (int i_ord = 0; i_ord < 8; ++i_ord) {
		T h_temp = (T)0.0;
		int3 ind_ord = make_int3(2 * ((i_ord >> 2) & 1), 2 * ((i_ord >> 1) & 1), 2*(i_ord & 1));  // (i/4)%2, (i/2)%2 and i%2
		for (int i_p = 0; i_p < 8; ++i_p) {
			h_temp += b[((i_p >> 2) & 1) + ind_ord.x][((i_p >> 1) & 1) + ind_ord.y][(i_p & 1) + ind_ord.z]
					* H[i_ord*N + I[i_p]];
		}
		if ((i_ord & 1) == 1) 		 h_temp *= hx;
		if (((i_ord >> 1) & 1) == 1) h_temp *= hy;
		if (((i_ord >> 2) & 1) == 1) h_temp *= hz;
		h_out += h_temp;
	}
	return h_out;
}


template<typename T>
__device__ T device_hermite_mult_2D_warp(T *H, T b[][4], int I[], int I_w[], T L, long long int N, T hx, T hy)
{
	T h_out = (T)0.0;
	for (int i_ord = 0; i_ord < 4; ++i_ord) {
		T h_temp = (T)0.0;
		int2 ind_ord = make_int2(2 * ((i_ord >> 1) & 1), 2*(i_ord & 1));  // (i_ord / 2) % 2 and (i_ord) % 2
		for (int i_p = 0; i_p < 4; ++i_p) {
			if (i_ord == 0) {
				h_temp += b[((i_p >> 1) & 1) + ind_ord.x][(i_p & 1) + ind_ord.y] * (H[I[i_p]] + I_w[i_p]*L);
			}
			else {
				h_temp += b[((i_p >> 1) & 1) + ind_ord.x][(i_p & 1) + ind_ord.y] * H[i_ord*N + I[i_p]];
			}
		}
		if ((i_ord & 1) == 1) 		 h_temp *= hx;
		if (((i_ord >> 1) & 1) == 1) h_temp *= hy;
		h_out += h_temp;
	}
	return h_out;
}
template<typename T>
__device__ T device_hermite_mult_3D_warp(T *H, T b[][4][4], int I[], int I_w[], T L, long long int N, T hx, T hy, T hz)
{
	T h_out = (T)0.0;
	for (int i_ord = 0; i_ord < 8; ++i_ord) {
		T h_temp = (T)0.0;
		int3 ind_ord = make_int3(2 * ((i_ord >> 2) & 1), 2 * ((i_ord >> 1) & 1), 2*(i_ord & 1));  // (i/4)%2, (i/2)%2 and i%2
		for (int i_p = 0; i_p < 8; ++i_p) {
			if (i_ord == 0) {
				h_temp += b[((i_p >> 2) & 1) + ind_ord.x][((i_p >> 1) & 1) + ind_ord.y][(i_p & 1) + ind_ord.z]
						* (H[I[i_p]] + I_w[i_p]*L);
			}
			else {
				h_temp += b[((i_p >> 2) & 1) + ind_ord.x][((i_p >> 1) & 1) + ind_ord.y][(i_p & 1) + ind_ord.z]
						* H[i_ord*N + I[i_p]];
			}
		}
		if ((i_ord & 1) == 1) 		 h_temp *= hx;
		if (((i_ord >> 1) & 1) == 1) h_temp *= hy;
		if (((i_ord >> 2) & 1) == 1) h_temp *= hz;
		h_out += h_temp;
	}
	return h_out;
}


template<typename T>
__device__ T device_hermite_mult_2D(T *H, T bX[], T bY[], int I[], long long int N, T hx, T hy)
{
	T h_out = (T)0.0;
	for (int i_ord = 0; i_ord < 4; ++i_ord) {
		T h_temp = (T)0.0;
		int2 ind_ord = make_int2(2 * ((i_ord >> 1) & 1), 2*(i_ord & 1));  // (i/2)%2 and i%2
		for (int i_p = 0; i_p < 4; ++i_p) {
			h_temp += bY[((i_p >> 1) & 1) + ind_ord.x] * bX[(i_p & 1) + ind_ord.y] * H[i_ord*N + I[i_p]];
		}
		if ((i_ord & 1) == 1) 		 h_temp *= hx;
		if (((i_ord >> 1) & 1) == 1) h_temp *= hy;
		h_out += h_temp;
	}
	return h_out;
}
// save memory by not storing the matrix b but computing it in the function
template<typename T>
__device__ T device_hermite_mult_3D(T *H, T bX[], T bY[], T bZ[], int I[], long long int N, T hx, T hy, T hz)
{
	T h_out = (T)0.0;
	for (int i_ord = 0; i_ord < 8; ++i_ord) {
		T h_temp = (T)0.0;
		int3 ind_ord = make_int3(2*(i_ord & 1), 2 * ((i_ord >> 1) & 1), 2 * ((i_ord >> 2) & 1));  // (i/4)%2, (i/2)%2 and i%2
		for (int i_p = 0; i_p < 4; ++i_p) {
			h_temp += bX[(i_p& 1) + ind_ord.x] * bY[((i_p >> 1) & 1) + ind_ord.y] * bZ[((i_p >> 2) & 1) + ind_ord.z]
					* H[i_ord*N + I[i_p]];
		}
		if ((i_ord & 1) == 1) 		 h_temp *= hx;
		if (((i_ord >> 1) & 1) == 1) h_temp *= hy;
		if (((i_ord >> 2) & 1) == 1) h_temp *= hz;
		h_out += h_temp;
	}
	return h_out;
}


// combine the build of the index vector to reduce redundant code
template<typename T>
__device__ void device_init_ind(int *I, T *dxy, T x, T y, TCudaGrid2D Grid) {
	//cell index
	int Ix0 = floor((x-Grid.bounds[0])/Grid.hx); int Iy0 = floor((y-Grid.bounds[2])/Grid.hy);
	int Ix1 = Ix0 + 1; int Iy1 = Iy0 + 1;

	//dx, dy
	dxy[0] = (x-Grid.bounds[0])/Grid.hx - Ix0; dxy[1] = (y-Grid.bounds[2])/Grid.hy - Iy0;

	// project into domain, < bounds[0/2] check is needed as integer division rounds towards 0
	Ix0 -= (Ix0/Grid.NX - (Ix0 < 0))*Grid.NX; Iy0 -= (Iy0/Grid.NY - (Iy0 < 0))*Grid.NY;
	Ix1 -= (Ix1/Grid.NX - (Ix1 < 0))*Grid.NX; Iy1 -= (Iy1/Grid.NY - (Iy1 < 0))*Grid.NY;

	// I00, I10, I01, I11 in Vector to shorten function calls
	I[0] = Iy0 * Grid.NX + Ix0; I[1] = Iy0 * Grid.NX + Ix1; I[2] = Iy1 * Grid.NX + Ix0; I[3] = Iy1 * Grid.NX + Ix1;
}
template<typename T>
__device__ void device_init_ind(int *I, T *dxyz, T x, T y, T z, TCudaGrid2D Grid) {
	//cell index
	int Ix0 = floor((x-Grid.bounds[0])/Grid.hx);
	int Iy0 = floor((y-Grid.bounds[2])/Grid.hy);
	int Iz0 = floor((z-Grid.bounds[4])/Grid.hz);
	int Ix1 = Ix0 + 1; int Iy1 = Iy0 + 1; int Iz1 = Iz0 + 1;

	//dx, dy, dz
	dxyz[0] = (x-Grid.bounds[0])/Grid.hx - Ix0;
	dxyz[1] = (y-Grid.bounds[2])/Grid.hy - Iy0;
	dxyz[1] = (z-Grid.bounds[4])/Grid.hz - Iz0;

	// project into domain, < bounds[0/2] check is needed as integer division rounds towards 0
	Ix0 -= (Ix0/Grid.NX - (Ix0 < 0))*Grid.NX; Iy0 -= (Iy0/Grid.NY - (Iy0 < 0))*Grid.NY; Iz0 -= (Iz0/Grid.NZ - (Iz0 < 0))*Grid.NZ;
	Ix1 -= (Ix1/Grid.NX - (Ix1 < 0))*Grid.NX; Iy1 -= (Iy1/Grid.NY - (Iy1 < 0))*Grid.NY; Iz1 -= (Iz1/Grid.NZ - (Iz1 < 0))*Grid.NZ;

	// I000, I100, I010, I110, I001, I101, I011, I111 in Vector to shorten function calls
	I[0] = Iz0 * Grid.NX*Grid.NY + Iy0 * Grid.NX + Ix0; I[1] = Iz0 * Grid.NX*Grid.NY + Iy0 * Grid.NX + Ix1;
	I[2] = Iz0 * Grid.NX*Grid.NY + Iy1 * Grid.NX + Ix0; I[3] = Iz0 * Grid.NX*Grid.NY + Iy1 * Grid.NX + Ix1;
	I[4] = Iz1 * Grid.NX*Grid.NY + Iy0 * Grid.NX + Ix0; I[5] = Iz1 * Grid.NX*Grid.NY + Iy0 * Grid.NX + Ix1;
	I[6] = Iz1 * Grid.NX*Grid.NY + Iy1 * Grid.NX + Ix0; I[7] = Iz1 * Grid.NX*Grid.NY + Iy1 * Grid.NX + Ix1;
}
template<typename T>
__device__ void device_init_ind_diff(int *I, int *I_w, T *dxy, T x, T y, TCudaGrid2D Grid) {
	// cell index
	int Ix0 = floor((x-Grid.bounds[0])/Grid.hx); int Iy0 = floor((y-Grid.bounds[2])/Grid.hy);
	int Ix1 = Ix0 + 1; int Iy1 = Iy0 + 1;

	//dx, dy
	dxy[0] = (x-Grid.bounds[0])/Grid.hx - Ix0; dxy[1] = (y-Grid.bounds[2])/Grid.hy - Iy0;

	//warping, compute projection needed to map onto L.x/L.y domain, add 1 for negative values to accommodate sign
	I_w[0] = Ix0/Grid.NX - (Ix0 < 0); I_w[1] = Iy0/Grid.NY - (Iy0 < 0);
	I_w[2] = Ix1/Grid.NX - (Ix1 < 0); I_w[3] = Iy1/Grid.NY - (Iy1 < 0);

	// project back into domain
	Ix0 -= I_w[0]*Grid.NX; Iy0 -= I_w[1]*Grid.NY; Ix1 -= I_w[2]*Grid.NX; Iy1 -= I_w[3]*Grid.NY;

	// I00, I10, I01, I11 in Vector to shorten function calls
	I[0] = Iy0 * Grid.NX + Ix0; I[1] = Iy0 * Grid.NX + Ix1; I[2] = Iy1 * Grid.NX + Ix0; I[3] = Iy1 * Grid.NX + Ix1;
}
template<typename T>
__device__ void device_init_ind_diff(int *I, int *I_w, T *dxyz, T x, T y, T z, TCudaGrid2D Grid) {
	// cell index
	int Ix0 = floor((x-Grid.bounds[0])/Grid.hx);
	int Iy0 = floor((y-Grid.bounds[2])/Grid.hy);
	int Iz0 = floor((z-Grid.bounds[4])/Grid.hz);
	int Ix1 = Ix0 + 1; int Iy1 = Iy0 + 1; int Iz1 = Iz0 + 1;

	//dx, dy
	dxyz[0] = (x-Grid.bounds[0])/Grid.hx - Ix0;
	dxyz[1] = (y-Grid.bounds[2])/Grid.hy - Iy0;
	dxyz[1] = (z-Grid.bounds[4])/Grid.hz - Iz0;

	//warping, compute projection needed to map onto L.x/L.y domain, add 1 for negative values to accommodate sign
	I_w[0] = Ix0/Grid.NX - (Ix0 < 0); I_w[1] = Iy0/Grid.NY - (Iy0 < 0);
	I_w[2] = Ix1/Grid.NX - (Ix1 < 0); I_w[3] = Iy1/Grid.NY - (Iy1 < 0);
	I_w[4] = Iz0/Grid.NZ - (Iz0 < 0); I_w[5] = Iz1/Grid.NZ - (Iz1 < 0);

	// project back into domain
	Ix0 -= I_w[0]*Grid.NX; Iy0 -= I_w[1]*Grid.NY; Ix1 -= I_w[2]*Grid.NX; Iy1 -= I_w[3]*Grid.NY; Iz0 -= I_w[4]*Grid.NZ; Iz1 -= I_w[5]*Grid.NZ;

	// I000, I100, I010, I110, I001, I101, I011, I111 in Vector to shorten function calls
	I[0] = Iz0 * Grid.NX*Grid.NY + Iy0 * Grid.NX + Ix0; I[1] = Iz0 * Grid.NX*Grid.NY + Iy0 * Grid.NX + Ix1;
	I[2] = Iz0 * Grid.NX*Grid.NY + Iy1 * Grid.NX + Ix0; I[3] = Iz0 * Grid.NX*Grid.NY + Iy1 * Grid.NX + Ix1;
	I[4] = Iz1 * Grid.NX*Grid.NY + Iy0 * Grid.NX + Ix0; I[5] = Iz1 * Grid.NX*Grid.NY + Iy0 * Grid.NX + Ix1;
	I[6] = Iz1 * Grid.NX*Grid.NY + Iy1 * Grid.NX + Ix0; I[7] = Iz1 * Grid.NX*Grid.NY + Iy1 * Grid.NX + Ix1;
}


// combine the build of the cubic vectors to reduce redundant code, build from later matrices due to less computations
template<typename T>
__device__ void device_build_b(T *bX, T *bY, T dx, T dy) {
	bX[0] = H_f3_3(1-dx); bX[1] = H_f3_3(dx); bX[2] = -H_f4_3(1-dx); bX[3] = H_f4_3(dx);
	bY[0] = H_f3_3(1-dy); bY[1] = H_f3_3(dy); bY[2] = -H_f4_3(1-dy); bY[3] = H_f4_3(dy);
//	bX[0] = H_f1_3(dx); bX[1] = H_f1_3(1-dx); bX[2] = H_f2_3(dx); bX[3] = -H_f2_3(1-dx);
//	bY[0] = H_f1_3(dy); bY[1] = H_f1_3(1-dy); bY[2] = H_f2_3(dy); bY[3] = -H_f2_3(1-dy);
}
template<typename T>
__device__ void device_build_b(T *bX, T *bY, T *bZ, T dx, T dy, T dz) {
	bX[0] = H_f3_3(1-dx); bX[1] = H_f3_3(dx); bX[2] = -H_f4_3(1-dx); bX[3] = H_f4_3(dx);
	bY[0] = H_f3_3(1-dy); bY[1] = H_f3_3(dy); bY[2] = -H_f4_3(1-dy); bY[3] = H_f4_3(dy);
	bZ[0] = H_f3_3(1-dz); bZ[1] = H_f3_3(dz); bZ[2] = -H_f4_3(1-dz); bZ[3] = H_f4_3(dz);
}
template<typename T>
__device__ void device_build_bx(T *bX, T *bY, T dx, T dy) {
	bX[0] = -H_f3x_3(1-dx); bX[1] = H_f3x_3(dx); bX[2] =  H_f4x_3(1-dx); bX[3] = H_f4x_3(dx);
	bY[0] =  H_f3_3(1-dy);  bY[1] = H_f3_3(dy);  bY[2] = -H_f4_3(1-dy);  bY[3] = H_f4_3(dy);
//	bX[0] = H_f1x_3(dx); bX[1] = -H_f1x_3(1-dx); bX[2] = H_f2x_3(dx); bX[3] =  H_f2x_3(1-dx);
//	bY[0] = H_f1_3(dy);  bY[1] =  H_f1_3(1-dy);  bY[2] = H_f2_3(dy);  bY[3] = -H_f2_3(1-dy);
}
template<typename T>
__device__ void device_build_bx(T *bX, T *bY, T *bZ, T dx, T dy, T dz) {
	bX[0] = -H_f3x_3(1-dx); bX[1] = H_f3x_3(dx); bX[2] =  H_f4x_3(1-dx); bX[3] = H_f4x_3(dx);
	bY[0] =  H_f3_3(1-dy);  bY[1] = H_f3_3(dy);  bY[2] = -H_f4_3(1-dy);  bY[3] = H_f4_3(dy);
	bZ[0] =  H_f3_3(1-dz);  bZ[1] = H_f3_3(dz);  bZ[2] = -H_f4_3(1-dz);  bZ[3] = H_f4_3(dz);
}
template<typename T>
__device__ void device_build_by(T *bX, T *bY, T dx, T dy) {
	bX[0] =  H_f3_3(1-dx);  bX[1] = H_f3_3(dx);  bX[2] = -H_f4_3(1-dx);  bX[3] = H_f4_3(dx);
	bY[0] = -H_f3x_3(1-dy); bY[1] = H_f3x_3(dy); bY[2] =  H_f4x_3(1-dy); bY[3] = H_f4x_3(dy);
//	bX[0] = H_f1_3(dx);  bX[1] =  H_f1_3(1-dx);  bX[2] = H_f2_3(dx);  bX[3] = -H_f2_3(1-dx);
//	bY[0] = H_f1x_3(dy); bY[1] = -H_f1x_3(1-dy); bY[2] = H_f2x_3(dy); bY[3] =  H_f2x_3(1-dy);
}
template<typename T>
__device__ void device_build_by(T *bX, T *bY, T *bZ, T dx, T dy, T dz) {
	bX[0] =  H_f3_3(1-dx);  bX[1] = H_f3_3(dx);  bX[2] = -H_f4_3(1-dx);  bX[3] = H_f4_3(dx);
	bY[0] = -H_f3x_3(1-dy); bY[1] = H_f3x_3(dy); bY[2] =  H_f4x_3(1-dy); bY[3] = H_f4x_3(dy);
	bZ[0] =  H_f3_3(1-dz);  bZ[1] = H_f3_3(dz);  bZ[2] = -H_f4_3(1-dz);  bZ[3] = H_f4_3(dz);
}
template<typename T>
__device__ void device_build_bz(T *bX, T *bY, T *bZ, T dx, T dy, T dz) {
	bX[0] =  H_f3_3(1-dx);  bX[1] = H_f3_3(dx);  bX[2] = -H_f4_3(1-dx);  bX[3] = H_f4_3(dx);
	bY[0] =  H_f3_3(1-dy);  bY[1] = H_f3_3(dy);  bY[2] = -H_f4_3(1-dy);  bY[3] = H_f4_3(dy);
	bZ[0] = -H_f3x_3(1-dz); bZ[1] = H_f3x_3(dz); bZ[2] =  H_f4x_3(1-dz); bZ[3] = H_f4x_3(dz);
}
// combine the build of the quintic vectors to reduce redundant code, build from later matrices due to less computations
template<typename T>
__device__ void device_build_b_5(T *bX, T *bY, T dx, T dy) {
	bX[0] = H_f4_5(1-dx); bX[1] = H_f4_5(dx); bX[2] = -H_f5_5(1-dx); bX[3] = H_f5_5(dx); bX[4] = H_f6_5(1-dx); bX[5] = H_f6_5(dx);
	bY[0] = H_f4_5(1-dy); bY[1] = H_f4_5(dy); bY[2] = -H_f5_5(1-dy); bY[3] = H_f5_5(dy); bY[4] = H_f6_5(1-dy); bY[5] = H_f6_5(dy);
}
template<typename T>
__device__ void device_build_bx_5(T *bX, T *bY, T dx, T dy) {
	bX[0] = -H_f4x_5(1-dx); bX[1] = H_f4x_5(dx); bX[2] =  H_f5x_5(1-dx); bX[3] = H_f5x_5(dx); bX[4] = -H_f6x_5(1-dx); bX[5] = H_f6x_5(dx);
	bY[0] =  H_f4_5(1-dy);  bY[1] = H_f4_5(dy);  bY[2] = -H_f5_5(1-dy);  bY[3] = H_f5_5(dy);  bY[4] =  H_f6_5(1-dy);  bY[5] = H_f6_5(dy);
}
template<typename T>
__device__ void device_build_by_5(T *bX, T *bY, T dx, T dy) {
	bX[0] =  H_f4_5(1-dx);  bX[1] = H_f4_5(dx);  bX[2] = -H_f5_5(1-dx);  bX[3] = H_f5_5(dx);  bX[4] =  H_f6_5(1-dx);  bX[5] = H_f6_5(dx);
	bY[0] = -H_f4x_5(1-dy); bY[1] = H_f4x_5(dy); bY[2] =  H_f5x_5(1-dy); bY[3] = H_f5x_5(dy); bY[4] = -H_f6x_5(1-dy); bY[5] = H_f6x_5(dy);
}

// function to construct b in matrix form
template<typename T>
__device__ void device_build_b_mat(T b[][4], T bX[], T bY[]) {
	for (int i_x = 0; i_x < 4; ++i_x) {
		for (int i_y = 0; i_y < 4; ++i_y) {
			b[i_x][i_y] = bX[i_y]*bY[i_x];
		}
	}
}
template<typename T>
__device__ void device_build_b_mat(T b[][4][4], T bX[], T bY[], T bZ[]) {
	for (int i_x = 0; i_x < 4; ++i_x) {
		for (int i_y = 0; i_y < 4; ++i_y) {
			for (int i_z = 0; i_z < 4; ++i_z) {
				b[i_x][i_y][i_z] = bX[i_z]*bY[i_y]*bZ[i_x];
			}
		}
	}
}



__device__ double device_hermite_interpolate_2D(double *H, double x, double y, TCudaGrid2D Grid)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; double dxy[2];
	device_init_ind<double>(I, dxy, x, y, Grid);
	double bX[4], bY[4];
	device_build_b<double>(bX, bY, dxy[0], dxy[1]);
	return device_hermite_mult_2D<double>(H, bX, bY, I, Grid.N, Grid.hx, Grid.hy);
}
__device__ float device_hermite_interpolate_2D(float *H, float x, float y, TCudaGrid2D Grid)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; float dxy[2];
	device_init_ind<float>(I, dxy, x, y, Grid);
	float bX[4], bY[4];
	device_build_b<float>(bX, bY, dxy[0], dxy[1]);
	return device_hermite_mult_2D<float>(H, bX, bY, I, Grid.N, Grid.hx, Grid.hy);
}


__device__ double device_hermite_interpolate_dx_2D(double *H, double x, double y, TCudaGrid2D Grid)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; double dxy[2];
	device_init_ind<double>(I, dxy, x, y, Grid);
	double bX[4], bY[4];
	device_build_bx<double>(bX, bY, dxy[0], dxy[1]);
	return device_hermite_mult_2D<double>(H, bX, bY, I, Grid.N, Grid.hx, Grid.hy) / Grid.hx;
}
__device__ float device_hermite_interpolate_dx_2D(float *H, float x, float y, TCudaGrid2D Grid)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; float dxy[2];
	device_init_ind<float>(I, dxy, x, y, Grid);
	float bX[4], bY[4];
	device_build_bx<float>(bX, bY, dxy[0], dxy[1]);
	return device_hermite_mult_2D<float>(H, bX, bY, I, Grid.N, Grid.hx, Grid.hy) / Grid.hx;
}


__device__ double device_hermite_interpolate_dy_2D(double *H, double x, double y, TCudaGrid2D Grid)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; double dxy[2];
	device_init_ind<double>(I, dxy, x, y, Grid);
	double bX[4], bY[4];
	device_build_by<double>(bX, bY, dxy[0], dxy[1]);
	return device_hermite_mult_2D<double>(H, bX, bY, I, Grid.N, Grid.hx, Grid.hy) / Grid.hy;
}
__device__ float device_hermite_interpolate_dy_2D(float *H, float x, float y, TCudaGrid2D Grid)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; float dxy[2];
	device_init_ind<float>(I, dxy, x, y, Grid);
	float bX[4], bY[4];
	device_build_by<float>(bX, bY, dxy[0], dxy[1]);
	return device_hermite_mult_2D<float>(H, bX, bY, I, Grid.N, Grid.hx, Grid.hy) / Grid.hy;
}


__device__ void device_hermite_interpolate_dx_dy_2D(double *H, double x, double y, double *fx, double *fy, TCudaGrid2D Grid)
{
	*fx = device_hermite_interpolate_dx_2D(H, x, y, Grid);
	*fy = device_hermite_interpolate_dy_2D(H, x, y, Grid);
}
__device__ void device_hermite_interpolate_dx_dy_2D(float *H, float x, float y, float *fx, float *fy, TCudaGrid2D Grid)
{
	*fx = device_hermite_interpolate_dx_2D(H, x, y, Grid);
	*fy = device_hermite_interpolate_dy_2D(H, x, y, Grid);
}


// special function for map advection to compute dx and dy directL.y at the same positions with variable setting amount
__device__ void device_hermite_interpolate_grad_2D(double *H, double *x, double *u, TCudaGrid2D Grid, int n_l)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; double dxy[2];
	device_init_ind<double>(I, dxy, x[0], x[1], Grid);

	// compute x- and y-derivatives, giving -v and u
	for (int i_xy = 0; i_xy <= 1; ++i_xy) {
		double bX[4], bY[4];
		if (i_xy == 0) device_build_bx<double>(bX, bY, dxy[0], dxy[1]);
		else device_build_by<double>(bX, bY, dxy[0], dxy[1]);
		double b[4][4]; device_build_b_mat(b, bX, bY);

		if (i_xy == 0) {
			for (int i_l = 0; i_l < n_l; i_l++) {
				u[2*i_l+1] = -device_hermite_mult_2D<double>(H + 4*Grid.N*i_l, b, I, Grid.N, Grid.hx, Grid.hy)/Grid.hx;
			}
		}
		else {
			for (int i_l = 0; i_l < n_l; i_l++) {
				u[2*i_l  ] = device_hermite_mult_2D<double>(H + 4*Grid.N*i_l, b, I, Grid.N, Grid.hx, Grid.hy)/Grid.hy;
			}
		}
	}
}
__device__ void device_hermite_interpolate_grad_2D(float *H, float *x, float *u, TCudaGrid2D Grid, int n_l)
{
	int I[4]; float dxy[2];
	device_init_ind<float>(I, dxy, x[0], x[1], Grid);
	for (int i_xy = 0; i_xy <= 1; ++i_xy) {
		float bX[4], bY[4];
		if (i_xy == 0) device_build_bx<float>(bX, bY, dxy[0], dxy[1]);
		else device_build_by<float>(bX, bY, dxy[0], dxy[1]);
		float b[4][4]; device_build_b_mat(b, bX, bY);

		if (i_xy == 0) {
			for (int i_l = 0; i_l < n_l; i_l++) {
				u[2*i_l+1] = -device_hermite_mult_2D<float>(H + 4*Grid.N*i_l, b, I, Grid.N, Grid.hx, Grid.hy)/Grid.hx;
			}
		}
		else {
			for (int i_l = 0; i_l < n_l; i_l++) {
				u[2*i_l  ] = device_hermite_mult_2D<float>(H + 4*Grid.N*i_l, b, I, Grid.N, Grid.hx, Grid.hy)/Grid.hy;
			}
		}
	}
}


//diffeomorphisms provide warped interpolations with a jump at the boundaries
__device__ void device_diffeo_interpolate_2D(double *Hx, double *Hy, double x, double y, double *x2,  double *y2, TCudaGrid2D Grid)
{
	int I[4]; int I_w[4]; double dxy[2];
	device_init_ind_diff<double>(I, I_w, dxy, x, y, Grid);
	//jump on warping
	double L[2] = {Grid.NX*Grid.hx, Grid.NY*Grid.hy};

	double bX[4], bY[4]; device_build_b<double>(bX, bY, dxy[0], dxy[1]);
	double b[4][4]; device_build_b_mat(b, bX, bY);

	int I_w_n[4] = {I_w[0], I_w[2], I_w[0], I_w[2]};  // copy I_w values for x-dir
	*x2 = device_hermite_mult_2D_warp<double>(Hx, b, I, I_w_n, L[0], Grid.N, Grid.hx, Grid.hy);
	I_w_n[0] = I_w[1]; I_w_n[1] = I_w[1]; I_w_n[2] = I_w[3]; I_w_n[3] = I_w[3];  // copy I_w values for x-dir
	*y2 = device_hermite_mult_2D_warp<double>(Hy, b, I, I_w_n, L[1], Grid.N, Grid.hx, Grid.hy);
}
__device__ void device_diffeo_interpolate_2D(float *Hx, float *Hy, float x, float y, float *x2,  float *y2, TCudaGrid2D Grid)
{
	int I[4]; int I_w[4]; float dxy[2];
	device_init_ind_diff<float>(I, I_w, dxy, x, y, Grid);
	//jump on warping
	float L[2] = {static_cast<float>(Grid.NX*Grid.hx), static_cast<float>(Grid.NY*Grid.hy)};

	float bX[4], bY[4]; device_build_b<float>(bX, bY, dxy[0], dxy[1]);
	float b[4][4]; device_build_b_mat(b, bX, bY);

	int I_w_n[4] = {I_w[0], I_w[2], I_w[0], I_w[2]};  // copy I_w values for x-dir
	*x2 = device_hermite_mult_2D_warp<float>(Hx, b, I, I_w_n, L[0], Grid.N, Grid.hx, Grid.hy);
	I_w_n[0] = I_w[1]; I_w_n[1] = I_w[1]; I_w_n[2] = I_w[3]; I_w_n[3] = I_w[3];  // copy I_w values for x-dir
	*y2 = device_hermite_mult_2D_warp<float>(Hy, b, I, I_w_n, L[1], Grid.N, Grid.hx, Grid.hy);
}


// compute determinant of gradient of flowmap
__device__ double  device_diffeo_grad_2D(double *Hx, double *Hy, double x, double y, TCudaGrid2D Grid)																							// time cost
{
	int I[4]; int I_w[4]; double dxy[2];
	device_init_ind_diff<double>(I, I_w, dxy, x, y, Grid);
	//jump on warping
	double L[2] = {Grid.NX*Grid.hx, Grid.NY*Grid.hy};
	double Xx, Xy, Yx, Yy;  // fx/dx, fx/dy fy/dx fy/dy
	// compute x- and y-derivatives
	for (int i_xy = 0; i_xy <= 1; ++i_xy) {
		double bX[4], bY[4];
		if (i_xy == 0) device_build_bx<double>(bX, bY, dxy[0], dxy[1]);
		else device_build_by<double>(bX, bY, dxy[0], dxy[1]);
		double b[4][4]; device_build_b_mat(b, bX, bY);

		int I_w_n[4] = {I_w[0], I_w[2], I_w[0], I_w[2]};  // copy I_w values for x-dir
		double x_der =  device_hermite_mult_2D_warp<double>(Hx, b, I, I_w_n, L[0], Grid.N, Grid.hx, Grid.hy)/Grid.hx;
		I_w_n[0] = I_w[1]; I_w_n[1] = I_w[1]; I_w_n[2] = I_w[3]; I_w_n[3] = I_w[3];  // copy I_w values for x-dir
		double y_der =  device_hermite_mult_2D_warp<double>(Hy, b, I, I_w_n, L[1], Grid.N, Grid.hx, Grid.hy)/Grid.hy;

		if (i_xy == 0) { Xx = x_der; Yx = y_der;}
		else { Xy = x_der, Yy = y_der;}
	}

	return Xx*Yy - Xy*Yx;
}
__device__ float  device_diffeo_grad_2D(float *Hx, float *Hy, float x, float y, TCudaGrid2D Grid)																							// time cost
{
	int I[4]; int I_w[4]; float dxy[2];
	device_init_ind_diff<float>(I, I_w, dxy, x, y, Grid);
	//jump on warping
	float L[2] = {static_cast<float>(Grid.NX*Grid.hx), static_cast<float>(Grid.NY*Grid.hy)};

	float Xx, Xy, Yx, Yy;  // fx/dx, fx/dy fy/dx fy/dy
	// compute x- and y-derivatives
	for (int i_xy = 0; i_xy <= 1; ++i_xy) {
		float bX[4], bY[4];
		if (i_xy == 0) device_build_bx<float>(bX, bY, dxy[0], dxy[1]);
		else device_build_by<float>(bX, bY, dxy[0], dxy[1]);
		float b[4][4]; device_build_b_mat(b, bX, bY);

		int I_w_n[4] = {I_w[0], I_w[2], I_w[0], I_w[2]};  // copy I_w values for x-dir
		float x_der =  device_hermite_mult_2D_warp<float>(Hx, b, I, I_w_n, L[0], Grid.N, Grid.hx, Grid.hy)/Grid.hx;
		I_w_n[0] = I_w[1]; I_w_n[1] = I_w[1]; I_w_n[2] = I_w[3]; I_w_n[3] = I_w[3];  // copy I_w values for x-dir
		float y_der =  device_hermite_mult_2D_warp<float>(Hy, b, I, I_w_n, L[1], Grid.N, Grid.hx, Grid.hy)/Grid.hy;

		if (i_xy == 0) { Xx = x_der; Yx = y_der;}
		else { Xy = x_der, Yy = y_der;}
	}

	return Xx*Yy - Xy*Yx;
}



