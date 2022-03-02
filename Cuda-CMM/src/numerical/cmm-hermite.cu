#include "cmm-hermite.h"

#include "../grid/cmm-grid2d.h"

/*******************************************************************
*						  Hermite interpolation					   *
*******************************************************************/
// this computation is used very often, using it as a function greatly reduces code-redundancy
// however, here we have 8 scattered memory accesses where I[0] - I[1] and I[2] - I[3] are paired (in theory)
__device__ double device_hermite_mult_2D(double *H, double b[][4], int I[], long int N, double h)
{
	return b[0][0]* H[    I[0]] + b[0][1]* H[    I[1]] + b[1][0]* H[    I[2]] + b[1][1]* H[    I[3]] // Point interpolation
	    + (b[0][2]* H[1*N+I[0]] + b[0][3]* H[1*N+I[1]] + b[1][2]* H[1*N+I[2]] + b[1][3]* H[1*N+I[3]]) * (h)  // dx
	    + (b[2][0]* H[2*N+I[0]] + b[2][1]* H[2*N+I[1]] + b[3][0]* H[2*N+I[2]] + b[3][1]* H[2*N+I[3]]) * (h)  // dy
	    + (b[2][2]* H[3*N+I[0]] + b[2][3]* H[3*N+I[1]] + b[3][2]* H[3*N+I[2]] + b[3][3]* H[3*N+I[3]]) * (h*h);  // dx dy
}


// save memory by not storing the matrix b but computing it in the function
__device__ double device_hermite_mult_2D(double *H, double bX[], double bY[], int I[], long int N, double h)
{
	return bX[0]*bY[0]* H[    I[0]] + bX[1]*bY[0]* H[    I[1]] + bX[0]*bY[1]* H[    I[2]] + bX[1]*bY[1]* H[    I[3]] // Point interpolation
	    + (bX[2]*bY[0]* H[1*N+I[0]] + bX[3]*bY[0]* H[1*N+I[1]] + bX[2]*bY[1]* H[1*N+I[2]] + bX[3]*bY[1]* H[1*N+I[3]]) * (h)  // dx
	    + (bX[0]*bY[2]* H[2*N+I[0]] + bX[1]*bY[2]* H[2*N+I[1]] + bX[0]*bY[3]* H[2*N+I[2]] + bX[1]*bY[3]* H[2*N+I[3]]) * (h)  // dy
	    + (bX[2]*bY[2]* H[3*N+I[0]] + bX[3]*bY[2]* H[3*N+I[1]] + bX[2]*bY[3]* H[3*N+I[2]] + bX[3]*bY[3]* H[3*N+I[3]]) * (h*h);  // dx dy
}


// combine the build of the index vector to reduce redundant code
__device__ void device_init_ind(int *I, double *dxy, double x, double y, TCudaGrid2D Grid) {
	//cell index
	int Ix0 = floor(x/Grid.hx); int Iy0 = floor(y/Grid.hy);
	int Ix1 = Ix0 + 1; int Iy1 = Iy0 + 1;

	//dx, dy
	dxy[0] = x/Grid.hx - Ix0; dxy[1] = y/Grid.hy - Iy0;

	// project into domain, < 0 check is needed as integer division rounds towards 0
	Ix0 -= (Ix0/Grid.NX - (Ix0 < 0))*Grid.NX; Iy0 -= (Iy0/Grid.NY - (Iy0 < 0))*Grid.NY;
	Ix1 -= (Ix1/Grid.NX - (Ix1 < 0))*Grid.NX; Iy1 -= (Iy1/Grid.NY - (Iy1 < 0))*Grid.NY;

	// I00, I10, I01, I11 in Vector to shorten function calls
	I[0] = Iy0 * Grid.NX + Ix0; I[1] = Iy0 * Grid.NX + Ix1; I[2] = Iy1 * Grid.NX + Ix0; I[3] = Iy1 * Grid.NX + Ix1;
}


// combine the build of the cubic vectors to reduce redundant code, build from later matrices due to less computations
__device__ void device_build_b(double *bX, double *bY, double dx, double dy) {
	bX[0] = H_f3_3(1-dx); bX[1] = H_f3_3(dx); bX[2] = -H_f4_3(1-dx); bX[3] = H_f4_3(dx);
	bY[0] = H_f3_3(1-dy); bY[1] = H_f3_3(dy); bY[2] = -H_f4_3(1-dy); bY[3] = H_f4_3(dy);
//	bX[0] = H_f1_3(dx); bX[1] = H_f1_3(1-dx); bX[2] = H_f2_3(dx); bX[3] = -H_f2_3(1-dx);
//	bY[0] = H_f1_3(dy); bY[1] = H_f1_3(1-dy); bY[2] = H_f2_3(dy); bY[3] = -H_f2_3(1-dy);
}
__device__ void device_build_bx(double *bX, double *bY, double dx, double dy) {
	bX[0] = -H_f3x_3(1-dx); bX[1] = H_f3x_3(dx); bX[2] =  H_f4x_3(1-dx); bX[3] = H_f4x_3(dx);
	bY[0] =  H_f3_3(1-dy);  bY[1] = H_f3_3(dy);  bY[2] = -H_f4_3(1-dy);  bY[3] = H_f4_3(dy);
//	bX[0] = H_f1x_3(dx); bX[1] = -H_f1x_3(1-dx); bX[2] = H_f2x_3(dx); bX[3] =  H_f2x_3(1-dx);
//	bY[0] = H_f1_3(dy);  bY[1] =  H_f1_3(1-dy);  bY[2] = H_f2_3(dy);  bY[3] = -H_f2_3(1-dy);
}
__device__ void device_build_by(double *bX, double *bY, double dx, double dy) {
	bX[0] =  H_f3_3(1-dx);  bX[1] = H_f3_3(dx);  bX[2] = -H_f4_3(1-dx);  bX[3] = H_f4_3(dx);
	bY[0] = -H_f3x_3(1-dy); bY[1] = H_f3x_3(dy); bY[2] =  H_f4x_3(1-dy); bY[3] = H_f4x_3(dy);
//	bX[0] = H_f1_3(dx);  bX[1] =  H_f1_3(1-dx);  bX[2] = H_f2_3(dx);  bX[3] = -H_f2_3(1-dx);
//	bY[0] = H_f1x_3(dy); bY[1] = -H_f1x_3(1-dy); bY[2] = H_f2x_3(dy); bY[3] =  H_f2x_3(1-dy);
}
// combine the build of the quintic vectors to reduce redundant code, build from later matrices due to less computations
__device__ void device_build_b_5(double *bX, double *bY, double dx, double dy) {
	bX[0] = H_f4_5(1-dx); bX[1] = H_f4_5(dx); bX[2] = -H_f5_5(1-dx); bX[3] = H_f5_5(dx); bX[4] = H_f6_5(1-dx); bX[5] = H_f6_5(dx);
	bY[0] = H_f4_5(1-dy); bY[1] = H_f4_5(dy); bY[2] = -H_f5_5(1-dy); bY[3] = H_f5_5(dy); bY[4] = H_f6_5(1-dy); bY[5] = H_f6_5(dy);
}
__device__ void device_build_bx_5(double *bX, double *bY, double dx, double dy) {
	bX[0] = -H_f4x_5(1-dx); bX[1] = H_f4x_5(dx); bX[2] =  H_f5x_5(1-dx); bX[3] = H_f5x_5(dx); bX[4] = -H_f6x_5(1-dx); bX[5] = H_f6x_5(dx);
	bY[0] =  H_f4_5(1-dy);  bY[1] = H_f4_5(dy);  bY[2] = -H_f5_5(1-dy);  bY[3] = H_f5_5(dy);  bY[4] =  H_f6_5(1-dy);  bY[5] = H_f6_5(dy);
}
__device__ void device_build_by_5(double *bX, double *bY, double dx, double dy) {
	bX[0] =  H_f4_5(1-dx);  bX[1] = H_f4_5(dx);  bX[2] = -H_f5_5(1-dx);  bX[3] = H_f5_5(dx);  bX[4] =  H_f6_5(1-dx);  bX[5] = H_f6_5(dx);
	bY[0] = -H_f4x_5(1-dy); bY[1] = H_f4x_5(dy); bY[2] =  H_f5x_5(1-dy); bY[3] = H_f5x_5(dy); bY[4] = -H_f6x_5(1-dy); bY[5] = H_f6x_5(dy);
}



__device__ double device_hermite_interpolate_2D(double *H, double x, double y, TCudaGrid2D Grid)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; double dxy[2];
	device_init_ind(I, dxy, x, y, Grid);

	double bX[4], bY[4];
	device_build_b(bX, bY, dxy[0], dxy[1]);

	return device_hermite_mult_2D(H, bX, bY, I, Grid.N, Grid.h);
}


__device__ double device_hermite_interpolate_dx_2D(double *H, double x, double y, TCudaGrid2D Grid)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; double dxy[2];
	device_init_ind(I, dxy, x, y, Grid);

	double bX[4], bY[4];
	device_build_bx(bX, bY, dxy[0], dxy[1]);

	return device_hermite_mult_2D(H, bX, bY, I, Grid.N, Grid.h) / Grid.hx;
}


__device__ double device_hermite_interpolate_dy_2D(double *H, double x, double y, TCudaGrid2D Grid)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; double dxy[2];
	device_init_ind(I, dxy, x, y, Grid);

	double bX[4], bY[4];
	device_build_by(bX, bY, dxy[0], dxy[1]);

	return device_hermite_mult_2D(H, bX, bY, I, Grid.N, Grid.h) / Grid.hy;
}


__device__ void device_hermite_interpolate_dx_dy_2D(double *H, double x, double y, double *fx, double *fy, TCudaGrid2D Grid)
{
	*fx = device_hermite_interpolate_dx_2D(H, x, y, Grid);
	*fy = device_hermite_interpolate_dy_2D(H, x, y, Grid);
}


// special function for map advection to compute dx and dy directly at the same positions with variable setting amount
__device__ void device_hermite_interpolate_grad_2D(double *H, double *x, double *u, TCudaGrid2D Grid, int n_l)
{
	// init array containing the indexes and finite differences dx dy to neighbours
	int I[4]; double dxy[2];
	device_init_ind(I, dxy, x[0], x[1], Grid);

	// computing all dx-interpolations, giving -v
	{
		double bX[4], bY[4];
		device_build_bx(bX, bY, dxy[0], dxy[1]);

		double b[4][4] = {
							bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
							bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
							bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
							bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
						};

		for (int i_l = 0; i_l < n_l; i_l++) u[2*i_l+1] = -device_hermite_mult_2D(H + 4*Grid.N*i_l, b, I, Grid.N, Grid.h)/Grid.hx;
//		for (int i_l = 0; i_l < n_l; i_l++) u[2*i_l+1] = -device_hermite_mult_2D(H + 4*NX*NY*i_l, bX, bY, I, NX*NY, h)/h;

	}
	// compute all dy-interpolations, giving u
	{
		double bX[4], bY[4];
		device_build_by(bX, bY, dxy[0], dxy[1]);

		double b[4][4] = {
							bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
							bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
							bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
							bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
						};

		for (int i_l = 0; i_l < n_l; i_l++) u[2*i_l  ] = device_hermite_mult_2D(H + 4*Grid.N*i_l, b, I, Grid.N, Grid.h)/Grid.hy;
//		for (int i_l = 0; i_l < n_l; i_l++) u[2*i_l  ] = device_hermite_mult_2D(H + 4*NX*NY*i_l, bX, bY, I, NX*NY, h)/h;
	}
}


//diffeomorphisms provide warped interpolations with a jump at the boundaries
__device__ void device_diffeo_interpolate_2D(double *Hx, double *Hy, double x, double y, double *x2,  double *y2, TCudaGrid2D Grid)
{
	// build up all needed positioning
	// cell index of footpoints
	int Ix0 = floor(x/Grid.hx); int Iy0 = floor(y/Grid.hy);
	int Ix1 = Ix0 + 1; int Iy1 = Iy0 + 1;

	//dx, dy, distance to footpoints
	double dx = x/Grid.hx - Ix0;
	double dy = y/Grid.hy - Iy0;

	//warping, compute projection needed to map onto LX/LY domain, add 1 for negative values to accommodate sign
	int I_w[4] = {Ix0/Grid.NX - (Ix0 < 0), Iy0/Grid.NY - (Iy0 < 0), Ix1/Grid.NX - (Ix1 < 0), Iy1/Grid.NY - (Iy1 < 0)};

	//jump on warping
	double2 L;
	L.x = Grid.NX*Grid.hx; L.y = Grid.NY*Grid.hy;

	// project back into domain
	Ix0 -= I_w[0]*Grid.NX; Iy0 -= I_w[1]*Grid.NY; Ix1 -= I_w[2]*Grid.NX; Iy1 -= I_w[3]*Grid.NY;

	// I00, I10, I01, I11 in Vector to shorten function calls
	int I[4] = {Iy0 * Grid.NX + Ix0, Iy0 * Grid.NX + Ix1, Iy1 * Grid.NX + Ix0, Iy1 * Grid.NX + Ix1};

	double bX[4], bY[4];
	device_build_b(bX, bY, dx, dy);


	double b[4][4] = {
						bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
						bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
						bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
						bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
					};

	*x2 =  b[0][0]* (Hx[I[0]] + I_w[0]*L.x) + b[0][1]* (Hx[I[1]] + I_w[2]*L.x) + b[1][0]* (Hx[I[2]] + I_w[0]*L.x) + b[1][1]* (Hx[I[3]] + I_w[2]*L.x) // Point interpolation
	    + (b[0][2]*  Hx[1*Grid.N+I[0]]      + b[0][3]*  Hx[1*Grid.N+I[1]]      + b[1][2]*  Hx[1*Grid.N+I[2]]      + b[1][3]*  Hx[1*Grid.N+I[3]]) * (Grid.hx)  // dx
	    + (b[2][0]*  Hx[2*Grid.N+I[0]]      + b[2][1]*  Hx[2*Grid.N+I[1]]      + b[3][0]*  Hx[2*Grid.N+I[2]]      + b[3][1]*  Hx[2*Grid.N+I[3]]) * (Grid.hy)  // dy
	    + (b[2][2]*  Hx[3*Grid.N+I[0]]      + b[2][3]*  Hx[3*Grid.N+I[1]]      + b[3][2]*  Hx[3*Grid.N+I[2]]      + b[3][3]*  Hx[3*Grid.N+I[3]]) * (Grid.hx*Grid.hy);  // dx dy

	*y2 =  b[0][0]* (Hy[I[0]] + I_w[1]*L.y) + b[0][1]* (Hy[I[1]] + I_w[1]*L.y) + b[1][0]* (Hy[I[2]] + I_w[3]*L.y) + b[1][1]* (Hy[I[3]] + I_w[3]*L.y) // Point interpolation
	    + (b[0][2]*  Hy[1*Grid.N+I[0]]      + b[0][3]*  Hy[1*Grid.N+I[1]]      + b[1][2]*  Hy[1*Grid.N+I[2]]      + b[1][3]*  Hy[1*Grid.N+I[3]]) * (Grid.hx)  // dx
	    + (b[2][0]*  Hy[2*Grid.N+I[0]]      + b[2][1]*  Hy[2*Grid.N+I[1]]      + b[3][0]*  Hy[2*Grid.N+I[2]]      + b[3][1]*  Hy[2*Grid.N+I[3]]) * (Grid.hy)  // dy
	    + (b[2][2]*  Hy[3*Grid.N+I[0]]      + b[2][3]*  Hy[3*Grid.N+I[1]]      + b[3][2]*  Hy[3*Grid.N+I[2]]      + b[3][3]*  Hy[3*Grid.N+I[3]]) * (Grid.hx*Grid.hy);  // dx dy
}


// compute determinant of gradient of flowmap
__device__ double  device_diffeo_grad_2D(double *Hx, double *Hy, double x, double y, TCudaGrid2D Grid)																							// time cost
{
	// build up all needed positioning
	// cell index of footpoints
	int Ix0 = floor(x/Grid.hx); int Iy0 = floor(y/Grid.hy);
	int Ix1 = Ix0 + 1; int Iy1 = Iy0 + 1;

	//dx, dy, distance to footpoints
	double dx = x/Grid.hx - Ix0;
	double dy = y/Grid.hy - Iy0;

	//warping, compute projection needed to map onto LX/LY domain, add 1 for negative values to accommodate sign
	int Ix0W = Ix0/Grid.NX - (Ix0 < 0); int Ix1W = Ix1/Grid.NX - (Ix1 < 0);
	int Iy0W = Iy0/Grid.NY - (Iy0 < 0); int Iy1W = Iy1/Grid.NY - (Iy1 < 0);

	//jump on warping
	double LX = Grid.NX*Grid.hx; double LY = Grid.NY*Grid.hy;

	// project back into domain
	Ix0 -= Ix0W*Grid.NX; Iy0 -= Iy0W*Grid.NY; Ix1 -= Ix1W*Grid.NX; Iy1 -= Iy1W*Grid.NY;

	// I00, I10, I01, I11 in Vector to shorten function calls
	int I[4] = {Iy0 * Grid.NX + Ix0, Iy0 * Grid.NX + Ix1, Iy1 * Grid.NX + Ix0, Iy1 * Grid.NX + Ix1};

	double Xx, Xy, Yx, Yy;  // fx/dx, fx/dy fy/dx fy/dy
	// compute x derivatives
	{
		double bX[4], bY[4];
		device_build_bx(bX, bY, dx, dy);

		double b[4][4] = {
							bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
							bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
							bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
							bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
						};

		Xx =  (b[0][0]* (Hx[I[0]] + Ix0W*LX) + b[0][1]* (Hx[I[1]] + Ix1W*LX) + b[1][0]* (Hx[I[2]] + Ix0W*LX) + b[1][1]* (Hx[I[3]] + Ix1W*LX) // Point interpolation
			+ (b[0][2]*  Hx[1*Grid.N+I[0]]        + b[0][3]*  Hx[1*Grid.N+I[1]]        + b[1][2]*  Hx[1*Grid.N+I[2]]        + b[1][3]*  Hx[1*Grid.N+I[3]]) * (Grid.hx)  // dx
			+ (b[2][0]*  Hx[2*Grid.N+I[0]]        + b[2][1]*  Hx[2*Grid.N+I[1]]        + b[3][0]*  Hx[2*Grid.N+I[2]]        + b[3][1]*  Hx[2*Grid.N+I[3]]) * (Grid.hy)  // dy
			+ (b[2][2]*  Hx[3*Grid.N+I[0]]        + b[2][3]*  Hx[3*Grid.N+I[1]]        + b[3][2]*  Hx[3*Grid.N+I[2]]        + b[3][3]*  Hx[3*Grid.N+I[3]]) * (Grid.hx*Grid.hy))/Grid.hx;  // dx dy

		Yx =  (b[0][0]* (Hy[I[0]] + Iy0W*LY) + b[0][1]* (Hy[I[1]] + Iy0W*LY) + b[1][0]* (Hy[I[2]] + Iy1W*LY) + b[1][1]* (Hy[I[3]] + Iy1W*LY) // Point interpolation
			+ (b[0][2]*  Hy[1*Grid.N+I[0]]        + b[0][3]*  Hy[1*Grid.N+I[1]]        + b[1][2]*  Hy[1*Grid.N+I[2]]        + b[1][3]*  Hy[1*Grid.N+I[3]]) * (Grid.hx)  // dx
			+ (b[2][0]*  Hy[2*Grid.N+I[0]]        + b[2][1]*  Hy[2*Grid.N+I[1]]        + b[3][0]*  Hy[2*Grid.N+I[2]]        + b[3][1]*  Hy[2*Grid.N+I[3]]) * (Grid.hy)  // dy
			+ (b[2][2]*  Hy[3*Grid.N+I[0]]        + b[2][3]*  Hy[3*Grid.N+I[1]]        + b[3][2]*  Hy[3*Grid.N+I[2]]        + b[3][3]*  Hy[3*Grid.N+I[3]]) * (Grid.hx*Grid.hy))/Grid.hy;  // dx dy
	}
	// compute y derivatives
	{
		double bX[4], bY[4];
		device_build_by(bX, bY, dx, dy);

		double b[4][4] = {
							bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
							bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
							bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
							bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
						};

		Xy =  (b[0][0]* (Hx[I[0]] + Ix0W*LX)      + b[0][1]* (Hx[I[1]] + Ix1W*LX)      + b[1][0]* (Hx[I[2]] + Ix0W*LX)      + b[1][1]* (Hx[I[3]] + Ix1W*LX) // Point interpolation
			+ (b[0][2]*  Hx[1*Grid.N+I[0]]        + b[0][3]*  Hx[1*Grid.N+I[1]]        + b[1][2]*  Hx[1*Grid.N+I[2]]        + b[1][3]*  Hx[1*Grid.N+I[3]]) * (Grid.hx)  // dx
			+ (b[2][0]*  Hx[2*Grid.N+I[0]]        + b[2][1]*  Hx[2*Grid.N+I[1]]        + b[3][0]*  Hx[2*Grid.N+I[2]]        + b[3][1]*  Hx[2*Grid.N+I[3]]) * (Grid.hy)  // dy
			+ (b[2][2]*  Hx[3*Grid.N+I[0]]        + b[2][3]*  Hx[3*Grid.N+I[1]]        + b[3][2]*  Hx[3*Grid.N+I[2]]        + b[3][3]*  Hx[3*Grid.N+I[3]]) * (Grid.hx*Grid.hy))/Grid.hy;  // dx dy

		Yy =  (b[0][0]* (Hy[I[0]] + Iy0W*LY)      + b[0][1]* (Hy[I[1]] + Iy0W*LY)      + b[1][0]* (Hy[I[2]] + Iy1W*LY)      + b[1][1]* (Hy[I[3]] + Iy1W*LY) // Point interpolation
			+ (b[0][2]*  Hy[1*Grid.N+I[0]]        + b[0][3]*  Hy[1*Grid.N+I[1]]        + b[1][2]*  Hy[1*Grid.N+I[2]]        + b[1][3]*  Hy[1*Grid.N+I[3]]) * (Grid.hx)  // dx
			+ (b[2][0]*  Hy[2*Grid.N+I[0]]        + b[2][1]*  Hy[2*Grid.N+I[1]]        + b[3][0]*  Hy[2*Grid.N+I[2]]        + b[3][1]*  Hy[2*Grid.N+I[3]]) * (Grid.hy)  // dy
			+ (b[2][2]*  Hy[3*Grid.N+I[0]]        + b[2][3]*  Hy[3*Grid.N+I[1]]        + b[3][2]*  Hy[3*Grid.N+I[2]]        + b[3][3]*  Hy[3*Grid.N+I[3]]) * (Grid.hx*Grid.hy))/Grid.hy;  // dx dy
	}

	return Xx*Yy - Xy*Yx;
}





/******************************************************************/
/*******************************************************************
*							   Old								   *
*******************************************************************/
/******************************************************************/
//void hermite_interpolation_test()
//{
//}
//
//
//__global__ void kernel_hermite_interpolation(double *H, double *F, int NXH, int NYH, int NXF, int NYF, double hH, double hF)
//{
//}





