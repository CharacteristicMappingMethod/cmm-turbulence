#include "cudahermite2d.h"


/*******************************************************************
*						  Hermite interpolation					   *
*******************************************************************/
// this computation is used very often, using it as a function greatly reduces code-redundancy
__device__ double device_hermite_mult(double *H, double b[][4], int I[], long int N, double h) // @suppress("Type cannot be resolved")
{
	return     b[0][0]*(H[I[0]])
			+  b[0][1]*(H[I[1]])
			+  b[1][0]*(H[I[2]])
			+  b[1][1]*(H[I[3]])

			+ (b[0][2]* H[1*N+I[0]] + b[0][3]* H[1*N+I[1]] + b[1][2]* H[1*N+I[2]] + b[1][3]* H[1*N+I[3]]) * (h)
			+ (b[2][0]* H[2*N+I[0]] + b[2][1]* H[2*N+I[1]] + b[3][0]* H[2*N+I[2]] + b[3][1]* H[2*N+I[3]]) * (h)
	        + (b[2][2]* H[3*N+I[0]] + b[2][3]* H[3*N+I[1]] + b[3][2]* H[3*N+I[2]] + b[3][3]* H[3*N+I[3]]) * (h*h);
}



__device__ double device_hermite_interpolate(double *H, double x, double y, int NX, int NY, double h) // @suppress("Type cannot be resolved")
{
	//cell index
	int Ix0 = floor(x/h);
	int Iy0 = floor(y/h);
	int Ix1 = Ix0 + 1;
	int Iy1 = Iy0 + 1;
	
	//dx, dy
	double dx = x/h - Ix0;
	double dy = y/h - Iy0;
		
	long int N = NX*NY;
	
	Ix0 = (Ix0+100 * NX)%NX;
	Iy0 = (Iy0+100 * NY)%NY;
	
	Ix1 = (Ix1+100 * NX)%NX;
	Iy1 = (Iy1+100 * NY)%NY;

	// I00, I10, I01, I11 in Vector to shorten function calls
	int I[4] = {Iy0 * NX + Ix0, Iy0 * NX + Ix1, Iy1 * NX + Ix0, Iy1 * NX + Ix1};
	
	
	double bX[4] = {Hf(dx), Hf(1-dx), Hg(dx), -Hg(1-dx)};
	double bY[4] = {Hf(dy), Hf(1-dy), Hg(dy), -Hg(1-dy)};
	
	
	double b[4][4] = {
						bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
						bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
						bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
						bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
					};
	return device_hermite_mult(H, b, I, N, h);
}


__device__ double device_hermite_interpolate_dx(double *H, double x, double y, int NX, int NY, double h)
{
	//cell index
	int Ix0 = floor(x/h);
	int Iy0 = floor(y/h);
	int Ix1 = Ix0 + 1;
	int Iy1 = Iy0 + 1;
	
	//dx, dy
	double dx = x/h - Ix0;
	double dy = y/h - Iy0;
		
	long int N = NX*NY;
	
	Ix0 = (Ix0+100 * NX)%NX;
	Iy0 = (Iy0+100 * NY)%NY;
	
	Ix1 = (Ix1+100 * NX)%NX;
	Iy1 = (Iy1+100 * NY)%NY;
	
	// I00, I10, I01, I11 in Vector to shorten function calls
	int I[4] = {Iy0 * NX + Ix0, Iy0 * NX + Ix1, Iy1 * NX + Ix0, Iy1 * NX + Ix1};
	
	
	double bX[4] = {Hfx(dx), -Hfx(1-dx), Hgx(dx), Hgx(1-dx)};
	double bY[4] = {Hf(dy), Hf(1-dy), Hg(dy), -Hg(1-dy)};
	
	
	double b[4][4] = {
						bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
						bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
						bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
						bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
					};
	return device_hermite_mult(H, b, I, N, h)/h;
}


__device__ double device_hermite_interpolate_dy(double *H, double x, double y, int NX, int NY, double h)
{
	//cell index
	int Ix0 = floor(x/h);
	int Iy0 = floor(y/h);
	int Ix1 = Ix0 + 1;
	int Iy1 = Iy0 + 1;
	
	//dx, dy
	double dx = x/h - Ix0;
	double dy = y/h - Iy0;
	
	long int N = NX*NY;
	
	Ix0 = (Ix0+100 * NX)%NX;
	Iy0 = (Iy0+100 * NY)%NY;
	
	Ix1 = (Ix1+100 * NX)%NX;
	Iy1 = (Iy1+100 * NY)%NY;
	
	// I00, I10, I01, I11 in Vector to shorten function calls
	int I[4] = {Iy0 * NX + Ix0, Iy0 * NX + Ix1, Iy1 * NX + Ix0, Iy1 * NX + Ix1};
	
	
	double bX[4] = {Hf(dx), Hf(1-dx), Hg(dx), -Hg(1-dx)};
	double bY[4] = {Hfx(dy), -Hfx(1-dy), Hgx(dy), Hgx(1-dy)};
	
	
	double b[4][4] = {
						bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
						bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
						bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
						bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
					};
	return device_hermite_mult(H, b, I, N, h)/h;
}


__device__ void device_hermite_interpolate_dx_dy(double *H, double x, double y, double *fx, double *fy, int NX, int NY, double h)
{
	*fx = device_hermite_interpolate_dx(H, x, y, NX, NY, h);
	*fy = device_hermite_interpolate_dy(H, x, y, NX, NY, h);
}


// special function for map advection to compute dx and dy directly at the same positions
__device__ void device_hermite_interpolate_dx_dy_1(double *H1, double x, double y, double *u1, double *v1, int NX, int NY, double h)
{
	// build up all needed positioning
	//cell index
	int Ix0 = floor(x/h);
	int Iy0 = floor(y/h);
	int Ix1 = Ix0 + 1;
	int Iy1 = Iy0 + 1;

	//dx, dy
	double dx = x/h - Ix0;
	double dy = y/h - Iy0;

	long int N = NX*NY;

	Ix0 = (Ix0+100 * NX)%NX;
	Iy0 = (Iy0+100 * NY)%NY;

	Ix1 = (Ix1+100 * NX)%NX;
	Iy1 = (Iy1+100 * NY)%NY;

	// I00, I10, I01, I11 in Vector to shorten function calls
	int I[4] = {Iy0 * NX + Ix0, Iy0 * NX + Ix1, Iy1 * NX + Ix0, Iy1 * NX + Ix1};

	// computing all dx-interpolations, giving -v
	{
		double bX[4] = {Hfx(dx), -Hfx(1-dx), Hgx(dx), Hgx(1-dx)};
		double bY[4] = {Hf(dy), Hf(1-dy), Hg(dy), -Hg(1-dy)};


		double b[4][4] = {
							bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
							bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
							bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
							bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
						};
		*v1 =  -device_hermite_mult(H1, b, I, N, h)/h;

	}
	// compute all dy-interpolations, giving u
	{
		double bX[4] = {Hf(dx), Hf(1-dx), Hg(dx), -Hg(1-dx)};
		double bY[4] = {Hfx(dy), -Hfx(1-dy), Hgx(dy), Hgx(1-dy)};


		double b[4][4] = {
							bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
							bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
							bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
							bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
						};
		*u1 = 	device_hermite_mult(H1, b, I, N, h)/h;
	}
}


// make it easier for RKthree, avoid redundant operations, naming is set to u and v to avoid confusion
__device__ void  device_hermite_interpolate_dx_dy_3(double *H1, double *H2, double *H3, double x, double y, double *u1, double *v1, double *u2, double *v2, double *u3, double *v3, int NX, int NY, double h)
{
	// build up all needed positioning
	//cell index
	int Ix0 = floor(x/h);
	int Iy0 = floor(y/h);
	int Ix1 = Ix0 + 1;
	int Iy1 = Iy0 + 1;

	//dx, dy
	double dx = x/h - Ix0;
	double dy = y/h - Iy0;

	long int N = NX*NY;

	Ix0 = (Ix0+100 * NX)%NX;
	Iy0 = (Iy0+100 * NY)%NY;

	Ix1 = (Ix1+100 * NX)%NX;
	Iy1 = (Iy1+100 * NY)%NY;

	// I00, I10, I01, I11 in Vector to shorten function calls
	int I[4] = {Iy0 * NX + Ix0, Iy0 * NX + Ix1, Iy1 * NX + Ix0, Iy1 * NX + Ix1};

	// computing all dx-interpolations, giving -v
	{
		double bX[4] = {Hfx(dx), -Hfx(1-dx), Hgx(dx), Hgx(1-dx)};
		double bY[4] = {Hf(dy), Hf(1-dy), Hg(dy), -Hg(1-dy)};
	
	
		double b[4][4] = {
							bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
							bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
							bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
							bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
						};
		*v1 = -device_hermite_mult(H1, b, I, N, h)/h;
		*v2 = -device_hermite_mult(H2, b, I, N, h)/h;
		*v3 = -device_hermite_mult(H3, b, I, N, h)/h;
	}
	// compute all dy-interpolations, giving u
	{
		double bX[4] = {Hf(dx), Hf(1-dx), Hg(dx), -Hg(1-dx)};
		double bY[4] = {Hfx(dy), -Hfx(1-dy), Hgx(dy), Hgx(1-dy)};
	
	
		double b[4][4] = {
							bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
							bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
							bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
							bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
						};
		*u1 = 	device_hermite_mult(H1, b, I, N, h)/h;
		*u2 = 	device_hermite_mult(H2, b, I, N, h)/h;
		*u3 = 	device_hermite_mult(H3, b, I, N, h)/h;
	}
}


//diffeomorHsms provides a warped interpolation with a jump at the boundaries
__device__ void  device_diffeo_interpolate(double *Hx, double *Hy, double x, double y, double *x2,  double *y2, int NX, int NY, double h)
{
	
	//cell index
	int Ix0 = floor(x/h);
	int Iy0 = floor(y/h);
	int Ix1 = Ix0 + 1;
	int Iy1 = Iy0 + 1;
	
	//dx, dy
	double dx = x/h - Ix0;
	double dy = y/h - Iy0;
		
	long int N = NX*NY;
	
	//warping
	int Ix0Wrap = ((int)((Ix0 + 100 * NX)/NX) - 100);
	int Iy0Wrap = ((int)((Iy0 + 100 * NY)/NY) - 100);
	int Ix1Wrap = ((int)((Ix1 + 100 * NX)/NX) - 100);
	int Iy1Wrap = ((int)((Iy1 + 100 * NY)/NY) - 100);
	
	//jump on warping
	double xJump = NX*h;
	double yJump = NY*h;
	
	Ix0 = (Ix0+100 * NX)%NX;
	Iy0 = (Iy0+100 * NY)%NY;
	Ix1 = (Ix1+100 * NX)%NX;
	Iy1 = (Iy1+100 * NY)%NY;
	
		
	int I00 = Iy0 * NX + Ix0;
	int I10 = Iy0 * NX + Ix1;
	int I01 = Iy1 * NX + Ix0;
	int I11 = Iy1 * NX + Ix1;
	
	
	double bX[4] = {Hf(dx), Hf(1-dx), Hg(dx), -Hg(1-dx)};
	double bY[4] = {Hf(dy), Hf(1-dy), Hg(dy), -Hg(1-dy)};
	
	
	double b[4][4] = {
						bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
						bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
						bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
						bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
					};
	
	
	
	*x2	=		  b[0][0]*(Hx[I00] + Ix0Wrap*xJump)
				+ b[0][1]*(Hx[I10] + Ix1Wrap*xJump)
				+ b[1][0]*(Hx[I01] + Ix0Wrap*xJump) 
				+ b[1][1]*(Hx[I11] + Ix1Wrap*xJump)
				
				+ (b[0][2]* Hx[1*N+I00]  + b[0][3]* Hx[1*N+I10]  + b[1][2]* Hx[1*N+I01]  + b[1][3]* Hx[1*N+I11])  * (h)
				+ (b[2][0]* Hx[2*N+I00]  + b[2][1]* Hx[2*N+I10]  + b[3][0]* Hx[2*N+I01]  + b[3][1]* Hx[2*N+I11])  * (h)
				+ (b[2][2]* Hx[3*N+I00]  + b[2][3]* Hx[3*N+I10]  + b[3][2]* Hx[3*N+I01]  + b[3][3]* Hx[3*N+I11])  * (h*h);	
				
	*y2	=		  b[0][0]*(Hy[I00] + Iy0Wrap*yJump)
				+ b[0][1]*(Hy[I10] + Iy0Wrap*yJump)
				+ b[1][0]*(Hy[I01] + Iy1Wrap*yJump) 
				+ b[1][1]*(Hy[I11] + Iy1Wrap*yJump)
				
				+ (b[0][2]* Hy[1*N+I00]  + b[0][3]* Hy[1*N+I10]  + b[1][2]* Hy[1*N+I01]  + b[1][3]* Hy[1*N+I11])  * (h)
				+ (b[2][0]* Hy[2*N+I00]  + b[2][1]* Hy[2*N+I10]  + b[3][0]* Hy[2*N+I01]  + b[3][1]* Hy[2*N+I11])  * (h)
				+ (b[2][2]* Hy[3*N+I00]  + b[2][3]* Hy[3*N+I10]  + b[3][2]* Hy[3*N+I01]  + b[3][3]* Hy[3*N+I11])  * (h*h);	
	
	
}


__device__ double  device_diffeo_grad(double *Hx, double *Hy, double x, double y, int NX, int NY, double h)																							// time cost
{
	//cell index
	int Ix0 = floor(x/h);
	int Iy0 = floor(y/h);
	int Ix1 = Ix0 + 1;
	int Iy1 = Iy0 + 1;
	
	//dx, dy
	double dx = x/h - Ix0;
	double dy = y/h - Iy0;
		
	long int N = NX*NY;
	
	//warping
	int Ix0Wrap = ((int)((Ix0 + 100 * NX)/NX) - 100);
	int Iy0Wrap = ((int)((Iy0 + 100 * NY)/NY) - 100);
	int Ix1Wrap = ((int)((Ix1 + 100 * NX)/NX) - 100);
	int Iy1Wrap = ((int)((Iy1 + 100 * NY)/NY) - 100);
	
	//jump on warping
	double xJump = NX*h;
	double yJump = NY*h;
	
	Ix0 = (Ix0+100 * NX)%NX;
	Iy0 = (Iy0+100 * NY)%NY;
	Ix1 = (Ix1+100 * NX)%NX;
	Iy1 = (Iy1+100 * NY)%NY;
	
		
	int I00 = Iy0 * NX + Ix0;
	int I10 = Iy0 * NX + Ix1;
	int I01 = Iy1 * NX + Ix0;
	int I11 = Iy1 * NX + Ix1;
	
	double Xx, Xy, Yx, Yy;
	
	double bX[4] = {Hfx(dx), -Hfx(1-dx), Hgx(dx), Hgx(1-dx)};
	double bY[4] = {Hf(dy), Hf(1-dy), Hg(dy), -Hg(1-dy)};
	
	
	double b[4][4] = {
						bX[0]*bY[0], bX[1]*bY[0], bX[2]*bY[0], bX[3]*bY[0],
						bX[0]*bY[1], bX[1]*bY[1], bX[2]*bY[1], bX[3]*bY[1],
						bX[0]*bY[2], bX[1]*bY[2], bX[2]*bY[2], bX[3]*bY[2],
						bX[0]*bY[3], bX[1]*bY[3], bX[2]*bY[3], bX[3]*bY[3]
					};
	
	
	
	Xx	=		(  b[0][0]*(Hx[I00] + Ix0Wrap*xJump)
				+ b[0][1]*(Hx[I10] + Ix1Wrap*xJump)
				+ b[1][0]*(Hx[I01] + Ix0Wrap*xJump) 
				+ b[1][1]*(Hx[I11] + Ix1Wrap*xJump)
				
				+ (b[0][2]* Hx[1*N+I00]  + b[0][3]* Hx[1*N+I10]  + b[1][2]* Hx[1*N+I01]  + b[1][3]* Hx[1*N+I11])  * (h)
				+ (b[2][0]* Hx[2*N+I00]  + b[2][1]* Hx[2*N+I10]  + b[3][0]* Hx[2*N+I01]  + b[3][1]* Hx[2*N+I11])  * (h)
				+ (b[2][2]* Hx[3*N+I00]  + b[2][3]* Hx[3*N+I10]  + b[3][2]* Hx[3*N+I01]  + b[3][3]* Hx[3*N+I11])  * (h*h) ) / h;	
				
	Yx	=		(  b[0][0]*(Hy[I00] + Iy0Wrap*yJump)
				+ b[0][1]*(Hy[I10] + Iy0Wrap*yJump)
				+ b[1][0]*(Hy[I01] + Iy1Wrap*yJump) 
				+ b[1][1]*(Hy[I11] + Iy1Wrap*yJump)
				
				+ (b[0][2]* Hy[1*N+I00]  + b[0][3]* Hy[1*N+I10]  + b[1][2]* Hy[1*N+I01]  + b[1][3]* Hy[1*N+I11])  * (h)
				+ (b[2][0]* Hy[2*N+I00]  + b[2][1]* Hy[2*N+I10]  + b[3][0]* Hy[2*N+I01]  + b[3][1]* Hy[2*N+I11])  * (h)
				+ (b[2][2]* Hy[3*N+I00]  + b[2][3]* Hy[3*N+I10]  + b[3][2]* Hy[3*N+I01]  + b[3][3]* Hy[3*N+I11])  * (h*h) ) / h;	
				
				
	double dX[4] = {Hf(dx), Hf(1-dx), Hg(dx), -Hg(1-dx)};
	double dY[4] = {Hfx(dy), -Hfx(1-dy), Hgx(dy), Hgx(1-dy)};
	
	double d[4][4] = {
						dX[0]*dY[0], dX[1]*dY[0], dX[2]*dY[0], dX[3]*dY[0],
						dX[0]*dY[1], dX[1]*dY[1], dX[2]*dY[1], dX[3]*dY[1],
						dX[0]*dY[2], dX[1]*dY[2], dX[2]*dY[2], dX[3]*dY[2],
						dX[0]*dY[3], dX[1]*dY[3], dX[2]*dY[3], dX[3]*dY[3]
					};
					
	Xy	=		( d[0][0]*(Hx[I00] + Ix0Wrap*xJump)
				+ d[0][1]*(Hx[I10] + Ix1Wrap*xJump)
				+ d[1][0]*(Hx[I01] + Ix0Wrap*xJump) 
				+ d[1][1]*(Hx[I11] + Ix1Wrap*xJump)
				
				+ (d[0][2]* Hx[1*N+I00]  + d[0][3]* Hx[1*N+I10]  + d[1][2]* Hx[1*N+I01]  + d[1][3]* Hx[1*N+I11])  * (h)
				+ (d[2][0]* Hx[2*N+I00]  + d[2][1]* Hx[2*N+I10]  + d[3][0]* Hx[2*N+I01]  + d[3][1]* Hx[2*N+I11])  * (h)
				+ (d[2][2]* Hx[3*N+I00]  + d[2][3]* Hx[3*N+I10]  + d[3][2]* Hx[3*N+I01]  + d[3][3]* Hx[3*N+I11])  * (h*h) ) / h;	
				
	Yy	=		( d[0][0]*(Hy[I00] + Iy0Wrap*yJump)
				+ d[0][1]*(Hy[I10] + Iy0Wrap*yJump)
				+ d[1][0]*(Hy[I01] + Iy1Wrap*yJump) 
				+ d[1][1]*(Hy[I11] + Iy1Wrap*yJump)
				
				+ (d[0][2]* Hy[1*N+I00]  + d[0][3]* Hy[1*N+I10]  + d[1][2]* Hy[1*N+I01]  + d[1][3]* Hy[1*N+I11])  * (h)
				+ (d[2][0]* Hy[2*N+I00]  + d[2][1]* Hy[2*N+I10]  + d[3][0]* Hy[2*N+I01]  + d[3][1]* Hy[2*N+I11])  * (h)
				+ (d[2][2]* Hy[3*N+I00]  + d[2][3]* Hy[3*N+I10]  + d[3][2]* Hy[3*N+I01]  + d[3][3]* Hy[3*N+I11])  * (h*h) ) / h;	
					
					
	return Xx*Yy - Xy*Yx;				
}





/******************************************************************/
/*******************************************************************
*							   Old								   *
*******************************************************************/
/******************************************************************/



void hermite_interpolation_test()
{
}


__global__ void kernel_hermite_interpolation(double *H, double *F, int NXH, int NYH, int NXF, int NYF, double hH, double hF)
{
}





