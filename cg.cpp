#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "Timer.h"

using namespace std;

// test function
// prints the gird of size nx ny ( including boundary points ) to the given ofstream
void gridOutputToFile( int rank, double* gird, int nx, int ny ){
	string s = std::to_string( rank );
	string path = "./out";
	path.append( s );
	path.append( ".txt" );
	
	ofstream out;
	out.open( path, ios_base::app );
	out.precision(4);
	out.setf( std::ios::fixed, std:: ios::floatfield );
	out << setw(4);
	out << "----------------  gird -----------------------" << endl;
	for( int i=ny-1; i>=0; i-- ){
		if( i == 0 || i == ny-2 ){
			for( int k=0; k<nx; k++ ){
				out << "-------";
			}
			out << endl;
		}
		for( int j=0; j<nx; j++ ){
			if( j == 1 || j == nx-1 ){
				out << " | ";
			}
			out << gird[i*nx+j] << " ";
		}
		out << endl;
	}
	out << endl;
	out.close();
}

// distributs ny inner gird points to numProcessors Prozessor as equaly as possible
void length( int numProcessors, int ny, int *array ){
	int rest = ny % numProcessors;
	int y = (ny-rest) / numProcessors;
	for( int i=0; i<numProcessors; i++ ){
		array[i] = y;
		if( rest > 0){
			array[i]++;
			rest--;
		}
	}
}

// computes the residuum of gird and f_x_y with the given stencile. nx and ny are the hole number of points in the x/y-dimension
double residuum(double* __restrict grid, double* __restrict f_x_y, double stencile_hor, double stencile_vert, double stencile_mid, int nx, int ny, int nyAlt, int rank, int numberProcesses){
	double residuum = 0.0;
	double sum = 0.0;
	double red = 0.0;
	
	// send boundry condition with mpi
	if( (rank > 0) && (rank < numberProcesses-1) ){
		// send "downwards"
		MPI_Status status; 
		MPI_Bsend( grid+nx, nx, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD );
		MPI_Recv(  grid,    nx, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status );
		// send "upwards"
		MPI_Status status1;
		MPI_Bsend( grid+(ny-2)*nx, nx, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD );
		MPI_Recv(  grid+(ny-1)*nx, nx, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &status1 );
	}
	if( rank == numberProcesses-1 ){
		MPI_Status status;
		MPI_Bsend( grid+nx, nx, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD );
		MPI_Recv(  grid,    nx, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status );
	}
	if( rank == 0 ){
		MPI_Status status;
		MPI_Bsend( grid+(ny-2)*nx, nx, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD );
		MPI_Recv(  grid+(ny-1)*nx, nx, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &status );
	}
	
	for(int i=1; i<ny-1; i++){
		for(int k=1; k<nx-1; k++){
			double temp =  f_x_y[i*nx+k] - (stencile_vert*(grid[i*nx+(k-1)]+grid[i*nx+(k+1)]) + stencile_hor*(grid[(i-1)*nx+k]+grid[(i+1)*nx+k]) + (stencile_mid*grid[i*nx+k]));
			red += temp*temp;
		}
	}
	MPI_Allreduce(&red, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	residuum = sqrt( sum / ((double)((nx-2)*(nyAlt-2))));
	return residuum;
}


// computes the skalarprodukt of a and b. nx and ny are the hole number of points in x/y-dimension
double skalarprodukt(double* __restrict a, double* __restrict b, int nx, int ny){
	double skalarprodukt = 0;

	for(int i=1; i<ny-1; i++){
		for(int j=1; j<nx-1; j++){
			skalarprodukt += a[i*nx+j] * b[i*nx+j];
		}
	}
	return skalarprodukt;
}


void cg_parallel(double* __restrict f_x_y, double* __restrict grid, int nx, int nyAlt, int ny, double stencile_hor, double stencile_vert, double stencile_mid, double* __restrict r, double epsilon, double* __restrict d, int c, double* __restrict z, int numberProcesses, int rank){
	double alpha = 0.0;
	double delta_1 = 0.0;
	double betta = 0.0;
	double delta_0 = 0.0;

	// 1)
	for(int i=1; i<ny-1; i++){
		for(int j=1; j<nx-1; j++){
			r[i*nx+j] = f_x_y[i*nx+j] - (stencile_vert*(grid[i*nx+(j-1)]+grid[i*nx+(j+1)]) + stencile_hor*(grid[(i-1)*nx+j]+grid[(i+1)*nx+j]) + (stencile_mid*grid[i*nx+j]));
		}
	}

	// 2)
	double temp1 = skalarprodukt(r, r, nx, ny);
	MPI_Allreduce(&temp1, &delta_0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// 3)
	if(residuum(grid, f_x_y, stencile_hor, stencile_vert, stencile_mid, nx, ny, nyAlt, rank, numberProcesses) < epsilon){
		cout << "number of needed iterations: 0" << endl;
		return;
	}

	// 4)
	for(int i=1; i<ny-1; i++){
		for(int j=1; j<nx-1; j++){
			d[i*nx+j] = r[i*nx+j];
		}
	}

	
	// 5-15)
	for(int k=1; k<=c; k++){
		// send boundry condition with mpi
		if( (rank > 0) && (rank < numberProcesses-1) ){
			// send "downwards"
			MPI_Status status; 
			MPI_Bsend( d+nx, nx, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD );
			MPI_Recv(  d,    nx, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status );
			// send "upwards"
			MPI_Status status1;
			MPI_Bsend( d+(ny-2)*nx, nx, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD );
			MPI_Recv(  d+(ny-1)*nx, nx, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &status1 );
		}
		if( rank == numberProcesses-1 ){
			MPI_Status status;
			MPI_Bsend( d+nx, nx, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD );
			MPI_Recv(  d,    nx, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status );
		}
		if( rank == 0 ){
			MPI_Status status;
			MPI_Bsend( d+(ny-2)*nx, nx, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD );
			MPI_Recv(  d+(ny-1)*nx, nx, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &status );
		}
		
		// 6)
		for(int i=1; i<ny-1; i++){
			for(int j=1; j<nx-1; j++){
				z[i*nx+j] = stencile_vert*(d[i*nx+(j-1)]+d[i*nx+(j+1)]) + stencile_hor*(d[(i-1)*nx+j]+d[(i+1)*nx+j]) + (stencile_mid*d[i*nx+j]);
			}
		}

		// 7)
		double temp2 = skalarprodukt(d, z, nx, ny);
		double skalarprodukt_d_z;
		MPI_Allreduce(&temp2, &skalarprodukt_d_z, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		alpha = delta_0 / skalarprodukt_d_z;

		// 8)
		for(int i=1; i<ny-1; i++){
			for(int j=1; j<nx-1; j++){
				grid[i*nx+j] = grid[i*nx+j] + alpha * d[i*nx+j];
			}
		}

		// 9)
		for(int i=1; i<ny-1; i++){
			for(int j=1; j<nx-1; j++){
				r[i*nx+j] = r[i*nx+j] - alpha * z[i*nx+j];
			}
		}

		// 10)
		double temp3 = skalarprodukt(r, r, nx, ny);
		MPI_Allreduce(&temp3, &delta_1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// 11)
		if(residuum(grid, f_x_y, stencile_hor, stencile_vert, stencile_mid, nx, ny, nyAlt, rank, numberProcesses) < epsilon){
			cout << "number of needed iterations: " << k << endl;
			return;
		}

		// 12)
		betta = delta_1/delta_0;

		// 13)
		for(int i=1; i<ny-1; i++){
			for(int j=1; j<nx-1; j++){
				int a = i*nx+j;
				d[a] = r[a] + betta * d[a];
			}
		}

		// 14)
		delta_0 = delta_1;
	}
	cout << "number of needed iterations: " << c << endl;

}

// parallel version of CG
void parallelCG(char **argv, int numberProcesses, int rank){
	
	//================================= DEFINING AND INITALISASING OF VARIABLES AND CONSTANTS ================================================
	//
	int nx = 0;				// number of grid intervals in x direction including boundary points
	int	ny = 0;				// number of grid intervals in y direction including boundary points
	int c = 0;				// number of maximum interations of cg 
	double epsilon = 0.0;	// stop condition
	nx = atoi(argv[1]);
	ny = atoi(argv[2]);
	c = atoi(argv[3]);
	epsilon = atof(argv[4]);

	// Anlegen der Konstanten:
	double const h_x = 2.0 / ((double)(nx));			// mesh size in x dimension
	double const h_y = 1.0 / ((double)(ny));			// mesh size in y dimension
	double const stencile_vert = (-1.0) / (h_x*h_x);	// stencile for west and east
	double const stencile_hor  = (-1.0) / (h_y*h_y);	// stencile for north and south
	double const stencile_mid  = ( 2.0  / (h_x*h_x)) 
	+ ( 2.0 / (h_y*h_y) ) + 4.0*M_PI*M_PI;				// stencile for the middle
	int displs[numberProcesses];						// displasment of data in the output grid for the gatherv function
	int numberGridPoints[numberProcesses];				// number of inner grid points in y dimension each processor computes on
	length( numberProcesses, ny-1, numberGridPoints );
	int nyNeu = numberGridPoints[rank]+1; 				// total number of grid intervalls in y dimension the actuall processor computes on
	int startY = 1;										// number of the first y element the actuall processor computes on 
	for(int i=0; i<rank; i++){
		startY += numberGridPoints[i];
	}

	//========================================================= MEMORY ALOCATION ==========================================================
	//
	// allocate memory for f_x_y:
	double *f_x_y = NULL;
	f_x_y = (double*)(calloc((nx+1)*(nyNeu+1), sizeof(double)));
	if(f_x_y == NULL){
		perror( "malloc" );
		exit(EXIT_FAILURE);
	}

	// allocate memory for the grid:
	double *grid = NULL;
	grid = (double*)(calloc((nx+1)*(nyNeu+1), sizeof(double)));
	if(grid == NULL){
		perror( "malloc" );
		exit(EXIT_FAILURE);
	}

	// Variablen für CG:
	double* r = NULL;
	r = (double*)(calloc((nx+1)*(nyNeu+1), sizeof(double)));
	if(r == NULL){
		perror( "malloc" );
		exit(EXIT_FAILURE);
	}	
	double* d = NULL;
	d = (double*)(calloc((nx+1)*(nyNeu+1), sizeof(double)));
	if(d == NULL){
		perror( "malloc" );
		exit(EXIT_FAILURE);
	}
	double* z = NULL;
	z = (double*)(calloc((nx+1)*(nyNeu+1), sizeof(double)));
	if(z == NULL){
		perror( "malloc" );
		exit(EXIT_FAILURE);
	}

	// allocate and attach buffer for MPI_Bsend
	int size = 0;
	MPI_Pack_size( 4*nx , MPI_DOUBLE, MPI_COMM_WORLD, &size );
	double *b = (double*)(malloc( size+4*MPI_BSEND_OVERHEAD ));
	if( b == NULL ){
		perror( "malloc" );
		exit( EXIT_FAILURE );
	}
	MPI_Buffer_attach( b, size+2*MPI_BSEND_OVERHEAD );
	
	//======================================= INITALISATION =======================================================
	//
	// Initialisiation of f_x_y
	for(int i=0; i<nyNeu; i++){
		for(int k=0; k<=nx; k++){
			f_x_y[(i+1)*(nx+1)+k] = 4.0*M_PI*M_PI*sin(2.0*M_PI*(2.0/(double)(nx))*k)*sinh(2.0*M_PI*(1.0/(double)(ny))*(i+startY));
			grid[i*(nx+1)+k] = 0.0;
		}
	}

	// only the last processor must set the upper boundry conditions
	if(rank == numberProcesses-1){
		for(int k=0; k<nx; k++){
			grid[nyNeu*(nx+1)+k] = sin(2.0*M_PI*(double)(k)*(2.0/(double)(nx)))*sinh(2.0*M_PI);
		}
		grid[nyNeu*(nx+1)+nx] = 0.0;
	}

	//============================================ CALCULATION AND TIME MESSURMENT ==================================================================
	//
	siwir::Timer* timer = new siwir::Timer();
	//*********************************FUNKTIONSAUFRUF***********************************
	cg_parallel(f_x_y, grid, nx+1, ny, nyNeu+1, stencile_hor, stencile_vert, stencile_mid, r, epsilon, d, c, z, numberProcesses, rank);
	double time = timer->elapsed();
	fprintf(stdout, "node: %d Timer Parallel CG: %lf\n", rank, time);

	//======================================== COLLECT LOCAL GRIDS TO ONE GRID IN RANK 0 ============================================================
	//
	// processor with rank 0 allocates a gird of size (nx+1)*(ny+1) for output
	double *gridOutput = NULL;
	if(rank == 0){
		if((gridOutput = (double*)(calloc((nx+1)*(ny+1), sizeof(double)))) == NULL){
			perror( "malloc" );
			exit(EXIT_FAILURE);
		}
		for( int i=0; i<(nx+1)*(ny+1); i++ ){
			gridOutput[i] = 0.0;
		}
	}
	
	// calculate number of grid points each processor sends to rank 0 for output
	for( int i=0; i<numberProcesses; i++ ){
		numberGridPoints[i] *= (nx+1);
	}
	displs[0]=0;
	for( int i=1; i<numberProcesses; i++ ){
		displs[i] = displs[i-1]+numberGridPoints[i-1];
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gatherv(grid+(nx+1), (nx+1)*(nyNeu-1), MPI_DOUBLE, gridOutput+(nx+1), numberGridPoints, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// set boundry condition in gridOutput
	if( rank == 0 ){
		for(int k=0; k<nx; k++){
			gridOutput[ny*(nx+1)+k] = sin(2.0*M_PI*(double)(k)*(2.0/(double)(nx)))*sinh(2.0*M_PI);
		}
		gridOutput[ny*(nx+1)+nx] = 0.0;
	}

	//=========================================================== WRITE OUTPUT TO FILE ================================================================
	//
	MPI_Barrier( MPI_COMM_WORLD );
	double res = residuum(grid, f_x_y, stencile_hor, stencile_vert, stencile_mid, nx+1, nyNeu+1, ny+1, rank, numberProcesses);
	if(rank == 0){
		// Oeffnen der Ausgabedatei
		ofstream out;
		out.open("./solution.txt");

		// Ausgabe fuer solution.txt
		out << "# x y u(x,y)\n";
		for(int j=0; j<=ny; j++){
			for(int t=0; t<=nx; t++){
				double f = (double)t/(double)nx;
				double q = (double)j/(double)ny;
				out << f*2 << " " << q << " " << gridOutput[j*(nx+1)+t] << "\n";
			}
			out << "\n";
		}
		out.close();
		cout << "Residuum parallel CG: " << res << endl;
	}

	//============================================================= FREES ============================================================================
	//
	if( rank == 0){
		free(gridOutput);
	}
	free(grid);
	free(f_x_y);
	free(z);
	free(d);
	free(r);
	
	// free allocated buffer 
	void *bbuf = NULL;
	int bs = 0;
	if( MPI_Buffer_detach( &bbuf, &bs ) != MPI_SUCCESS ){
		if( bs != 0 )
		free( bbuf );
	}
}

// main: generall initialisation and calling parallelCG or serialCG
int main(int argc, char **argv){
	// Ueberpruefung, ob Eingabeparamter passen
	if(argc != 5){
		fprintf(stderr, "Usage: ./cg nx ny interations epsilon\n");
		exit(EXIT_SUCCESS);
	}

	MPI_Init(&argc, &argv);
	
	int numberProcesses = 0;
	int rank = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &numberProcesses);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Zu viele Prozessoren:
	int ny = atoi(argv[2]);
	if(numberProcesses >= ny){
		fprintf(stderr, "to many processors for this problem, reducing to %d processors\n", ny-1 );
		numberProcesses = ny-1;
	}

	if(numberProcesses == 1){
		printf( "executing serial version of cg\n");
		serialCG(argv);
	}
	else{
		if( rank == 0 )
			printf( "executing parallel version of cg\n");
		parallelCG(argv, numberProcesses, rank);
	}
	MPI_Finalize();
}
