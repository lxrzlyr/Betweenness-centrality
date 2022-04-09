#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "access.h"
#include "cublas_v2.h"
#include <time.h>
#include <omp.h>
#include "function.h"
using namespace std;


int main()
{
	//Set the CPU timing function and start timing
	clock_t   start, finish; 
	start = clock();

	// Define matrix order
	cout << "Please enter the order of the matrix" << endl;
	int n = 0;
	cin >> n;

	//p is the probability of network connection
	double p = 0;
	cout << "Please enter the network node connection probability" << endl;
	double lambda = 2;//Average degree coefficient
	cin >> p;
	p = 1.0*lambda*log(n)/(double)max(log((n-1)*p), 1.0);//average degree of the network
	
	//Set the return array, which can be output in order
	vector<double> retarr(n);


    //GPU timing starts
	cudaEvent_t Gstart, Gstop;
	cudaEventCreate(&Gstart);
	cudaEventCreate(&Gstop);
	cudaEventRecord(Gstart, 0);

	//Open the file and read in the A matrix
	ifstream fin("matrix.txt");
	long long** A = new long long* [n];
	for (int i = 0; i < n; i++)
	{
		A[i] = new long long[n];
		for (int j = 0; j < n; j++)
			fin >> A[i][j];
	}

	

//Open the parallel area and assign a task number to the cpu
#pragma omp parallel for
	for (int k = 0; k < 8; k++)//Set the number of parallel tasks, which can be modified according to device performance
	{
		omp(A, n, k,retarr,p);
	}


	//Release the system memory occupied by the A matrix and print the result
	delete[]A;
	for(int i=0;i<n;i++)
		cout << "Nodes" << i << "betweenness centrality index is" << retarr[i] << endl;
	

	//GPU timing out
	cudaEventRecord(Gstop, 0);
	cudaEventSynchronize(Gstop);
	float elapsedTime;//event time
	cudaEventElapsedTime(&elapsedTime, Gstart, Gstop);


	//End CPU timer
	finish = clock();


	//Print run time
	cout << "CPU time (total time the program runs) is:" << double(finish - start) / 1000 << "s" << endl;
	cout << "GPU time (total time the program runs) is:" << elapsedTime / 1000 << "s" << endl;

	return 0;
}

//Parallel region main function
void omp(long long** A, int n, int k, vector<double>& ret, int p)
{

	// Create and initialize CUBLAS library objects
	cublasHandle_t cuHandle;
	cublasStatus_t status = cublasCreate(&cuHandle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
			cout << "CUBLAS object instantiation error" << endl;
		}
		getchar();
		//return EXIT_FAILURE;
	}


	//Node number to be calculated
	int tmp = 0;//task area capacity
	int start = 0;
	int end = 0;//Task area end number
	tmp = (n - n % 8) / 8;//The device supports up to 8 cores in parallel
	start = tmp * k;//Determine the starting node number of the task area
	end = start + tmp;//Determine the task area termination node number
	if (k == 7)
		end = n;//The final task area includes the calculation of the remainder


	//initialization matrix
	long long** B = new long long* [n];
	long long** C = new long long* [n];
	long long** tmp_A = new long long* [n];
	long long** tmp_B = new long long* [n];
	for (int i = 0; i < n; i++)
	{
		B[i] = new long long[n];
		C[i] = new long long[n];
		tmp_A[i] = new long long[n];
		tmp_B[i] = new long long[n];
		for (int j = 0; j < n; j++)
		{
			tmp_A[i][j] = 0;
			tmp_B[i][j] = 0;
		}
	}

	//node compute omp
	for (int index_s = start; index_s < end; index_s++)
	{
		//Matrix and register map initialization
		uniformMat(A, B, n, index_s);//Initialize the B matrix
		Matrix_subtraction(A, B, C, n);//Initialize the C matrix
		unordered_map<pair<int, int>, double, hash_pair, equal_key> mp;//Initialize the register map
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
			{
				tmp_A[i][j] = A[i][j];
				tmp_B[i][j] = B[i][j];
			}


		//Call betweenness calculation function betfun
		ret[index_s] = betfun(A, B, C, tmp_A, tmp_B, n, p, index_s, mp, cuHandle);
		//cout<<"Node "<<index_s<<"betweenness centrality is"<<ret[index_s]<<endl;



		for (unordered_map<pair<int, int>, double, hash_pair>::iterator iter = mp.begin(); iter != mp.end();)
			mp.erase(iter++);
		mp.clear();
	}

	//free memory
	delete[]B;
	delete[]C;
	delete[]tmp_A;
	delete[]tmp_B;

	// Free CUBLAS library objects
	cublasDestroy(cuHandle);
}