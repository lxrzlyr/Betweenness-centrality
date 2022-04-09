#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "access.h"
#include "cublas_v2.h"
#include <time.h>
#include <omp.h>
using namespace std;

//Build a hash function to initialize the key pairs of the unordered_map
struct hash_pair {
	template <class T1, class T2>
	size_t operator()(const pair<T1, T2>& p) const
	{
		auto hash1 = hash<T1>{}(p.first);
		auto hash2 = hash<T2>{}(p.second);
		return hash1 ^ hash2;
	}
};


// Key-value comparison, the comparison definition of hash collision, needs until two custom objects are equal
struct equal_key {
	template<typename T, typename U>
	bool operator ()(const std::pair<T, U>& p1, const std::pair<T, U>& p2) const {
		return p1.first == p2.first && p1.second == p2.second;
	}
};



//The main function of the parallel area
void omp(long long** A, int n, int k, vector<double>& ret, int p);
// Read in and initialize the adjacency matrix
void uniformMat(long long**& A, long long**& B, int n, int index);
// cuBLAS implements matrix multiplication (the multiplication result is used to update the value of the B matrix)
void matMult_cuBLAS1(long long**& A, long long**& B, int n, cublasHandle_t cuHandle);
//Matrix subtraction
void Matrix_subtraction(long long**& A, long long**& B, long long**& C, int n);
//Matrix search function
double search(unordered_map<pair<int, int>, double, hash_pair, equal_key>& mp,
	long long**& tmp_A, long long**& tmp_B, long long**& C, int n,
	int index_s, double mat_ret);
//Matrix power difference function
double betfun(long long**& A, long long**& B, long long**& C, long long**& tmp_A, long long**& tmp_B,
	int n, int p, int index_s, unordered_map<pair<int, int>, double, hash_pair, equal_key>& mp,
	cublasHandle_t cuHandle);






// cuBLAS implements matrix multiplication (the multiplication result is used to update the value of the B matrix)
void matMult_cuBLAS1(long long**& A, long long**& B, int n, cublasHandle_t cuHandle)
{
	int rowSizeA = n;
	int rowSizeB = n;
	int colSizeA = n;
	int colSizeB = n;


	// 1. Make room in memory for the matrix to be calculated
	double* h_A = (double*)malloc(rowSizeA * colSizeA * sizeof(double));
	double* h_B = (double*)malloc(rowSizeB * colSizeB * sizeof(double));
	double* h_C = (double*)malloc(rowSizeA * colSizeB * sizeof(double));

	// 2. Initialize calculation matrices h_A and h_B
	for (int i = 0; i < rowSizeA; i++)
		for (int j = 0; j < colSizeA; j++)
			h_A[i * colSizeA + j] = (double)A[i][j];
	for (int i = 0; i < rowSizeB; i++)
		for (int j = 0; j < colSizeB; j++)
			h_B[i * colSizeB + j] = (double)B[i][j];

	// 3. Open up space in the video memory for the matrix to be calculated and the result matrix
	double* d_A, * d_B, * d_C;
	cudaMalloc((void**)&d_A, rowSizeA * colSizeA * sizeof(double));
	cudaMalloc((void**)&d_B, rowSizeB * colSizeB * sizeof(double));
	cudaMalloc((void**)&d_C, rowSizeA * colSizeB * sizeof(double));

	// 4. Copy CPU data to GPU
	cublasSetVector(rowSizeA * colSizeA, sizeof(double), h_A, 1, d_A, 1);
	cublasSetVector(rowSizeB * colSizeB, sizeof(double), h_B, 1, d_B, 1);

	// 5. Pass the parameters into the matrix multiplication function and execute the kernel function, matrix multiplication
	double a = 1; double b = 0;
	cublasDgemm(cuHandle, CUBLAS_OP_T, CUBLAS_OP_T, rowSizeA, colSizeB, colSizeA, &a, d_A, colSizeA, d_B, colSizeB, &b, d_C, rowSizeA);

	// 6. Take the result of the operation from the GPU to the CPU
	cublasGetVector(rowSizeA * colSizeB, sizeof(double), d_C, 1, h_C, 1);

	// 7. Assign the result to the result matrix
	for (int i = 0; i < rowSizeA; i++)
		for (int j = 0; j < colSizeB; j++)
			B[i][j] = static_cast<long long>(h_C[j * rowSizeA + i]);

	// 8. Clean up used memory
	free(h_A); free(h_B); free(h_C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}


// read in and initialize the adjacency matrix
void uniformMat(long long**& A, long long**& B, int n, int index) {

	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			if (j == index || i == index)
				B[i][j] = 0;
			else
				B[i][j] = A[i][j];
		}
}


//matrix subtraction
void Matrix_subtraction(long long**& A, long long**& B, long long**& C, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			C[i][j] = A[i][j] - B[i][j];
}


//Matrix search function
double search(unordered_map<pair<int, int>, double, hash_pair, equal_key>& mp, 
	          long long**& tmp_A, long long**& tmp_B, long long**& C, int n, 
	          int index_s, double mat_ret)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if (tmp_A[i][j] != 0 && mp[{i, j}] == 0 && i != index_s && j != index_s)
			{
				if (C[i][j] == 0)
				{
					mp[{i, j}] = -1;
					mat_ret = mat_ret + 0;
				}
				else
				{
					mp[{i, j}] = (1.0 * C[i][j] / tmp_A[i][j]);
					mat_ret = mat_ret + mp[{i, j}];
				}
			}
	return mat_ret;
}


//Matrix power difference function
double betfun(long long**& A, long long**& B, long long**& C, long long**& tmp_A, long long**& tmp_B,
	         int n, int p, int index_s, unordered_map<pair<int, int>, double, hash_pair, equal_key>& mp,
	         cublasHandle_t cuHandle)
{
	double mat_ret = 0;//Set return value

//Search function
	mat_ret = search(mp, tmp_A, tmp_B, C, n, index_s, mat_ret);


	while (p > 0)
	{
		matMult_cuBLAS1(A, tmp_A, n, cuHandle);
		matMult_cuBLAS1(B, tmp_B, n, cuHandle);
		Matrix_subtraction(tmp_A, tmp_B, C, n);
		mat_ret = search(mp, tmp_A, tmp_B, C, n, index_s, mat_ret);
		p--;
	}

	mat_ret = 2 * mat_ret / ((n - 1) * (n - 2));
	return mat_ret;
}


