#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <omp.h>

using namespace std;
int** matMult_cuBLAS(int** A, int** B, int rowSizeA, int colSizeA, int colSizeB, cublasHandle_t cuHandle);
static void Strassen_power(int** A, int** B, int** C, const int Index_Ax, const int Index_Bx, const int Index_By);
static void Standard_power(int** A, int** B, int** C, const int Index_Ax, const int Index_Bx, const int Index_By);
static void Coppersmith_power(int** A, int** B, int** C, const int Index_Ax, const int Index_Bx, const int Index_By);

//Matrix multiplication
int** matMult_cuBLAS(int** A, int** B, int rowSizeA, int colSizeA, int colSizeB, cublasHandle_t cuHandle) {
	// 1.Define result matrix
	int** C = new int* [rowSizeA];
	for (long i = 0; i < rowSizeA; i++)
		C[i] = new int[colSizeB];

	// 2.Make room in memory for the matrix to be computed
	double* h_A = (double*)malloc(rowSizeA * colSizeA * sizeof(double));
	double* h_B = (double*)malloc(colSizeA * colSizeB * sizeof(double));
	double* h_C = (double*)malloc(rowSizeA * colSizeB * sizeof(double));

	// 3.Initialize the calculation matrix h_A and h_B
	for (int i = 0; i < rowSizeA; i++)
		for (int j = 0; j < colSizeA; j++)
			h_A[i * colSizeA + j] = (double)A[i][j];
	for (int i = 0; i < colSizeA; i++)
		for (int j = 0; j < colSizeB; j++)
			h_B[i * colSizeB + j] = (double)B[i][j];

	// 4.Make room in the video memory for the matrix to be calculated and the result matrix
	double* d_A, * d_B, * d_C;
	cudaMalloc((void**)&d_A, rowSizeA * colSizeA * sizeof(double));
	cudaMalloc((void**)&d_B, colSizeA * colSizeB * sizeof(double));
	cudaMalloc((void**)&d_C, rowSizeA * colSizeB * sizeof(double));

	// 5.Copy CPU data to GPU
	cublasSetVector(rowSizeA * colSizeA, sizeof(double), h_A, 1, d_A, 1);
	cublasSetVector(colSizeA * colSizeB, sizeof(double), h_B, 1, d_B, 1);

	// 6.The parameters passed into the matrix multiplication function, please refer to the function manual for the specific meaning. And perform the kernel function, matrix multiplication
	double a = 1; double b = 0;
	cublasDgemm(cuHandle, CUBLAS_OP_T, CUBLAS_OP_T, rowSizeA, colSizeB, colSizeA, &a, d_A, colSizeA, d_B, colSizeB, &b, d_C, rowSizeA);

	// 7.Get the result of the operation from the GPU to the CPU
	cublasGetVector(rowSizeA * colSizeB, sizeof(double), d_C, 1, h_C, 1);

	// 8.Assign the result to the result matrix
	for (int i = 0; i < rowSizeA; i++)
		for (int j = 0; j < colSizeB; j++)
			C[i][j] = static_cast<int>(h_C[j * rowSizeA + i]);

	// 9.Clear used memory
	free(h_A); free(h_B); free(h_C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

	return C;
}


/*
 * matA M*K
 * matB K*N
 * matC M*N
 * matC = matA * matB
 * S1 = A21 + A22     T1 = B12 - B11
 * S2 = S1 - A11      T2 = B22 - T1
 * S3 = A11 - A21     T3 = B22 - B12
 * S4 = A12 - S2      T4 = T2 - B21
 * M1 = A11 * B11     U1 = M1 + M2
 * M2 = A12 * B21     U2 = M1 + M6
 * M3 = S4 * B22      U3 = U2 + M7
 * M4 = A22 * T4      U4 = U2 + M5
 * M5 = S1 * T1       U5 = U4 + M3
 * M6 = S2 * T2       U6 = U3 - U4
 * M7 = S3 * T3       U7 = U3 + M5
 * C11 = U1
 * C12 = U5
 * C21 = U6
 * C22 = U7
 */
static void Coppersmith_power(int** A11, int** A12, int** A21, int** A22, int** B11, int** B12, int** B21, int** B22, int** C, const int Index_Ax, const int Index_Bx, const int Index_By)//????
{
	int n = Index_Ax;
	//Set the CPU timing function and start timing
	clock_t   start, finish;
	start = clock();
	cout << "Coppersmith Calculation Start" << endl;
	// Create and initialize CUBLAS library objects
	cublasHandle_t cuHandle;
	cublasStatus_t status = cublasCreate(&cuHandle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
			cout << "CUBLAS object instantiation error" << endl;
		}
		getchar();
	}

	//Use four matrices such as M_1, M_2, M_3, M_4 to represent 7 matrices of M1-M7;
	int** M_1 = new int* [Index_Ax / 2];
	int** M_2 = new int* [Index_Ax / 2];
	int** M_3 = new int* [Index_Ax / 2];
	int** M_4 = new int* [Index_Ax / 2];
	int** M_5 = new int* [Index_Ax / 2];
	int** M_6 = new int* [Index_Ax / 2];
	int** M_7 = new int* [Index_Ax / 2];
	for (int i = 0; i < Index_Ax / 2; i++)
	{
		M_1[i] = new int[Index_Bx / 2];
		M_2[i] = new int[Index_Bx / 2];
		M_3[i] = new int[Index_Bx / 2];
		M_4[i] = new int[Index_Bx / 2];
		M_5[i] = new int[Index_Bx / 2];
		M_6[i] = new int[Index_Bx / 2];
		M_7[i] = new int[Index_Bx / 2];
	}
	int temp_A = Index_Ax / 2;
	int temp_B = Index_Bx / 2;
	/*
	 * S1 = A21 + A22     T1 = B12 - B11
	 * S2 = S1 - A11      T2 = B22 - T1
	 * S3 = A11 - A21     T3 = B22 - B12
	 * S4 = A12 - S2      T4 = T2 - B21
	 * M1 = A11 * B11     U1 = M1 + M2
	 * M2 = A12 * B21     U2 = M1 + M6
	 * M3 = S4 * B22      U3 = U2 + M7
	 * M4 = A22 * T4      U4 = U2 + M5
	 * M5 = S1 * T1       U5 = U4 + M3
	 * M6 = S2 * T2       U6 = U3 - U4
	 * M7 = S3 * T3       U7 = U3 + M5
	 * C11 = M1 + M2 = A11 * B11 +  A12 * B21 //M1, M2
	 * U2 = M1 + M6 = M1 + (A21 + A22 - A11) * (B22 - B12 + B11) //M1, M6, U2
	 * U3 = U2 + M7 = U2 + (A11 - A21) * (B22 - B12) // U2, U3, M7
	 * U4 = U2 + M5 = U2 + (A21 + A22) * (B12 - B11) // U2, U3, M5, U4
	 * C22 = U3 + M5 // U3, M5
	 * C21 = U3 - M4 // U3, M4
	 * C12 = U4 + M3 = U4 + S4 * B22 =U4 + (A12 - A21 - A22 + A11) * B22 // U4, M3
	 */

	for (int i = 0; i < Index_Ax / 2; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < Index_By / 2; j++)
		{
			//S1 = A21 + A22     T1 = B12 - B11
			M_1[i][j] = A21[i][j] + A22[i][j];
			M_2[i][j] = B12[i][j] - B11[i][j];
			//S3 = A11 - A21     T3 = B22 - B12
			M_3[i][j] = A11[i][j] - A21[i][j];
			M_4[i][j] = B22[i][j] - B12[i][j];
			//S2 = S1 - A11      T2 = B22 - T1
			M_5[i][j] = M_1[i][j] - A11[i][j];
			M_6[i][j] = B22[i][j] - M_2[i][j];
		}
	}
	//Parallel section 1
	M_1 = matMult_cuBLAS(M_1, M_2, Index_Ax / 2, Index_Bx / 2, Index_By / 2, cuHandle);//M5 = S1 * T1
	M_3 = matMult_cuBLAS(M_3, M_4, Index_Ax / 2, Index_Bx / 2, Index_By / 2, cuHandle);//M7 = S3 * T3
	M_7 = matMult_cuBLAS(M_5, M_6, Index_Ax / 2, Index_Bx / 2, Index_By / 2, cuHandle);//M6 = S2 * T2

	//Parallel section 2
	M_2 = matMult_cuBLAS(A11, B11, Index_Ax / 2, Index_Bx / 2, Index_By / 2, cuHandle);//M1 = A11 * B11   
	M_4 = matMult_cuBLAS(A12, B21, Index_Ax / 2, Index_Bx / 2, Index_By / 2, cuHandle);//M2 = A12 * B21

	//M_1=M5;M_2=M1;M_3=M7;M_4=M2;M_5=S2;M_6=T2;M_7=M6
	for (int i = 0; i < Index_Ax / 2; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < Index_By / 2; j++)
		{
			//C11 = M1 + M2
			C[i][j] = M_2[i][j] + M_4[i][j];
			//U2 = M1 + M6
			M_7[i][j] = M_2[i][j] + M_7[i][j];
			//U3 = U2 + M7
			M_3[i][j] = M_7[i][j] + M_3[i][j];
			//C22 = U3 + M5
			C[i + temp_A][j + temp_A] = M_3[i][j] + M_1[i][j];
			//U4 = U2 + M5
			M_1[i][j] = M_7[i][j] + M_1[i][j];
			//S4 = A12 - S2      
			M_2[i][j] = A12[i][j] - M_5[i][j];
			//T4 = T2 - B21
			M_4[i][j] = M_6[i][j] - B21[i][j];
		}
	}
	//M_1=U4;M_2=S4;M_3=U3;M_4=T4;M_5=S2;M_6=T2;M_7=U2
	M_5 = matMult_cuBLAS(M_2, B22, Index_Ax / 2, Index_Bx / 2, Index_By / 2, cuHandle);//M3 = S4 * B22
	M_6 = matMult_cuBLAS(A22, M_4, Index_Ax / 2, Index_Bx / 2, Index_By / 2, cuHandle);//M4 = A22 * T4

	//M_1=U4;M_2=S4;M_3=U3;M_4=T4;M_5=M3;M_6=M4;M_7=U2
	for (int i = 0; i < Index_Ax / 2; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < Index_By / 2; j++)
		{
			//C21 = U3 - M4
			C[i + temp_A][j] = M_3[i][j] - M_6[i][j];
			//C12 = U4 + M3
			C[i][j + temp_A] = M_1[i][j] + M_5[i][j];
		}
	}

	// Free CUBLAS library objects and memory
	cublasDestroy(cuHandle);
	for (int i = 0; i < Index_Ax / 2; i++)
	{
		delete[]M_5[i];         
		delete[]M_6[i];         
		delete[]M_1[i];        
		delete[]M_2[i];       
		delete[]M_3[i];       
		delete[]M_4[i];    
	}
	delete[]M_5;
	delete[]M_6;
	delete[]M_1;
	delete[]M_2;
	delete[]M_3;
	delete[]M_4;
	//End CPU timer
	finish = clock();

	//Print run time
	cout << "Coppersmith Time (total time the program runs) is：" << double(finish - start) / 1000 << "s" << endl;
	cout << "Result output" << endl;

	// Output matrix C, that is, the result of the operation
	ofstream fout("Coppersmith.txt");
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			fout << C[i][j] << " ";
		fout << endl;
	}
	cout << endl;
	cout << endl;
	cout << endl;
	cout << endl;
	/*for (int i = 0; i < n ; i++)
	{
		for (int j = 0; j < n ; j++)
		{
			if (i < n / 2 && j < n / 2)
				cout << A11[i][j] << " ";
			if (i < n / 2 && j >= n / 2)
				cout << A12[i][j - n / 2] << " ";
			if (i >= n / 2 && j < n / 2)
				cout << A21[i - n / 2][j] << " ";
			if (i >= n / 2 && j >= n / 2)
				cout << A22[i - n / 2][j - n / 2] << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << endl;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i < n / 2 && j < n / 2)
				cout << B11[i][j] << " ";
			if (i < n / 2 && j >= n / 2)
				cout << B12[i][j - n / 2] << " ";
			if (i >= n / 2 && j < n / 2)
				cout << B21[i - n / 2][j] << " ";
			if (i >= n / 2 && j >= n / 2)
				cout << B22[i - n / 2][j - n / 2] << " ";
		}
		cout << endl;
	}*/

}
//Experimental Contrast Function
static void Standard_power(int** A, int** B, int** C, const int Index_Ax, const int Index_Bx, const int Index_By)
{
	int n = Index_Ax;
	cout << "Enter standard procedure" << endl;
	//Set the CPU timing function and start timing
	clock_t   start1, finish1;
	start1 = clock();
	cout << "Standard Calculation Start" << endl;
	// Create and initialize CUBLAS library objects
	cublasHandle_t cuHandle;
	cublasStatus_t status = cublasCreate(&cuHandle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
			cout << "CUBLAS object instantiation error" << endl;
		}
		getchar();
	}
	C = matMult_cuBLAS(A, B, n, n, n, cuHandle);
	finish1 = clock();
	//Print run time
	cout << "Standard Timing (total time the program runs) is：" << double(finish1 - start1) / 1000 << "s" << endl;
	cout << "Result output" << endl;
	// Output matrix C, that is, the result of the operation
	ofstream fout("Standard.txt");
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			fout << C[i][j] << " ";
		fout << endl;
	}
	// Free CUBLAS library objects
	cublasDestroy(cuHandle);
}

int main()
{
	int n = 0;
	cout << "Please enter the order of the matrix" << endl;
	cin >> n;
	//odd-order matrix
	if (n % 2 != 0)
	{
		n = n + 1;
		int** A11 = new int* [n / 2];
		int** A12 = new int* [n / 2];
		int** A21 = new int* [n / 2];
		int** A22 = new int* [n / 2];
		int** B11 = new int* [n / 2];
		int** B12 = new int* [n / 2];
		int** B21 = new int* [n / 2];
		int** B22 = new int* [n / 2];
		int** C = new int* [n];
		for (int i = 0; i < n; i++)
		{
			if (i < n / 2)
			{
				A11[i] = new int[n / 2];
				A12[i] = new int[n / 2];
				A21[i] = new int[n / 2];
				A22[i] = new int[n / 2];
				B11[i] = new int[n / 2];
				B12[i] = new int[n / 2];
				B21[i] = new int[n / 2];
				B22[i] = new int[n / 2];
			}
			C[i] = new int[n];
		}
		ifstream fin1("A.txt");
		for (int i = 0; i < n - 1; i++)
			for (int j = 0; j < n - 1; j++)
			{
				if (i < n / 2 && j < n / 2)
					fin1 >> A11[i][j];
				if (i < n / 2 && j >= n / 2)
					fin1 >> A12[i][j - n / 2];
				if (i >= n / 2 && j < n / 2)
					fin1 >> A21[i - n / 2][j];
				if (i >= n / 2 && j >= n / 2)
					fin1 >> A22[i - n / 2][j - n / 2];
			}
		ifstream fin2("B.txt");
		for (int i = 0; i < n - 1; i++)
			for (int j = 0; j < n - 1; j++)
			{
				if (i < n / 2 && j < n / 2)
					fin1 >> B11[i][j];
				if (i < n / 2 && j >= n / 2)
					fin1 >> B12[i][j - n / 2];
				if (i >= n / 2 && j < n / 2)
					fin1 >> B21[i - n / 2][j];
				if (i >= n / 2 && j >= n / 2)
					fin1 >> B22[i - n / 2][j - n / 2];
			}

		cout << "Program has read data" << endl;

		Coppersmith_power(A11, A12, A21, A22, B11, B12, B21, B22, C, n, n, n);
		//Standard_power(A, B, C, n, n, n);

		for (int i = 0; i < n; i++)
		{
			if (i < n / 2)
			{
				delete[]A11[i];
				delete[]A12[i];
				delete[]A21[i];
				delete[]A22[i];
				delete[]B11[i];
				delete[]B12[i];
				delete[]B21[i];
				delete[]B22[i];
			}
			delete[]C[i];
		}
		delete[]A11;
		delete[]A12;
		delete[]A21;
		delete[]A22;
		delete[]B11;
		delete[]B12;
		delete[]B21;
		delete[]B22;
		delete[]C;
		
	}
	//even-order matrix
	else
	{
		ifstream fin1("A.txt");
		int** A11 = new int* [n / 2];
		int** A12 = new int* [n / 2];
		int** A21 = new int* [n / 2];
		int** A22 = new int* [n / 2];
		int** B11 = new int* [n / 2];
		int** B12 = new int* [n / 2];
		int** B21 = new int* [n / 2];
		int** B22 = new int* [n / 2];
		int** C = new int* [n];
		for (int i = 0; i < n ; i++)
		{
			if (i < n / 2)
			{
				A11[i] = new int[n / 2];
				A12[i] = new int[n / 2];
				A21[i] = new int[n / 2];
				A22[i] = new int[n / 2];
				B11[i] = new int[n / 2];
				B12[i] = new int[n / 2];
				B21[i] = new int[n / 2];
				B22[i] = new int[n / 2];
			}
			C[i] = new int[n];
			for (int j = 0; j < n; j++)
			{
				if (i < n / 2 && j < n / 2)
					fin1 >> A11[i][j];
				if (i < n / 2 && j >= n / 2)
					fin1 >> A12[i][j - n / 2];
				if (i >= n / 2 && j < n / 2)
					fin1 >> A21[i - n / 2][j];
				if (i >= n / 2 && j >= n / 2)
					fin1 >> A22[i - n / 2][j - n / 2];
				C[i][j] = 0;
			}
		}
		ifstream fin2("B.txt");
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
			{
				if (i < n / 2 && j < n / 2)
					fin1 >> B11[i][j];
				if (i < n / 2 && j >= n / 2)
					fin1 >> B12[i][j - n / 2];
				if (i >= n / 2 && j < n / 2)
					fin1 >> B21[i - n / 2][j];
				if (i >= n / 2 && j >= n / 2)
					fin1 >> B22[i - n / 2][j - n / 2];
			}
		
		cout << "Program has read data" << endl;

		//Strassen_power(A11, A12, A21, A22, B11, B12, B21, B22, C, n, n, n);
		Coppersmith_power(A11, A12, A21, A22, B11, B12, B21, B22, C, n, n, n);
		//Standard_power(A, B, C, n, n, n);

		for (int i = 0; i < n; i++)
		{
			if (i < n / 2)
			{
				delete[]A11[i];
				delete[]A12[i];
				delete[]A21[i];
				delete[]A22[i];
				delete[]B11[i];
				delete[]B12[i];
				delete[]B21[i];
				delete[]B22[i];
			}
			delete[]C[i];
		}
		delete[]A11;
		delete[]A12;
		delete[]A21;
		delete[]A22;
		delete[]B11;
		delete[]B12;
		delete[]B21;
		delete[]B22;
		delete[]C;
	}
	cout << "Program ends" << endl;
	return 0;
}
	