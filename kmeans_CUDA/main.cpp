#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
};

// CUDA kernel to calculate the Euclidean distance between two points
__device__ double distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// CUDA kernel to assign each point to the nearest centroid
__global__ void assignClusters(Point* data, Point* centroids, int* labels, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double minDist = distance(data[i], centroids[0]);
        int label = 0;
        for (int j = 1; j < k; j++) {
            double dist = distance(data[i], centroids[j]);
            if (dist < minDist) {
                minDist = dist;
                label = j;
            }
        }
        if (labels[i] != label) {
            labels[i] = label;
        }
    }
}

// CUDA kernel to calculate the new centroids based on the assigned clusters
__global__ void calculateCentroids(Point* data, Point* centroids, int* labels, int n, int k, int* clusterCounts, Point* clusterSums) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int clusterIndex = labels[i];
        atomicAdd(&clusterCounts[clusterIndex], 1);
        atomicAdd(&clusterSums[clusterIndex].x, data[i].x);
        atomicAdd(&clusterSums[clusterIndex].y, data[i].y);
    }
}

// CUDA kernel to update the centroids' positions
__global__ void updateCentroids(Point* centroids, int* clusterCounts, Point* clusterSums, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        if (clusterCounts[i] > 0) {
            centroids[i].x = clusterSums[i].x / clusterCounts[i];
            centroids[i].y = clusterSums[i].y / clusterCounts[i];
        }
    }
}

void kMeans_cuda(vector<Point>& data, int k, int max_iterations, vector<Point>& centroids) {
    int n = data.size();
    vector<int> labels(n);
    vector<int> clusterCounts(k);
    vector<Point> clusterSums(k);

    // �����豸�ڴ�
    Point* d_data;
    Point* d_centroids;
    int* d_labels;
    int* d_clusterCounts;
    Point* d_clusterSums;
    cudaMalloc((void**)&d_data, n * sizeof(Point));
    cudaMalloc((void**)&d_centroids, k * sizeof(Point));
    cudaMalloc((void**)&d_labels, n * sizeof(int));
    cudaMalloc((void**)&d_clusterCounts, k * sizeof(int));
    cudaMalloc((void**)&d_clusterSums, k * sizeof(Point));

    // �����ݸ��Ƶ��豸�ڴ�
    cudaMemcpy(d_data, data.data(), n * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids.data(), k * sizeof(Point), cudaMemcpyHostToDevice);

    // ִ��K-means����
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        assignClusters << <(n + 255) / 256, 256 >> > (d_data, d_centroids, d_labels, n, k);
        cudaDeviceSynchronize();

        cudaMemset(d_clusterCounts, 0, k * sizeof(int));
        cudaMemset(d_clusterSums, 0, k * sizeof(Point));

        calculateCentroids << <(n + 255) / 256, 256 >> > (d_data, d_centroids, d_labels, n, k, d_clusterCounts, d_clusterSums);
        cudaDeviceSynchronize();

        updateCentroids << <(k + 255) / 256, 256 >> > (d_centroids, d_clusterCounts, d_clusterSums, k);
        cudaDeviceSynchronize();
    }

    // �����ĸ��ƻ������ڴ�
    cudaMemcpy(centroids.data(), d_centroids, k * sizeof(Point), cudaMemcpyDeviceToHost);

    // �ͷ��ڴ�
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_clusterCounts);
    cudaFree(d_clusterSums);
}

int main() {
    long long head, tail, freq;
    const int num_clusters = 3;
    const int max_iters = 10;

    //��������
    srand(time(NULL));
    vector<Point> data;
    vector<vector<double>> dataArray; //��Ϊ��Ҫ��ȡ���ϵ�������Ƶ�Ρ������Ϣ������Ԥ�ȴ���һ��Ƕ������
    ifstream inFile("C:/Users/Lenovo/Desktop/xclara.csv", ios::in);
    string linestr;
    //�ж�
    if (inFile.fail())
    {
        cout << "��ȡ�ļ�ʧ��" << endl;
    }
    while (getline(inFile, linestr))//���ж�ȡ�ļ���ֱ��Ϊ��Ϊֹ
    {
        stringstream ss(linestr);//��ɶ�ά��ṹ
        string str;//ÿ���еĵ����ַ�
        vector<double> lineArray;
        while (getline(ss, str, ','))
        {
            lineArray.push_back(atof(str.c_str()));//һ��������vector����
        }
        dataArray.push_back(lineArray);//ÿһ��vector���ݶ��ŵ�strArray��ȥ
    }
    for (int i = 0; i < 16384; ++i) {
        Point p = { dataArray[i][0] , dataArray[i][1] };
        data.push_back(p);
    }
    vector<Point> centroids = initCentroids(data, num_clusters);
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    kMeans_mpi(data, num_clusters, max_iters, centroids);
    // ��ӡ����
    cout << "���յ�����:\n";
    for (const auto& centroid : centroids) {
        cout << centroid.x << ", " << centroid.y << "\n";
    }
    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "mpi�㷨��" << (tail - head) * 1000.0 / freq
    << "ms" << endl;
    return 0;
}
