#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>  
#include <sstream>
#include <windows.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>


using namespace std;

const int kVectorSize = 4;
const int kAvxVecLen = 8;

// 定义向量类型
typedef __m256i AvxVec;

struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
};

// 定义函数来计算两点之间的欧几里得距离
double distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}
double distance_sse(Point p1, Point p2) {
    __m128d dx = _mm_set_pd1(p1.x - p2.x);
    __m128d dy = _mm_set_pd1(p1.y - p2.y);
    __m128d dx2 = _mm_mul_pd(dx, dx);
    __m128d dy2 = _mm_mul_pd(dy, dy);
    __m128d sum = _mm_add_pd(dx2, dy2);
    __m128d result = _mm_sqrt_pd(sum);
    double distance[kVectorSize];
    _mm_store_pd(distance, result);
    return distance[0];
}

vector<Point> initCentroids(vector<Point>& data, int k) {
    vector<Point> centroids;
    int n = data.size();
    for (int i = 0; i < k; i++) {
        int idx = rand() % n;
        centroids.push_back(data[idx]);
    }
    return centroids;
}


void kMeans(vector<Point>& data, int k, int  max_iterations, vector<Point>& centroids) {
    int n = data.size();
    vector<int> labels(n);
    vector<int>count(k);
    // 迭代更新聚类中心
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        // 将每个点分配到最近的聚类中心
        for (int i = 0; i < n; i++) {
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

        // 更新聚类中心的位置
        for (int i = 0; i < k; i++) {
            double sumX = 0, sumY = 0;
            int count = 0;
            for (int j = 0; j < n; j++) {
                if (labels[j] == i) {
                    sumX += data[j].x;
                    sumY += data[j].y;
                    count++;
                }
            }
            if (count > 0) {
                centroids[i].x = sumX / count;
                centroids[i].y = sumY / count;
            }
        }
    }
}

void kMeans_mpi(vector<Point>& points, int num_clusters, int  max_iters,vector<Point>& centroids) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int num_points = points.size();
    // 分散数据
    const int local_num_points = num_points / size;
    vector<Point> local_points(local_num_points);
    MPI_Scatter(points.data(), 2 * local_num_points, MPI_DOUBLE, local_points.data(), 2 * local_num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 广播质心
    MPI_Bcast(centroids.data(), 2 * num_clusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int iter = 0; iter < max_iters; ++iter) {
        vector<int> labels(local_num_points);
        for (int i = 0; i < local_num_points; ++i) {
            double min_dist = distance(points[i], centroids[0]);
            for (int j = 0; j < num_clusters; ++j) {
                double d = distance(local_points[i], centroids[j]);
                if (d < min_dist) {
                    min_dist = d;
                    labels[i] = j;
                }
            }
        }

        // 收集数据
        vector<int> all_labels(num_points);
        MPI_Gather(labels.data(), local_num_points, MPI_INT, all_labels.data(), local_num_points, MPI_INT, 0, MPI_COMM_WORLD);

        // 更新质心
        if (rank == 0) {
            vector<int> counts(num_clusters, 0);
            vector<Point> sums(num_clusters);

            for (int i = 0; i < num_points; ++i) {
                int cluster = all_labels[i];
                sums[cluster].x += points[i].x;
                sums[cluster].y += points[i].y;
                counts[cluster]++;
            }

            for (int i = 0; i < num_clusters; ++i) {
                if (counts[i] > 0) {
                    centroids[i].x = sums[i].x / counts[i];
                    centroids[i].y = sums[i].y / counts[i];
                }
            }
        }

        // 广播质心
        MPI_Bcast(centroids.data(), 2 * num_clusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void kMeans_sse(vector<Point>& data, int k, int  max_iterations, vector<Point>& centroids) {
    int n = data.size();
    vector<int> labels(n);
    vector<int>count(k);
    // 迭代更新聚类中心
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        // 将每个点分配到最近的聚类中心
        for (int i = 0; i < n; i++) {
            double minDist = distance_sse(data[i], centroids[0]);
            int label = 0;
            for (int j = 1; j < k; j++) {
                double dist = distance_sse(data[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    label = j;
                }
            }
            if (labels[i] != label) {
                labels[i] = label;
            }
        }

        // 更新聚类中心的位置
        for (int i = 0; i < k; i++) {
            double sumX = 0, sumY = 0;
            int count = 0;
            for (int j = 0; j < n; j++) {
                if (labels[j] == i) {
                    sumX += data[j].x;
                    sumY += data[j].y;
                    count++;
                }
            }
            if (count > 0) {
                centroids[i].x = sumX / count;
                centroids[i].y = sumY / count;
            }
        }
    }
}
void kMeans_sse2(vector<Point>& data, int k, int  max_iterations, vector<Point>& centroids) {
    int n = data.size();
    vector<int> labels(n);
    vector<int>count(k);
    // 迭代更新聚类中心
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        // 将每个点分配到最近的聚类中心
        for (int i = 0; i < n; i += kVectorSize) {
            __m128d min_distance = _mm_set_pd1(distance_sse(data[i], centroids[0]));
            __m128i min_index = _mm_set1_epi32(0);
            for (int j = 1; j < k; j++) {
                __m128d d = _mm_set_pd1(distance_sse(data[i], centroids[j]));
                __m128d mask = _mm_cmplt_pd(d, min_distance);
                min_distance = _mm_blendv_pd(min_distance, d, mask);
                min_index = _mm_blendv_epi8(min_index, _mm_set1_epi32(j), _mm_castpd_si128(mask));
            }
            double distance[kVectorSize];
            _mm_store_pd(distance, min_distance);
            int index[kVectorSize];
            _mm_store_si128((__m128i*)index, min_index);
            for (int j = 0; j < kVectorSize; j++) {
                centroids[index[j]].x += data[i + j].x;
                centroids[index[j]].y += data[i + j].y;
                count[index[j]]++;
            }
        }

        // 更新聚类中心的位置
        for (int i = 0; i < k; i++) {
            if (count[i] > 0) {
                centroids[i].x /= count[i];
                centroids[i].y /= count[i];
                count[i] = 0;
            }
        }
    }
}

void kMeans_mp(vector<Point>& data, int k, int  max_iterations, vector<Point>& centroids) {
    int n = data.size();
    vector<int> labels(n);
    vector<int>count(k);
    // 迭代更新聚类中心
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        // 将每个点分配到最近的聚类中心
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
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

        // 更新聚类中心的位置
        for (int i = 0; i < k; i++) {
            double sumX = 0, sumY = 0;
            int count = 0;
            for (int j = 0; j < n; j++) {
                if (labels[j] == i) {
                    sumX += data[j].x;
                    sumY += data[j].y;
                    count++;
                }
            }
            if (count > 0) {
                centroids[i].x = sumX / count;
                centroids[i].y = sumY / count;
            }
        }
    }
}


int main(int argc, char** argv) {
    long long head, tail, freq;
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int num_clusters = 3;
    const int max_iters = 10;

    //导入数据
    srand(time(NULL));
    vector<Point> data;
    vector<vector<double>> dataArray; //因为需要获取物料的重量、频次、体积信息，所以预先创建一个嵌套数组
    ifstream inFile("C:/Users/Lenovo/Desktop/xclara.csv", ios::in);
    string linestr;
    //判断
    if (inFile.fail())
    {
        cout << "读取文件失败" << endl;
    }
    while (getline(inFile, linestr))//逐行读取文件，直到为空为止
    {
        stringstream ss(linestr);//存成二维表结构
        string str;//每行中的单个字符
        vector<double> lineArray;
        while (getline(ss, str, ','))
        {
            lineArray.push_back(atof(str.c_str()));//一行数据以vector保存
        }
        dataArray.push_back(lineArray);//每一行vector数据都放到strArray中去
    }
    for (int i = 0; i <16384; ++i) {
        Point p = { dataArray[i][0] , dataArray[i][1] };
        data.push_back(p);
    }
    vector<Point> centroids = initCentroids(data, num_clusters);
    /*
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    kMeans(data, num_clusters, max_iters, centroids);
    // 打印质心
    if (rank == 0) {
        cout << "最终的质心:\n";
        for (const auto& centroid : centroids) {
            cout << centroid.x << ", " << centroid.y << "\n";
        }
    }
    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    if (rank == 0)
    cout << "平凡算法：" << (tail - head) * 1000.0 / freq
        << "ms" << endl;
*/
   
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    kMeans_mpi(data, num_clusters, max_iters, centroids);
    // 打印质心
    if (rank == 0) {
        cout << "最终的质心:\n";
        for (const auto& centroid : centroids) {
            cout << centroid.x << ", " << centroid.y << "\n";
        }
    }
    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    if (rank == 0)
    cout << "mpi算法：" << (tail - head) * 1000.0 / freq
        << "ms" << endl;
    MPI_Finalize();

 //------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    /*
 // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    kMeans_sse(data, num_clusters, max_iters, centroids);
    // 打印质心
    if (rank == 0) {
        cout << "最终的质心:\n";
        for (const auto& centroid : centroids) {
            cout << centroid.x << ", " << centroid.y << "\n";
        }
    }
    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    if (rank == 0)
    cout << "SSE1算法：" << (tail - head) * 1000.0 / freq
        << "ms" << endl;

 //------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    kMeans_sse2(data, num_clusters, max_iters, centroids);
    // 打印质心
    if (rank == 0) {
        cout << "最终的质心:\n";
        for (const auto& centroid : centroids) {
            cout << centroid.x << ", " << centroid.y << "\n";
        }
    }
    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    if (rank == 0)
    cout << "SSE2算法：" << (tail - head) * 1000.0 / freq
        << "ms" << endl;
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
    return 0;
}
