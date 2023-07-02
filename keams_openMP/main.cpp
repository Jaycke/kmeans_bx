#include<iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <fstream>  
#include <sstream>
#include <windows.h>

using namespace std;

const int kVectorSize = 4;
const int kAvxVecLen = 8;

// 定义向量类型
typedef __m256i AvxVec;


// 定义结构体来表示数据点
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

//定义函数来随机初始化质心
vector<Point> initCentroids(vector<Point>& data, int k) {
    vector<Point> centroids;
    int n = data.size();
    for (int i = 0; i < k; i++) {
        int idx = rand() % n;
        centroids.push_back(data[idx]);
    }
    return centroids;
}

vector<Point> initCentroids2(vector<Point>& data, int k) {
    vector<Point> centroids;
    int n = data.size();
    #pragma omp parallel for
    for (int i = 0; i < k; i++) {
        int idx = rand() % n;
        #pragma omp critical
        centroids.push_back(data[idx]);
    }
    return centroids;
}


// 定义k均值聚类函数
vector<int> kMeans(vector<Point>& data, int k) {
    int times = 0;
    int n = data.size();
    vector<int> labels(n);
    vector<Point> centroids = initCentroids(data, k);
    bool changed = true;
    while (changed) {
        changed = false;
        // 将每个数据点指定给最近的质心
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
                changed = true;
            }
        }
        // 更新质心
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
        times++;
    }
    cout << times << endl;
    return labels;
}

vector<int> kMeans2(vector<Point>& data, int k, int  max_iterations) {
    int n = data.size();
    vector<int> labels(n);
    vector<Point> centroids = initCentroids(data, k);
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

    return labels;
}

void kMeans3(vector<Point>& data, int k, int  max_iterations, vector<Point>& centroids) {
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

vector<int> kMeans4(vector<Point>& data, int k, int  max_iterations) {
    int n = data.size();
    vector<int> labels(n);
    vector<Point> centroids = initCentroids2(data, k);
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
        #pragma omp parallel for
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
                #pragma omp critical
                centroids[i].x = sumX / count;
                centroids[i].y = sumY / count;
            }
        }
    }

    return labels;
}
vector<int> kMeans_sse(vector<Point>& data, int k) {
    int times = 0;
    int n = data.size();
    vector<int> labels(n);
    vector<Point> centroids = initCentroids(data, k);
    bool changed = true;
    while (changed) {
        changed = false;

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
                changed = true;
            }
        }

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
        times++;
    }
    //cout << times << endl;
    return labels;
}

vector<int> kMeans_sse2(vector<Point>& data, int k, int  max_iterations) {
    int n = data.size();
    vector<int> labels(n);
    vector<Point> centroids = initCentroids(data, k);
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

    return labels;
}

vector<int> kMeans_sse3(vector<Point>& data, int k, int  max_iterations) {
    int n = data.size();
    vector<int> labels(n);
    vector<Point> centroids = initCentroids(data, k);
    vector<int>count(k);
    // 迭代更新聚类中心
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        // 将每个点分配到最近的聚类中心
        for (int i = 0; i < n; i += kVectorSize) {
            __m128d min_distance = _mm_set_pd1(distance(data[i], centroids[0]));
            __m128i min_index = _mm_set1_epi32(0);
            for (int j = 1; j < k; j++) {
                __m128d d = _mm_set_pd1(distance(data[i], centroids[j]));
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
    return labels;
}

vector<int> kMeans_avx(vector<Point>& data, int k, int  max_iterations) {
    //cout << "start" << endl;
    int n = data.size();
    vector<int> labels(n);
    vector<Point> centroids = initCentroids(data, k);
    vector<int>count(k);
    // 迭代更新聚类中心
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        // 将每个点分配到最近的聚类中心
        for (int i = 0; i < n; i += kAvxVecLen) {
            __m256d min_distance = _mm256_set1_pd(distance_sse(data[i], centroids[0]));
            __m256i min_index = _mm256_set1_epi32(0);
            for (int j = 1; j < k; j++) {
                __m256d d = _mm256_set1_pd(distance_sse(data[i], centroids[j]));
                __m256d mask = _mm256_cmp_pd(d, min_distance, _CMP_LT_OS);
                min_distance = _mm256_blendv_pd(min_distance, d, mask);
                min_index = _mm256_blendv_epi8(min_index, _mm256_set1_epi32(j), _mm256_castpd_si256(mask));
            }
            double distance[kVectorSize];
            _mm256_store_pd(distance, min_distance);
            int index[kVectorSize];
            _mm256_store_si256((__m256i*)index, min_index);
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
    return labels;
}

int main() {
    long long head, tail, freq;
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
    for (int i = 0; i < 16384; ++i) {
        Point p = { dataArray[i][0] , dataArray[i][1] };
        data.push_back(p);
    }
    vector<Point> centroids = initCentroids(data, num_clusters);
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    kMeans3(data, num_clusters, max_iters, centroids);
    // 打印质心
        cout << "最终的质心:\n";
        for (const auto& centroid : centroids) {
            cout << centroid.x << ", " << centroid.y << "\n";
        }
    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "openMP算法：" << (tail - head) * 1000.0 / freq
        << "ms" << endl;

    /*
    vector<Point> d1 = data;
    auto begin = std::chrono::high_resolution_clock::now();
    vector<int> labels = kMeans(d1, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("串行算法: %.3f seconds.\n", elapsed.count() * 1e-9);
    */
    /*
    vector<Point> d2 = data;
    auto begin2 = std::chrono::high_resolution_clock::now();
    vector<int> labels2 = kMeans2(d2, k, 100);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto elapsed2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);
    printf("串行2算法: %.3f seconds.\n", elapsed2.count() * 1e-9);

    vector<Point> d3 = data;
    auto begin3 = std::chrono::high_resolution_clock::now();
    vector<int> labels3 = kMeans3(d3, k, 100);
    auto end3 = std::chrono::high_resolution_clock::now();
    auto elapsed3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin3);
    printf("openMP1算法: %.3f seconds.\n", elapsed3.count() * 1e-9);

    vector<Point> d5 = data;
    auto begin5 = std::chrono::high_resolution_clock::now();
    vector<int> labels5 = kMeans4(d5, k, 100);
    auto end5 = std::chrono::high_resolution_clock::now();
    auto elapsed5 = std::chrono::duration_cast<std::chrono::nanoseconds>(end5 - begin5);
    printf("openMP2算法: %.3f seconds.\n", elapsed5.count() * 1e-9);

    vector<Point> d6 = data;
    auto begin6 = std::chrono::high_resolution_clock::now();
    vector<int> labels6 = kMeans_sse(data, k);
    auto end6= std::chrono::high_resolution_clock::now();
    auto elapsed6 = std::chrono::duration_cast<std::chrono::nanoseconds>(end6 - begin6);
    printf("SEE1: %.3f seconds.\n", elapsed6.count() * 1e-9);


    vector<Point> d7 = data;
    auto begin7 = std::chrono::high_resolution_clock::now();
    vector<int> labels7 = kMeans_sse2(data, k, 100);
    auto end7 = std::chrono::high_resolution_clock::now();
    auto elapsed7= std::chrono::duration_cast<std::chrono::nanoseconds>(end7 - begin7);
    printf("SEE2: %.3f seconds.\n", elapsed7.count() * 1e-9);

    vector<Point> d8 = data;
    auto begin8 = std::chrono::high_resolution_clock::now();
    vector<int> labels8 = kMeans_sse3(data, k, 100);
    auto end8 = std::chrono::high_resolution_clock::now();
    auto elapsed8 = std::chrono::duration_cast<std::chrono::nanoseconds>(end8 - begin8);
    printf("SEE3: %.3f seconds.\n", elapsed8.count() * 1e-9);
    */
    return 0;
}