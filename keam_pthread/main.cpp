#include<iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include<pthread.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

using namespace std;

int numThreads = 4;

// 定义结构体来表示数据点
struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
};

// 储存线程参数
struct thread_data_t {
    vector<Point>* points;
    vector<Point>* centroids;
    vector<int>* labels;
    int start;
    int end;
};

// 定义函数来计算两点之间的欧几里得距离
double distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
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

void* updateCentroidsThread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    vector<Point>& points = *(data->points);
    vector<int>& labels = *(data->labels);
    vector<Point>& centroids = *(data->centroids);
    int start = data->start;
    int end = data->end;
    for (int i = start; i < end; i++) {
        double minDist = distance(points[i], centroids[0]);
        int label = 0;
        for (int j = 1; j < centroids.size(); j++) {
            double dist = distance(points[i], centroids[j]);
            if (dist < minDist) {
                minDist = dist;
                label = j;
            }
        }
        if (labels[i] != label) {
            labels[i] = label;
        }
    }
    pthread_exit(NULL);
}

vector<int> kMeans3(vector<Point>& data, int k, int  max_iterations) {
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

        int stride = n / numThreads;
        pthread_t threads[stride];
        thread_data_t data_t[numThreads];
        for (int i = 0; i < numThreads; i++) {
            int start = i * stride;
            int end = (i == numThreads - 1) ? centroids.size() : (i + 1) * stride;
            data_t[i] = { &data,&centroids,&labels,start,end };
            pthread_create(&threads[i], NULL, updateCentroidsThread, (void*)&data_t[i]);
        }
        // 等待所有线程执行完毕
        for (int i = 0; i < numThreads; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    return labels;

}
/*
// 将数据点分配到最近的聚类中心
void assignPoints(vector<Point>& points, vector<Centroid>& centroids) {
    for (auto& point : points) {
        double minDist = INFINITY;
        Centroid* nearestCentroid = nullptr;
        // 找到距离该数据点最近的聚类中心
        for (auto& centroid : centroids) {
            double dist = distance(point, centroid.pos);
            if (dist < minDist) {
                minDist = dist;
                nearestCentroid = &centroid;
            }
        }
        // 将该数据点添加到对应的聚类中心中
        pthread_mutex_lock(&nearestCentroid->mutex);
        nearestCentroid->points.push_back(point);
        pthread_mutex_unlock(&nearestCentroid->mutex);
    }
}
*/

int main() {
    srand(time(NULL));
    vector<Point> data;
    for (int i = 0; i < 100000; i++) {
        double x = rand() % 1000;
        double y = rand() % 1000;
        data.push_back(Point(x, y));
    }
    int k = 10;

    vector<Point> d1 = data;
    auto begin1 = std::chrono::high_resolution_clock::now();
    vector<int> labels = kMeans(d1, k);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto elapsed1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin1);
    printf("串行算法: %.3f seconds.\n", elapsed1.count() * 1e-9);

    vector<Point> d2 = data;
    auto begin2 = std::chrono::high_resolution_clock::now();
    vector<int> labels2 = kMeans2(d2, k, 60);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto elapsed2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);
    printf("串行2算法: %.3f seconds.\n", elapsed2.count() * 1e-9);


    vector<Point> d3 = data;
    auto begin3 = std::chrono::high_resolution_clock::now();
    vector<int> labels3 = kMeans3(d3, k, 60);
    auto end3 = std::chrono::high_resolution_clock::now();
    auto elapsed3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin3);
    printf("pthread算法: %.3f seconds.\n", elapsed3.count() * 1e-9);
    return 0;
}
