#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <functional>

#include "./calc_hist.cpp"

using namespace std;


double measure_time_millis(const function<void()> &fun) {
    auto start = chrono::high_resolution_clock::now();
    fun();
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

double measureAvgTimeMillis(const function<void()> &fun, int times_to_run = 10) {
    double avg_time = 0;
    for (int i = 0; i < times_to_run; i++) {
        auto exc_time = measure_time_millis(fun);
        avg_time += exc_time;
    }
    return avg_time / times_to_run;
}

void run(const string &in_file_path, const string &out_file_path, int num_of_threads) {
    bool is_omp = num_of_threads != -1;
    CalcHist *executor;
    if (is_omp) {
        if (num_of_threads < 0) {
            cerr << "incorrect number of threads  must be positive!" << endl;
            exit(1);
        }
         if (num_of_threads == 0) {
            num_of_threads = omp_get_max_threads();
        }
        omp_set_num_threads(num_of_threads);
        executor = CalcHist::with_omp(num_of_threads);
    } else {
        executor = CalcHist::without_omp();
    }

    auto img_data = HistWriteRead::read_file(in_file_path);
    auto time = measure_time_millis([&] { executor->correctImage(*img_data, out_file_path); });

    printf("Time (%d thread(s)): %lfms\n", num_of_threads, time);

    delete img_data;
    delete executor;
}

int main(int argc, char *argv[]) {
    try {
        if (argc != 4) {
            cerr << "incorrect count of input args: " << argc-1 << ". Need: 3" << endl;
            exit(1);
        }
        auto in_file_path = string(argv[1]);
        auto out_file_path = string(argv[2]);
        int num_of_threads = atoi(argv[3]);
        run(in_file_path, out_file_path, num_of_threads);
    } catch (const exception &e) {
        fprintf(stderr, "The program was aborted: %s", e.what());
        return 1;
    }
    return 0;
}
