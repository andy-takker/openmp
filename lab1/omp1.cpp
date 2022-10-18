#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <omp.h>

using namespace std;

double target(float x) {
	// подинтегральное выражение
	return log(sin(x));
}

double calc_int_with_omp(float a, float b, int n) {   
	// расчет интеграла методом средних прямоугольников через OpenMP
    double h = (b - a) / n;
    double s = 0;   
    #pragma omp parallel reduction(+:s) 
    {
        #pragma omp for schedule(static)
        for (int i = 1; i < n; i++){
			double x_i = a + i * h;
			double x_i_prev = x_i - h;
            s += target((x_i+x_i_prev)/2) * (x_i - x_i_prev);  
		}
    }
    return s;
}
double calc_int(float a, float b, int n) {
	// расчет интеграла методом средних прямоугольников обычным циклом
    double h = (b - a) / n;
	double s = 0;
    for (int i = 1; i < n; i++){
		double x_i = a + i * h;
		double x_i_prev = x_i - h;
        s += target((x_i+x_i_prev)/2) * (x_i - x_i_prev);  
	}
    return s;
}

void write_answer(string output_file, double result) {
	// Вывод результата в файл
	ofstream out(output_file);
    if (out.is_open()) {
        out << result;
    }
    else {
        cerr << "Could not open the file "+ output_file;
        exit(1);
    }
    out.close();

    cout << "Done!" << endl;
}

double runge(float a, float b, float err, bool omp) {
	// расчет значения с указанной точностью
	int n = 1;
    float err_i = 0;
    float result = 0, result_prev = 0;  
    if (omp) 
		result_prev = calc_int_with_omp(a, b, n);
    else
		result_prev = calc_int(a, b, n);
	// Итеративно подсчитываем интеграл и увеличиваем разбиение шагов пока
	// не достигнем нужной точности
    do {
        n *= 10;
        if (omp) 
			result = calc_int_with_omp(a, b, n);
        else
			result = calc_int(a, b, n);
        err_i = abs(result_prev - result);
        result_prev = result;
    } while (err_i > err);
    return result_prev;
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        cerr << "incorrect count of input args: " << argc-1 << ". Need: 3" << endl;
        exit(1);
    }

    string input_file = argv[1];
    string output_file = argv[2];
    int treads = atoi(argv[3]);
    bool omp = true;


    if (treads == -1)
        omp = false;
    else {
        if (treads == 0) {
            treads = omp_get_max_threads();
        }
        omp_set_num_threads(treads);
    }  

    double a = 0, b = 0, err = 0;     
    ifstream in(input_file);
       double result = 0;

    if (in.is_open()) {
        if ((in >> a) && (in >> b) && (in >> err)) {
            double begin, end;
            begin = omp_get_wtime();
            result = runge(a, b, err, omp);
            end = omp_get_wtime();
            printf("Time (%i thread(s)): %g ms\n", treads, (end - begin)/1000);
			write_answer(output_file, result);
        }
        else {
            cerr << "Incorrect number of arguments in the file";
            exit(1);
        }
    }
    else {
        cerr << "Could not open the file " + input_file;
        exit(1);
    }
    in.close();
    exit(0);
}
