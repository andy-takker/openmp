#define _USE_MATH_DEFINES

#include <string>
#include <fstream>
#include <omp.h>
#include <cmath>

using namespace std;

void error(const char * msg){
    cerr << msg << endl;
    exit(1);
}

struct ImageData {
    string type_num;
    int width, height, max_val;
    unsigned char *pixel_data;
    int data_size;

    ImageData(string type_num,
              int width, int height, int max_val,
              unsigned char *pixel_data, int data_size
    ) : type_num(type_num), width(width), height(height), max_val(max_val), pixel_data(pixel_data), data_size(data_size) {}

    ~ImageData() {
        delete[] pixel_data;
    }
};

class HistWriteRead {
public:
    static ImageData *read_file(const string &img_file_path) {
        ifstream img_file;
        img_file.open(img_file_path, ifstream::in | ifstream::binary);
        if (!img_file.is_open()) {
            cerr << "Could not open the image: file: " + img_file_path << endl;
            exit(1);
        }

        string type_num;
        img_file >> type_num;
        if (type_num != "P5") {
            cerr << "Unsupported image type: " + type_num << endl;
            exit(1);
        }

        int width, height, max_val;
        img_file >> width >> height >> max_val;
        img_file.ignore(1);

        int data_size = width * height;
        auto data = new unsigned char[data_size];
        img_file.read(reinterpret_cast<char *>(data), data_size);
        img_file.close();

        return new ImageData(type_num, width, height, max_val, data, data_size);
    }
    static void write_file(const string &image_file_path, const ImageData &image_data) {
        ofstream out_file;
        out_file.open(image_file_path, ofstream::out | ofstream::binary);
        if (!out_file.is_open()){
            throw runtime_error("Could not open the image file: "+ image_file_path);
        }
        out_file << image_data.type_num << '\n'
            << image_data.width << ' ' << image_data.height << '\n'
            << image_data.max_val << '\n';
        out_file.write(reinterpret_cast<char *>(image_data.pixel_data), image_data.data_size);
        out_file.close();
    }
};

class CalcHist {
public:
    static CalcHist *with_omp(unsigned int num_of_threads = 0) {
        return new CalcHist(num_of_threads);
    }

    static CalcHist *without_omp() {
        return new CalcHist();
    }

    void blurImage(const ImageData &img_data, const string &output_file_path, int num_of_boxes, float sigma) const;

private:
    CalcHist(unsigned int num_of_threads) : num_of_threads(num_of_threads), is_omp(true) {};

    CalcHist() : is_omp(false) {};

    bool is_omp;
    unsigned int num_of_threads;
};

void CalcHist::blurImage(const ImageData &image_data, const string &output_file_path, const int kernel_width, const float sigma) const {   
    int radius = (kernel_width - 1) / 2;
    cout << "Kernel width (number_of_boxes): " << kernel_width << endl;
    cout << "Radius: " << radius << endl;
    
    double kernel[kernel_width][kernel_width] = {0};
    double sum = 0;
    for (int x = -radius; x <= radius; x++){
        for (int y = -radius; y <= radius; y++){
            double exp_numerator = (double) -(x*x+y*y);
            double exp_denominator = 2 * sigma * sigma;
            double e_exp = pow(M_E, exp_numerator / exp_denominator);
            double kernel_value = e_exp / (2 * M_PI * sigma * sigma);
            kernel[x+radius][y+radius] = kernel_value;
            sum += kernel_value;
        }
    }
    for (int x = 0; x < kernel_width; x++){
        for (int y = 0; y < kernel_width; y++)
            kernel[x][y] /= sum;
    }    
    auto new_data = new unsigned char[image_data.data_size];
    if(!is_omp){
        for (int y = radius; y < image_data.height - radius; y++){
            for (int x = radius; x < image_data.width - radius; x++){
                int index = image_data.width * y + x;
                double val = 0;
                for (int kernel_y = -radius; kernel_y <= radius; kernel_y++){
                    for (int kernel_x = -radius; kernel_x <= radius; kernel_x++){
                        double kernel_value = kernel[kernel_x+radius][kernel_y+radius];
                        int kernel_index = image_data.width * (y-kernel_y) + x-kernel_x;
                        val += image_data.pixel_data[kernel_index] * kernel_value;
                    }
                }
                new_data[index] = (char) val;
            }   
        }
    } else {
        #pragma omp parallel shared(image_data, new_data)
        {
            #pragma omp for schedule(static)
            for (int y = radius; y < image_data.height - radius; y++){
                for (int x = radius; x < image_data.width - radius; x++){
                    int index = image_data.width * y + x;
                    double val = 0;
                    for (int kernel_y = -radius; kernel_y <= radius; kernel_y++){
                        for (int kernel_x = -radius; kernel_x <= radius; kernel_x++){
                            double kernel_value = kernel[kernel_x+radius][kernel_y+radius];
                            int kernel_index = image_data.width * (y-kernel_y) + x-kernel_x;
                            val += image_data.pixel_data[kernel_index] * kernel_value;
                        }
                    }
                    new_data[index] = (char) val;
                }   
            }
        }
    }
    

    auto new_image_data = new ImageData(
        image_data.type_num, 
        image_data.width, 
        image_data.height, 
        image_data.max_val, 
        new_data, 
        image_data.data_size
    );
    HistWriteRead::write_file(output_file_path, *new_image_data);
}
