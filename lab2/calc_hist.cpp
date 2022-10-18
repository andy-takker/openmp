#include <string>
#include <fstream>
#include <omp.h>

using namespace std;

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
    static void write_file(const string &image_file_path, int * histogram) {
        ofstream out_file;
        out_file.open(image_file_path, ofstream::out | ofstream::binary);
        if (!out_file.is_open()) {
            cerr << "Could not open the image file: " + image_file_path << endl;
            exit(1);
        }
        for(int i = 0; i < 256; i++){
            out_file << histogram[i];
        }
        out_file << endl;
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

    void correctImage(const ImageData &img_data, const string &output_file_path) const;

private:
    CalcHist(unsigned int num_of_threads) : num_of_threads(num_of_threads), is_omp(true) {};

    CalcHist() : is_omp(false) {};

    bool is_omp;
    unsigned int num_of_threads;
};

void CalcHist::correctImage(const ImageData &img_data, const string &output_file_path) const {    
    int histogram[256] = {0};
    cout << img_data.data_size << endl;
    if(is_omp){
        #pragma omp parallel shared(img_data)
        {
            int histogram_private[256] = {0};
            #pragma omp for schedule(static)
            for (int i = 0; i < img_data.data_size; i++) {
                auto pixel = img_data.pixel_data[i];
                histogram_private[pixel]++;
            }
            #pragma omp critical
            {
                for(int i = 0; i < 256; i++){
                    histogram[i] += histogram_private[i];
                }
            }
        }
    } else {
        for (int i = 0; i < img_data.data_size; i++) {
                auto pixel = img_data.pixel_data[i];
                histogram[pixel]++;
            }
    }
    
    HistWriteRead::write_file(output_file_path, histogram);
}
