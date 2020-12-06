#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <set>

using namespace std;

using Matrix = vector< vector<int> >;
using KernelMatrix = vector< vector<double> >;

// list of possible kernels.
set<string> KERNELS = {"jaccard", "genetic"};

/** Read and write functions */
// function to read a .csv file
Matrix read_csv(string fileName) {
    fstream dataFile;
    dataFile.open(fileName);
    Matrix data;
    string row;
    while (getline(dataFile, row)) {
        vector<int> dataRow;
        for (char c: row) {
            if (c != ',') {
                int v = (int)c - '0';
                dataRow.push_back(v);
            }
        }
        data.push_back(dataRow);
    }
    dataFile.close();
    return data;
}

/** writes the information from the kernelMatrix output to the output file (fileName) */
void writeOut(string& fileName, const KernelMatrix& output, int s) {
    fstream fileOut;
    fileOut.open(fileName);
    int k = 0;
    for (int i = 0; i < s; ++i) {
        for (int j = 0; j < s; ++j) {
            fileOut << output[i][j];
            if (j != s-1) {
                fileOut << ", ";
            }
            else ++k;
        }
        fileOut << endl;
    }
    fileOut.close();
    cerr << k << endl;
}

/**************************************************************************** //
                        Kernel matrix computation:
/ *************************************************************************** //
    Different kernels:
    - Jaccard similarity -> done
    - Linear
*/

// NOTE: this version of the kernel considers a "2" value as a 1.
double jaccard_sim(const vector<int>& a, const vector<int>& b) {
    int n = a.size();
    int numerator = 0;
    int denominator = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] > 0 and b[i] > 0) {
            ++denominator;
            ++numerator;
        }
        else if (a[i] + b[i] > 0) {
            denominator++;
        }
    }
    return (double)numerator / denominator;
}

KernelMatrix jaccard_matrix(const Matrix& data) {
    int n = data.size();
    KernelMatrix matrix(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double v = jaccard_sim(data[i], data[j]);
            matrix[i][j] = v;
            matrix[j][i] = v;
        }
    }
    return matrix;
}

double genetic_sim(const vector<int>& a, const vector<int>& b) {
    int n = a.size();
    int numerator = 0;
    int denominator = 0; // the sum of the elements
    for (int i = 0; i < n; ++i) {
        denominator = denominator + a[i] + b[i];
        if (a[i] > 0 and a[i] == b[i]) numerator += a[i] + b[i];
    }
    if (denominator == 0){
        cerr << "wtf 0 division!!" << endl;
    }
    return (double)numerator / denominator;
}
KernelMatrix genetic_kernel(const Matrix& data) {
    int n = data.size();
    KernelMatrix matrix(n, vector<double>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double v = genetic_sim(data[i], data[j]);
            matrix[i][j] = v;
            matrix[j][i] = v;
        }
    }
    return matrix;
}

KernelMatrix compute_kernel_matrix(const Matrix& data, const string& function) {
    if (function == "jaccard") return jaccard_matrix(data);
    else if (function == "genetic") return genetic_kernel(data);
    // else if (function == "linear") return linear_matrix(data);
    else cerr << "ups no good kernel" << endl;
}

/** Introduce parameters for the calculation */
int enter_params(string& input, string& output, string& kFunc) {
    cout << "Enter the input file: ";
    cin  >> input;
    cout << endl << "Enter the output file name: ";
    cin >> output;
    cout << endl << "Enter the kernel function: ";
    cin >> kFunc;
    cout << endl;
    if (KERNELS.find(kFunc) != KERNELS.end()) return 0;
    else return -1;
}

int main() {
    string outputFileName;
    string dataFileName;
    string kernelFunction;
    if (enter_params(dataFileName, outputFileName, kernelFunction) != 0) {
        cerr << "This kernel doesn't exist!" << endl;
        return 0;
    }
    cout << dataFileName << ", " << outputFileName << ", " << kernelFunction << endl;

    // read the file
    Matrix dataMatrix = read_csv(dataFileName);
    int N = dataMatrix.size();
    // compute kernel Matrix
    KernelMatrix outputMatrix = compute_kernel_matrix(dataMatrix, kernelFunction);

    // write kernel matrix
    writeOut(outputFileName, outputMatrix, N);
}
