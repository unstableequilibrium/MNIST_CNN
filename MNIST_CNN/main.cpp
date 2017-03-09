#include "opencv2/opencv.hpp"
#include <fstream>
#include <random>
using namespace cv;
using namespace std;

enum {HEAVISIDE = 0, RELU, LOGISTIC, TANGH};

void load_MNIST_data(string file_imgs, string file_labels, vector<Mat> &imgs, vector<uchar> &labels, int data_size)
{
	std::ifstream in_imgs(file_imgs, ios::binary);
	std::ifstream in_lbs(file_labels, ios::binary);

	int sz_labels = data_size;
	in_lbs.seekg(8);
	labels.resize(sz_labels);
	imgs.resize(sz_labels);

	in_imgs.seekg(16);	
	int rows = 28, cols = 28;
	int num_bytes = rows * cols;
	 
	for (size_t i = 0; i < sz_labels; i++){
		in_lbs.read((char *) &labels[i], 1);
		imgs[i].create(rows, cols, CV_8U);
		in_imgs.read((char *) imgs[i].data, num_bytes);
	}
	in_lbs.close();
	in_imgs.close();
}

void load_MNIST_data(string file_imgs, string file_labels, vector<vector<double>> &imgs, vector<uchar> &labels, int data_size)
{
	std::ifstream in_imgs(file_imgs, ios::binary);
	std::ifstream in_lbs(file_labels, ios::binary);

	in_lbs.seekg(8);
	labels.resize(data_size);
	imgs.resize(data_size);

	in_imgs.seekg(16);
	int rows = 28, cols = 28;
	int num_bytes = rows * cols;
	uchar *buf = new uchar[num_bytes];
	for (size_t i = 0; i < data_size; i++){
		in_lbs.read((char *) &labels[i], 1);
		in_imgs.read((char *) buf, num_bytes);
		imgs[i].resize(num_bytes);
		for (size_t j = 0; j < num_bytes; j++){
			imgs[i][j] = double(buf[j]) / 255.0;
		}
	}
	delete buf;
	in_lbs.close();
	in_imgs.close();
}


void full_connected_layer(vector<double> input, vector<double> &output, int N, int M, vector<vector<double>> W)
{
	for (size_t i = 0; i < N; i++){
		output[i] = 0.0;
		for (size_t j = 0; j < M; j++){
			output[i] += W[i][j] * input[j];
		}
	}
}

void activation(vector<double> input, vector<double> &output, int N, double(*func)(double))
{
	for (size_t i = 0; i < N; i++){
		output[i] = func(input[i]);
	}
}

double heaviside(double z)
{
	if (z > 0)return 1.0;
	else return 0.0;
}

double relu(double z) 
{
	if (z > 0)return z;
	else return 0.0;
}

double logistic(double z)
{
	return 1. / (1. + exp(-z));
}

double logistic_grad(double z)
{
	double t = logistic(z);
	return t * (1.0 - t);
}

double relu_grad(double z)
{
	return heaviside(z);
}

double tanh_grad(double z)
{
	double t = tanh(z);
	return 1.0 - t * t;
}

// M inputs -> N outputs
void back_prop_out_layer(int N, int M, vector<double> x, vector<double> z, vector<double> e, double (*grad)(double), 
	vector<double> &delta, vector<vector<double>> &dw, double learning_rate = 0.01)
{
	for (size_t i = 0; i < N; i++){
		delta[i] = e[i] * grad(z[i]);
		for (size_t j = 0; j < M; j++){
			dw[i][j] = learning_rate * delta[i] * x[j];
		}
	}
}

// M inputs -> N outputs
void back_prop_hidden_layer(int N, int M, int K, vector<double> x, vector<double> z, vector<double> delta, vector<vector<double>> W, 
	double(*grad)(double), vector<double> &delta_new, vector<vector<double>> &dw, double learning_rate = 0.01)
{
	for (size_t i = 0; i < N; i++){
		delta_new[i] = 0;
		for (size_t k = 0; k < K; k++){
			delta_new[i] += W[k][i] * delta[k];
		}
		delta_new[i] *= grad(z[i]);
		for (size_t j = 0; j < M; j++){
			dw[i][j] = learning_rate * delta_new[i] * x[j];
		}
	}
}

void update_weights(int N, int M, vector<vector<double>> &W, vector<vector<double>> dw, double scale, double lambda)
{
	double _lam = (1.0 - lambda);
	for (size_t i = 0; i < N; i++){
		for (size_t j = 0; j < M; j++){
			W[i][j] = _lam * W[i][j] - scale * dw[i][j];
		}
	}
}

// init weights
void init_weights(int N, int M, vector<vector<double>> &W)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 0.1);
	for (size_t i = 0; i < N; i++){
		for (size_t j = 0; j < M; j++){
			W[i][j]  = distribution(generator);
		}
	}
}

// init sequence of samples
void reinit_seq(int N, vector<int> &a)
{
	for (size_t i = 0; i < N; i++){
		int idx = rand() % (N - i);
		int tmp = a[idx];
		a[idx] = a[N - 1 - i];
		a[N - 1 - i] = tmp;
	}
}

void init_seq(int N, vector<int> &a)
{
	for (size_t i = 0; i < N; i++){
		a[i] = i;
	}
}

void calc_error(int N, vector<double> desire, vector<double> out, vector<double> &err)
{
	for (size_t i = 0; i < N; i++){
		err[i] = out[i] - desire[i];
	}
}

int classification(int N, vector<double> out)
{
	int arg_max = 0;
	double max = out[0];
	for (size_t i = 1; i < N; i++){
		if (out[i] > max){
			arg_max = i;
			max = out[i];
		}
	}
	return arg_max;
}

void soft_max(int N, vector<double> in, vector<double> &out)
{
	double sum = 0.0;
	for (size_t i = 0; i < N; i++){
		out[i] = exp(in[i]);
		sum += out[i];
	}

	for (size_t i = 0; i < N; i++){
		out[i] /= sum;
	}
}

void add(int N, int M, vector<vector<double>> a, vector<vector<double>> &b)
{
	for (size_t i = 0; i < N; i++)
		for (size_t j = 0; j < M; j++)
			b[i][j] += a[i][j];
}


void convolution(int rows, int cols, int ker_sz, vector<double> input, vector<double> &output, vector<double> w_ker, int stride = 1, int padding = 0)
{
	int _cols = (cols + 2 * padding - ker_sz + 1);
	_cols = _cols / stride + ((_cols % stride) != 0);
	int idx = 0, i_max = rows + padding - ker_sz + 1, 
		j_max = cols + padding - ker_sz + 1;
	for (int i = -padding; i < i_max; i += stride){
		int _i = ((i + rows) % rows) * cols;
		for (int j = -padding; j < j_max; j += stride, idx++){
			output[idx] = 0;
			int _id = _i + (j + cols) % cols;
			int _id_ker = 0;
			for (int ky = 0; ky < ker_sz; ky++){
				int _idy = ((i + rows + ky) % rows) * cols;
				for (int kx = 0; kx < ker_sz; kx++, _id_ker++){
					int _idx = (j + cols + kx) % cols;
					output[idx] += input[_idy + _idx] * w_ker[_id_ker];
				}
			}
		}
	}
}

void max_pooling(int rows, int cols, double *input, double *output, int size = 2, int stride = 2, int padding = 0)
{
	int _cols = (cols + 2 * padding - size + 1);
	_cols = _cols / stride + ((_cols % stride) != 0);
	int idx = 0, i_max = rows + padding + size - 1,
				j_max = cols + padding + size - 1;
	for (int i = -padding; i < i_max; i += stride){
		int _i = ((i + rows) % rows) * cols;
		for (int j = -padding; j < j_max; j += stride, idx++){
			output[idx] = 0;
			int _id = _i + (j + cols) % cols;
			int _id_ker = 0;
			for (int ky = 0; ky < size; ky++){
				int _idy = ((i + rows + ky) % rows) * cols;
				for (int kx = 0; kx < size; kx++){
					int _idx = (j + cols + kx) % cols;
					if(output[idx] < input[_idy + _idx]) output[idx] = input[_idy + _idx];
				}
			}
		}
	}
}

void back_prop_from_full_to_conv()
{

}

int main(int, char**)
{
	// Load training data
	vector<Mat> mat_train_digits;
	vector<vector<double>> train_digits;
	vector<uchar> train_labels;
	string path("C:\\Users\\Boris\\Google Диск\\NeuralNets\\MNIST\\" );
	cout << "Start loading MNIST training data..." << endl;
	load_MNIST_data(path + string("train-images.idx3-ubyte"), path + string("train-labels.idx1-ubyte"),
					train_digits, train_labels, 60000);
	cout << "End loading MNIST training data." << endl;


	// Load test data
	vector<vector<double>> test_digits;
	vector<uchar> test_labels;
	cout << "Start loading MNIST test data..." << endl;
	load_MNIST_data(path + string("t10k-images.idx3-ubyte"), path + string("t10k-labels.idx1-ubyte"),
		test_digits, test_labels, 10000);
	cout << "End loading MNIST test data." << endl;


	bool isCNN = true;
	// input vector 
	int L0 = 28 * 28;
	int sz_data = 60000;

	double init_learn_rate = 0.01;
	double lambda = 0.01;
	int sz_batch = 1;
	vector<int> seq(sz_data);
	init_seq(sz_data, seq);

	double(*act_fun)(double) = logistic;
	double(*grad_act)(double) = logistic_grad;


	if (isCNN){
		// Convolutional NN
		// 1-st hidden layer - 6 convolution  filters with 5x5 kernels
		int L1 = 6;// number of feature maps
		int Lc = 24 * 24;
		int _sz = 5*5 + 1;
		vector<vector<double>> w1(L1, vector<double>(_sz, 0.0));
		vector<vector<double>> z1(L1, vector<double>(Lc, 0.0));
		// vector<double> delta1(L1, 0);
		vector<vector<double>> v1(L1, vector<double>(Lc, 0));
		vector<vector<double>> dw1(L1, vector<double>(_sz, 0));
		init_weights(L1, _sz, w1);

		// 2-nd maxpooling layer
		int L2 = L1;
		int Lm = 12 * 12;
		vector<double> v2(L2*Lm, 0);
		
		// 3-rd layer
		int L3 = 100;
		vector<double> v3(L3 + 1, 0);
		vector<double> z3(L3, 0);
		vector<double> delta3(L3, 0);
		vector<vector<double>> W3(L3, vector<double>(L2 * Lm + 1));
		vector<vector<double>> dw3(L3, vector<double>(L2 * Lm + 1));
		init_weights(L3, L2 * Lm + 1, W3);

		// out layer
		int L = 10;
		vector<double>vL(L, 0);
		vector<double>zL(L, 0);
		vector<double>eL(L, 0);
		vector<double> deltaL(L, 0);
		vector<vector<double>> WL(L, vector<double>(L3 + 1));
		vector<vector<double>> dwL(L, vector<double>(L3 + 1));
		init_weights(L, L3 + 1, WL);

		// epochs
		for (int t = 0; t < 30; t++){
			cout << "epoch # " << t << endl;
			// forward and backprop iterations
			double learn_rate = init_learn_rate * exp(-t / 5.0);
			reinit_seq(sz_data, seq);
			for (size_t i = 0; i < sz_data / sz_batch; i++){
				vector<vector<double>> _dw1(L1, vector<double>(_sz, 0));
				vector<vector<double>> _dw3(L3, vector<double>(L2 * Lm + 1));
				vector<vector<double>> _dwL(L, vector<double>(L3 + 1));

				for (size_t j = 0; j < sz_batch; j++){
					int idx = seq[i * sz_batch + j];
					vector<double> input = train_digits[idx];
					input.push_back(1.0);
					vector<double> out(10, 0);
					out[train_labels[idx]] = 1.0;

					// Forward step
					// 1-2 layers
					for (size_t k = 0; k < L1; k++){
						convolution(28, 28, 5, input, z1[k], w1[k]);
						activation(z1[k], v1[k], L1, act_fun);
						max_pooling(24, 24, (double *) (v1.data() + k*Lc), (double *) (v2.data() + k*Lm));
					}
					
					// 3-rd layer
					full_connected_layer(v2, z3, L3, L2 + 1, W3);
					activation(z3, v3, L3, act_fun);

					// 4-th layer
					full_connected_layer(v3, zL, L, L3 + 1, WL);
					activation(zL, vL, L, act_fun);


					// Error
					calc_error(L, out, vL, eL);

					// Back-propagation step
					back_prop_out_layer(L, L2 + 1, v2, zL, eL, grad_act, deltaL, dwL, learn_rate);
					back_prop_hidden_layer(L2, L1 + 1, L, v1, z2, deltaL, WL, grad_act, delta2, dw2, learn_rate);
					back_prop_hidden_layer(L1, L0 + 1, L2, input, z1, delta2, W2, grad_act, delta1, dw1, learn_rate);

					// update dw's
					add(L, L2 + 1, dwL, _dwL);
					add(L2, L1 + 1, dw2, _dw2);
					add(L1, L0 + 1, dw1, _dw1);
				}

				// Update weights
				update_weights(L, L2 + 1, WL, _dwL, 1.0 / double(sz_batch), lambda * learn_rate);
				update_weights(L2, L1 + 1, W2, _dw2, 1.0 / double(sz_batch), lambda * learn_rate);
				update_weights(L1, L0 + 1, W1, _dw1, 1.0 / double(sz_batch), lambda * learn_rate);
			}

			// test 
			int correct_counter = 0;
			for (size_t i = (t % 10) * 1000; i < (t % 10 + 1) * 1000; i++){
				// Forward step
				full_connected_layer(test_digits[i], z1, L1, L0 + 1, W1);
				activation(z1, v1, L1, act_fun);
				full_connected_layer(v1, z2, L2, L1 + 1, W2);
				activation(z2, v2, L2, act_fun);
				full_connected_layer(v2, zL, L, L2 + 1, WL);
				activation(zL, vL, L, act_fun);
				if (classification(L, vL) == test_labels[i])correct_counter++;
			}
			cout << "accuracy = " << double(correct_counter) / 1000 << endl;
		}
	}
	else{
		// Fully-connected NN
		// add 1 hidden layer
		int L1 = 250;
		vector<double> v1(L1 + 1, 0);
		vector<double> z1(L1, 0);
		vector<double> delta1(L1, 0);
		vector<vector<double>> W1(L1, vector<double>(L0 + 1));
		vector<vector<double>> dw1(L1, vector<double>(L0 + 1));
		init_weights(L1, L0 + 1, W1);

		int L2 = 80;
		vector<double> v2(L2 + 1, 0);
		vector<double> z2(L2, 0);
		vector<double> delta2(L2, 0);
		vector<vector<double>> W2(L2, vector<double>(L1 + 1));
		vector<vector<double>> dw2(L2, vector<double>(L1 + 1));
		init_weights(L2, L1 + 1, W2);

		// out layer
		int L = 10;
		vector<double>vL(L, 0);
		vector<double>zL(L, 0);
		vector<double>eL(L, 0);
		vector<double> deltaL(L, 0);
		vector<vector<double>> WL(L, vector<double>(L2 + 1));
		vector<vector<double>> dwL(L, vector<double>(L2 + 1));
		init_weights(L, L2 + 1, WL);

		for (int t = 0; t < 30; t++){
			cout << "epoch # " << t << endl;
			// forward and backprop iterations
			double learn_rate = init_learn_rate * exp(-t / 5.0);
			reinit_seq(sz_data, seq);
			for (size_t i = 0; i < sz_data / sz_batch; i++){
				vector<vector<double>> _dw1(L1, vector<double>(L0 + 1, 0));
				vector<vector<double>> _dw2(L2, vector<double>(L1 + 1, 0.0));
				vector<vector<double>> _dwL(L, vector<double>(L2 + 1, 0.0));

				for (size_t j = 0; j < sz_batch; j++){
					int idx = seq[i * sz_batch + j];
					vector<double> input = train_digits[idx];
					input.push_back(1.0);
					vector<double> out(10, 0);
					out[train_labels[idx]] = 1.0;

					// Forward step
					full_connected_layer(input, z1, L1, L0 + 1, W1);
					activation(z1, v1, L1, act_fun);
					full_connected_layer(v1, z2, L2, L1 + 1, W2);
					activation(z2, v2, L2, act_fun);
					full_connected_layer(v2, zL, L, L2 + 1, WL);
					activation(zL, vL, L, act_fun);

					// Error
					calc_error(L, out, vL, eL);

					// Back-propagation step
					back_prop_out_layer(L, L2 + 1, v2, zL, eL, grad_act, deltaL, dwL, learn_rate);
					back_prop_hidden_layer(L2, L1 + 1, L, v1, z2, deltaL, WL, grad_act, delta2, dw2, learn_rate);
					back_prop_hidden_layer(L1, L0 + 1, L2, input, z1, delta2, W2, grad_act, delta1, dw1, learn_rate);

					// update dw's
					add(L, L2 + 1, dwL, _dwL);
					add(L2, L1 + 1, dw2, _dw2);
					add(L1, L0 + 1, dw1, _dw1);
				}

				// Update weights
				update_weights(L, L2 + 1, WL, _dwL, 1.0f / double(sz_batch), lambda * learn_rate);
				update_weights(L2, L1 + 1, W2, _dw2, 1.0f / double(sz_batch), lambda * learn_rate);
				update_weights(L1, L0 + 1, W1, _dw1, 1.0f / double(sz_batch), lambda * learn_rate);
			}

			// test 
			int correct_counter = 0;
			for (size_t i = (t % 10) * 1000; i < (t % 10 + 1) * 1000; i++){
				// Forward step
				full_connected_layer(test_digits[i], z1, L1, L0 + 1, W1);
				activation(z1, v1, L1, act_fun);
				full_connected_layer(v1, z2, L2, L1 + 1, W2);
				activation(z2, v2, L2, act_fun);
				full_connected_layer(v2, zL, L, L2 + 1, WL);
				activation(zL, vL, L, act_fun);
				if (classification(L, vL) == test_labels[i])correct_counter++;
			}
			cout << "accuracy = " << double(correct_counter) / 1000 << endl;
		}

		int correct_counter = 0;
		for (size_t i = 0; i < 10000; i++){
			// Forward step
			full_connected_layer(test_digits[i], z1, L1, L0 + 1, W1);
			activation(z1, v1, L1, act_fun);
			full_connected_layer(v1, z2, L2, L1 + 1, W2);
			activation(z2, v2, L2, act_fun);
			full_connected_layer(v2, zL, L, L2 + 1, WL);
			activation(zL, vL, L, act_fun);
			if (classification(L, vL) == test_labels[i])correct_counter++;
		}
		cout << "result accuracy = " << double(correct_counter) / 10000 << endl;
	}
	


	/*
	int num = 34567;
	Mat out;
	resize(train_digits[num], out, Size(64, 64),0,0, INTER_CUBIC);
	
	imshow("digit", 255 - out);
	cout<<"digit = " <<(int)train_labels[num];
	*/
	

	waitKey();
	/*
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) // check if we succeeded
		return -1;
	Mat edges;
	namedWindow("edges", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, edges, CV_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);
		imshow("edges", edges);
		if ( >= 0)
			break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor

	*/

	return 0;
}


/* 

MNIST FORMAT FILES

There are 4 files:

train-images-idx3-ubyte: training set images
train-labels-idx1-ubyte: training set labels
t10k-images-idx3-ubyte:  test set images
t10k-labels-idx1-ubyte:  test set labels

The training set contains 60000 examples, and the test set 10000 examples.

The first 5000 examples of the test set are taken from the original NIST training set. The last 5000 are taken from the original NIST test set. The first 5000 are cleaner and easier than the last 5000.

TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TEST SET IMAGE FILE (t10k-images-idx3-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).


*/














/* 
	Sample of using gpu in opencv

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

int main(int argc, char* argv [])
{
	try
	{
		cv::Mat src_host = cv::imread("file.png", CV_LOAD_IMAGE_GRAYSCALE);
		cv::gpu::GpuMat dst, src;
		src.upload(src_host);

		cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

		cv::Mat result_host;
		dst.download(result_host);

		cv::imshow("Result", result_host);
		cv::waitKey();
	}
	catch (const cv::Exception& ex)
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}
	return 0;
}

*/