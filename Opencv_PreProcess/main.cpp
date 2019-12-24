#include <opencv/highgui.h>
#include <opencv2/world.hpp>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <CL/cl.h>
#include <chrono>
using namespace std::chrono;
#define  CHECK_ERROR(err) \
	if (err != CL_SUCCESS) { \
	printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
	exit(EXIT_FAILURE); \
	}
char *get_source_code(const char *file_name, size_t *len) {
	char *source_code;
	size_t length;
	FILE *file;
	errno_t err4 = fopen_s(&file, file_name, "rb");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char *)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';

	fclose(file);
	*len = length;
	return source_code;
}

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
char *kernel_source;
size_t kernel_source_size;
cl_kernel kernel;
cl_int err;

void releaseKernel()
{
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseDevice(device);
}

void kernelInit()
{
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	CHECK_ERROR(err);

	kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(program, 1, &device, "", NULL, NULL);

	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char *log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);
		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error : \n%s\n", log);
		free(log);
		exit(0);
	}
	CHECK_ERROR(err);
}

float* get_Mat_memory_float(std::vector<cv::Mat> imgs);
void pre_processing(float *input_data, float *output_data);
cv::Mat resize_img(cv::Mat img, bool is_input);
void show_img(cv::Size img, float* mem);
const int batch_size = 64;
const int width = 480;
const int height = 480;
int new_width = 0;
int new_height = 0;
int h1;
int h2;
const int mem_size = width * height * 3 * sizeof(float);
class Parallel_process : public cv::ParallelLoopBody
{
private:
	std::vector<cv::Mat> &imgs;
	bool is_input;
public:
	Parallel_process(std::vector<cv::Mat> &input, bool is_input) : imgs(input), is_input(is_input){}

	void operator() (const cv::Range& range) const
	{
		for (int i = range.start; i < range.end; i++) {
			cv::Mat img = imgs[i];
			cv::Size net(width, height);
			if (is_input) {
				cv::Size newSize = img.size();

				if (net.width / float(newSize.width) < net.height / float(newSize.height))
				{
					newSize.height = (int)((newSize.height * net.width) / newSize.width);
					newSize.width = net.width;
				}
				else
				{
					newSize.width = (int)((newSize.width * net.height) / newSize.height);
					newSize.height = net.height;
				}
				h1 = (height - new_height) / 2;
				h2 = (height + new_height) / 2;

				cv::Mat temp;				
				cv::cvtColor(img, temp, cv::COLOR_BGR2RGB);
				temp.convertTo(temp, CV_32FC3);
				temp /= 255.;
				new_width = newSize.width;
				new_height = newSize.height;
				cv::resize(temp, temp, cv::Size(newSize.width, newSize.height));
				imgs[i] = temp;
			}

			else {
				cv::Mat new_image = cv::Mat(cv::Size(net.width, net.height), CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
				imgs[i] = new_image;
			}
		}	
	}
	
	std::vector<cv::Mat> get_imgs() {
		return this->imgs;
	}	
};

int main(void) {
	auto time1 = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> input_imgs;
	std::vector<cv::Mat> output_imgs;
	for (int i = 0; i < batch_size; i++) {
		cv::Mat img;
		if(i % 2 == 0)
			img = cv::imread("test2.png", cv::IMREAD_COLOR);
		else
			img = cv::imread("woohee.png", cv::IMREAD_COLOR);
		
		cv::resize(img, img, cv::Size(480, 480));

		/*img = resize_img(img, true);
		cv::Mat img2 = resize_img(img, false);
		input_imgs.push_back(img);
		output_imgs.push_back(img2);*/
		
		input_imgs.push_back(img);
		output_imgs.push_back(img);
	}
	auto time2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();
	std::cout << "Image Input time : " << duration << "milliseconds" << std::endl;
	Parallel_process input_parallel(input_imgs, true);
	Parallel_process output_parallel(output_imgs, false);
	cv::parallel_for_(cv::Range(0, batch_size), input_parallel);
	cv::parallel_for_(cv::Range(0, batch_size), output_parallel);
	input_imgs = input_parallel.get_imgs();
	output_imgs = output_parallel.get_imgs();
	auto time3 = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2).count();
	std::cout << "Image resize time : " << duration << "milliseconds" << std::endl;
	float* input_mem = get_Mat_memory_float(input_imgs);
	float* output_mem = get_Mat_memory_float(output_imgs);
	auto time4 = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3).count();
	std::cout << "memory allocation time : " << duration << "milliseconds" << std::endl;
	//show_img(cv::Size(width, height), input_mem);
	//show_img(cv::Size(width, height), output_mem);
	auto time5 = std::chrono::high_resolution_clock::now();
	kernelInit();
	//auto time1 = std::chrono::high_resolution_clock::now();
	pre_processing(input_mem, output_mem);
	auto time6 = std::chrono::high_resolution_clock::now();
	releaseKernel();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(time6 - time5).count();
	std::cout << "openCL code time : " << duration << "milliseconds" << std::endl;
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(time6 - time2).count();
	std::cout << "Total code time : " << duration << "milliseconds" << std::endl;
	//show_img(cv::Size(width, height), output_mem);
	
}

void show_img(cv::Size img, float* mem) {
	for (int i = 0; i < batch_size; i++) {
		cv::Mat new_image = cv::Mat(cv::Size(img.width, img.height), CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
		int mem_size = width * height * 3;
		memcpy(new_image.data, mem + (i * mem_size), mem_size * sizeof(float));
		cv::imshow("ttt", new_image);
		cv::waitKey(0);
	}
}

float* get_Mat_memory_float(std::vector<cv::Mat> imgs) {
	cv::Size img_size = imgs[0].size();
	/*cv::Mat temp = imgs[0];
	cv::Mat batch_mat(img_size.width * batch_size, img_size.height, imgs[0].type());*/
	float *batch_mat = new float[batch_size * img_size.width * img_size.height * 3];
	//unsigned char* batch_mem = batch_mat.data;
	for (int i = 0; i < batch_size; i++) {		
		size_t mem_size =  3 * img_size.height * img_size.width; //type = channel * data_type
		memcpy(batch_mat + (mem_size * i), (float*)imgs[i].data, img_size.height * img_size.width * sizeof(float) * 3);
	}

	//memset(batch_mat, 0, img_size.height * img_size.width * sizeof(float) * 3 * 2);
	return batch_mat;
}

cv::Mat resize_img(cv::Mat img, bool is_input) {
	cv::Size net(width, height);
	if (is_input) {		
		cv::Size newSize = img.size();

		if (net.width / float(newSize.width) < net.height / float(newSize.height))
		{
			newSize.height = (int)((newSize.height * net.width) / newSize.width);
			newSize.width = net.width;
		}
		else
		{
			newSize.width = (int)((newSize.width * net.height) / newSize.height);
			newSize.height = net.height;
		}
		h1 = (height - new_height) / 2;
		h2 = (height + new_height) / 2;

		cv::Mat temp;
		cv::cvtColor(img, temp, cv::COLOR_BGR2RGB);
		temp.convertTo(temp, CV_32FC3);
		temp /= 255.;
		new_width = newSize.width;
		new_height = newSize.height;
		cv::resize(temp, temp, cv::Size(newSize.width, newSize.height));
		return temp;
	}
	
	else {
		cv::Mat new_image = cv::Mat(cv::Size(net.width, net.height), CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
		return new_image;
	}	
}

void pre_processing(float *input_data, float *output_data)
{
	kernel = clCreateKernel(program, "pre_process", &err);
	CHECK_ERROR(err);
	size_t in_size = sizeof(float) * new_width * new_height * 3 * batch_size;
	size_t out_size = sizeof(float) * width * height * 3 * batch_size;
	cl_mem m_img, m_dst;
	

	m_img = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, in_size, input_data, &err);
	CHECK_ERROR(err);
	m_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, out_size, output_data, &err);
	CHECK_ERROR(err);
	
	/*err = clEnqueueWriteBuffer(queue, m_img, CL_FALSE, 0, in_size, input_data, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, m_dst, CL_FALSE, 0, out_size, output_data, 0, NULL, NULL);
	CHECK_ERROR(err);*/

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_img);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &m_dst);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(int), &h1);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 3, sizeof(int), &height);
	CHECK_ERROR(err);
	
	size_t global_size[3] = { batch_size, new_width, new_height};
	
	//cl_event event;
	
	err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);

	clFinish(queue);

	/*err = clEnqueueReadBuffer(queue, m_dst, CL_TRUE, 0, out_size, output_data, 0, NULL, NULL);
	CHECK_ERROR(err);*/

	
	//clWaitForEvents(1, &event);
	clReleaseMemObject(m_img);
	clReleaseMemObject(m_dst);
	clReleaseKernel(kernel);
}