__kernel
void img_resize(){ 
}

__kernel 
void pre_process(__global float *input_Mat, __global float *output_Mat, int h1, int height){ 
	int stride = get_global_size(0);
	int batch_size = get_global_id(0);
	int global_row = get_global_id(2); //0~320 //new_height
	int global_col = get_global_id(1); //0~480 //new_width
	int new_height = get_global_size(2);
	int new_width = get_global_size(1);
		
	output_Mat[(batch_size*new_width*height * 3) + (global_row + h1)* new_width * 3 + global_col * 3 + 0] =	input_Mat[(batch_size*new_width*new_height * 3) + (global_row * new_width + global_col) * 3 + 0];
	output_Mat[(batch_size*new_width*height * 3) + (global_row + h1)* new_width * 3 + global_col * 3 + 1] = input_Mat[(batch_size*new_width*new_height * 3) + (global_row * new_width + global_col) * 3 + 1];
	output_Mat[(batch_size*new_width*height * 3) + (global_row + h1)* new_width * 3 + global_col * 3 + 2] = input_Mat[(batch_size*new_width*new_height * 3) + (global_row * new_width + global_col) * 3 + 2];	
}

//for (int i = 0; i < batch_size; i++) {
	//	cv::Mat a = cv::Mat(cv::Size(width, height), CV_32FC3, cv::Scalar(0, 0, 0));
	//	for (int j = 0; j < new_height; j++) {
	//		for (int k = 0; k < new_width; k++) {
	//			for (int z = 0; z < 3; z++) {
	//				output_mem[i*width*height * 3 + (j+h1) * new_width * 3 + k * 3 + z] =
	//					input_mem[(i*width*new_height * 3) + j * new_width * 3 + k * 3 + z];
	//				///*
	//				//output_mem[(new_width * (new_height * i + (j+h1)) + k) * 3 + z] = 
	//				//	input_mem[(new_width * (new_height * i + j) + k) * 3 + z];
	//				//*/
	//			}
	//		}
	//	}
	//}