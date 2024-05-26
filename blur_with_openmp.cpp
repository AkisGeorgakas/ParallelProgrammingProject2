#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int KERNEL_RADIUS = 8;
const float sigma = 3.f;

unsigned char blurAxis(int x, int y, int channel, int axis/*0: horizontal axis, 1: vertical axis*/, unsigned char* input, int width, int height)
{
	float sum_weight = 0.0f;
	float ret = 0.f;

	for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; offset++)
	{
		int offset_x = axis == 0 ? offset : 0;
		int offset_y = axis == 1 ? offset : 0;
		int pixel_y = std::max(std::min(y + offset_y, height - 1), 0);
		int pixel_x = std::max(std::min(x + offset_x, width - 1), 0);
		int pixel = pixel_y * width + pixel_x;

		float weight = std::exp(-(offset * offset) / (2.f * sigma * sigma));

		ret += weight * input[4 * pixel + channel];
		sum_weight += weight;
	}
	ret /= sum_weight;

	return (unsigned char)std::max(std::min(ret, 255.f), 0.f);
}

int gaussian_blur_separate_serial(const char* filename)
{
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load an image into an array of unsigned chars that is the size of width * height * number of channels. The channels are the Red, Green, Blue and Alpha channels of the image.
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return 0;
	}

	unsigned char* img_horizontal_blur = new unsigned char[width * height * 4];
	unsigned char* img_out = new unsigned char[width * height * 4];

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	// Horizontal Blur
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_horizontal_blur[4 * pixel + channel] = blurAxis(x, y, channel, 0, img_in, width, height);
			}
		}
	}
	// Vertical Blur
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_out[4 * pixel + channel] = blurAxis(x, y, channel, 1, img_horizontal_blur, width, height);
			}
		}
	}
	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur Separate - Serial: Time %dms\n", time);

	// Write the blurred image into a JPG file
	stbi_write_jpg("blurred_separate.jpg", width, height, 4/*channels*/, img_out, 90 /*quality*/);

	stbi_image_free(img_in);
	delete[] img_horizontal_blur;
	delete[] img_out;
	return time;
}


int gaussian_blur_separate_parallel(const char* filename)
{
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load an image into an array of unsigned chars that is the size of width * height * number of channels. The channels are the Red, Green, Blue and Alpha channels of the image.
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return 0;
	}

	unsigned char* img_horizontal_blur = new unsigned char[width * height * 4];
	unsigned char* img_out = new unsigned char[width * height * 4];

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	
		// Horizontal Blur
		#pragma omp parallel for num_threads(8)
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int pixel = y * width + x;

				for (int channel = 0; channel < 4; channel++)
				{
					img_horizontal_blur[4 * pixel + channel] = blurAxis(x, y, channel, 0, img_in, width, height);
				}
			}
		}

		// Vertical Blur
		#pragma omp parallel for num_threads(8)
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int pixel = y * width + x;

				for (int channel = 0; channel < 4; channel++)
				{
					img_out[4 * pixel + channel] = blurAxis(x, y, channel, 1, img_horizontal_blur, width, height);
				}
			}
		}

	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur Separate - Parallel: Time %dms\n", time);

	// Write the blurred image into a JPG file
	stbi_write_jpg("blurred_separate_parallel.jpg", width, height, 4/*channels*/, img_out, 90 /*quality*/);

	stbi_image_free(img_in);
	delete[] img_horizontal_blur;
	delete[] img_out;
	return time;
}

int bloom_parallel(const char* filename) {
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load an image into an array of unsigned chars that is the size of width * height * number of channels. The channels are the Red, Green, Blue and Alpha channels of the image.
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return 0;
	}


	unsigned char* luminance = new unsigned char[width * height];
	unsigned char* bloom_mask = new unsigned char[width * height * 4];
	unsigned char* bloom_horizontal_blur = new unsigned char[width * height * 4];
	unsigned char* bloom_blur = new unsigned char[width * height * 4];
	unsigned char* bloom_final = new unsigned char[width * height * 4];
	unsigned char* img_out = new unsigned char[width * height * 4];
	unsigned char max_luminance = 0;

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	int rows_remainder = height % 8;
	int rows_quotient = height / 8;
	int current_thread_rows = (rows_quotient)+1;
	int offset = 0;
	int* start_rows = new int[8];
	int* end_rows = new int[8];
	for (int i = 0; i < 8; i++) {
		if (i == rows_remainder) { current_thread_rows--; }

		start_rows[i] = (rows_quotient * i) + offset; //Calculate the start of this thread's area. Offset accounts for the extra rows appointed to previous threads. 
		end_rows[i] = start_rows[i] + current_thread_rows;

		if (offset < rows_remainder) { offset++; }
	}

	#pragma omp parallel num_threads(8)
	{
		int id = omp_get_thread_num();

		for (int y = start_rows[id]; y < end_rows[id]; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int pixel = y * width + x;
				
				luminance[pixel] = (img_in[4* pixel] + img_in[4 * pixel+1] + img_in[4 * pixel + 2])/3;
				#pragma omp critical (max)
				{
					if (luminance[pixel] > max_luminance) { max_luminance = luminance[pixel]; }
				}
			}
		}
		#pragma omp barrier
		#pragma omp single nowait
		{
		std::cout << "Calculated maximum luminance of all pixels = " << (int)max_luminance << std::endl;
		}
		for (int y = start_rows[id]; y < end_rows[id]; y++){
			for (int x = 0; x < width; x++){
				
				int pixel = y * width + x;
				double ninety_percent_of_max = max_luminance * 0.9;
				
				if (luminance[pixel] >= ninety_percent_of_max) {
					for (int channel = 0; channel < 4; channel++){
						bloom_mask[4 * pixel + channel] = img_in[4 * pixel + channel];
					}
				}
				else {
					for (int channel = 0; channel < 4; channel++){
						bloom_mask[4 * pixel + channel] = 0;
					}
				}
			}
		}
		#pragma omp barrier

		for (int y = start_rows[id]; y < end_rows[id]; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int pixel = y * width + x;

				for (int channel = 0; channel < 4; channel++)
				{
					bloom_horizontal_blur[4 * pixel + channel] = blurAxis(x, y, channel, 0, bloom_mask, width, height);
				}
			}
		}
		#pragma omp barrier
		for (int y = start_rows[id]; y < end_rows[id]; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int pixel = y * width + x;

				for (int channel = 0; channel < 4; channel++)
				{
					bloom_blur[4 * pixel + channel] = blurAxis(x, y, channel,1, bloom_horizontal_blur, width, height);
				}
			}
		}
		#pragma omp barrier
		#pragma omp single nowait
		{
			stbi_write_jpg("bloom_blurred.jpg", width, height, 4/*channels*/, bloom_blur, 90 /*quality*/);
		}
		for (int y = start_rows[id]; y < end_rows[id]; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int pixel = y * width + x;

				for (int channel = 0; channel < 4; channel++)
				{
					bloom_final[4 * pixel + channel] = std::min(bloom_blur[4 * pixel + channel] + img_in[4 * pixel + channel],255);
				}
			}
		}
		#pragma omp barrier
		#pragma omp single nowait
		{
			stbi_write_jpg("bloom_final.jpg", width, height, 4/*channels*/, bloom_final, 90 /*quality*/);
		}
	}


	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Bloom Parallel: Time %dms\n", time);

	stbi_image_free(img_in);
	delete[] luminance;
	delete[] bloom_mask;
	delete[] bloom_horizontal_blur;
	delete[] bloom_blur;
	delete[] bloom_final;
	delete[] img_out;
	return time;
}
void time_tester_separate_parallel() {

	int parallel_4_threads = 0;
	int serial_times = 0;

	const char* filename = "street_night.jpg";


	for (int i = 0; i < 4; i++) {
		parallel_4_threads += gaussian_blur_separate_parallel(filename);
	}

	for (int i = 0; i < 4; i++) {
		serial_times += gaussian_blur_separate_serial(filename);
	}


	double avg_4_threads = ((double)parallel_4_threads) / 4;
	double avg_serial_times = ((double)serial_times) / 4;

	std::printf("Average run time for 4 threads: %10f ms\n", avg_4_threads);
	std::printf("Average serial run time: %10f ms\n", avg_serial_times);
}

int main() {

	const char* filename2 = "street_night.jpg";
	//gaussian_blur_separate_serial(filename2);
	bloom_parallel(filename2);
	//gaussian_blur_separate_parallel(filename2);
	//time_tester_separate_parallel();
	//simple(filename2);
	return 0;
}