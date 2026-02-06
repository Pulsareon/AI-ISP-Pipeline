/*
 * Pulsareon AI-ISP (ncnn Edition) - Runnable Version
 * Target: PC Simulation / Snapdragon Port
 */

#include "net.h"
#include "gpu.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>

// Configuration
const int TILE_SIZE = 512;
const int TILE_OVERLAP = 32;
const int TARGET_LATENCY_MS = 16; 

class AI_ISP_Engine {
private:
    ncnn::Net net;
    ncnn::VulkanDevice* vkdev = 0;
    ncnn::Option opt;

public:
    AI_ISP_Engine() {
        // 1. Initialize Vulkan
#if NCNN_VULKAN
        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count > 0) {
            vkdev = ncnn::get_gpu_device(0);
            opt.use_vulkan_compute = true;
            opt.use_fp16_packed = true;
            opt.use_fp16_storage = true;
            printf("⚡ ncnn Vulkan Initialized\n");
        }
#endif
        opt.num_threads = 4;
        
        // Load dummy model for compilation test
        // In real usage: net.load_param("model.param");
    }

    ~AI_ISP_Engine() {
        net.clear();
    }

    cv::Mat process_frame_tiled(const cv::Mat& raw_input) {
        int w = raw_input.cols;
        int h = raw_input.rows;
        
        // Output RGB
        cv::Mat output = cv::Mat::zeros(h, w, CV_8UC3);

        for (int y = 0; y < h; y += (TILE_SIZE - TILE_OVERLAP * 2)) {
            for (int x = 0; x < w; x += (TILE_SIZE - TILE_OVERLAP * 2)) {
                
                // 1. Calculate safe crop coordinates
                int x_start = std::max(0, x - TILE_OVERLAP);
                int y_start = std::max(0, y - TILE_OVERLAP);
                int x_end = std::min(w, x + TILE_SIZE - TILE_OVERLAP); // Correction for boundary
                int y_end = std::min(h, y + TILE_SIZE - TILE_OVERLAP);
                
                // Adjust to ensure tile size fits
                if (x_end - x_start > TILE_SIZE) x_end = x_start + TILE_SIZE;
                if (y_end - y_start > TILE_SIZE) y_end = y_start + TILE_SIZE;

                cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);
                cv::Mat cv_tile = raw_input(roi).clone(); // Clone to ensure continuous memory

                // 2. ncnn wrapping
                ncnn::Mat in_tile = ncnn::Mat::from_pixels(
                    cv_tile.data, ncnn::Mat::PIXEL_GRAY, roi.width, roi.height
                );

                // 3. Inference
                ncnn::Extractor ex = net.create_extractor();
                ex.set_opt(opt);
                
                // Mock inference (Direct pass-through for compilation check)
                // ex.input("in", in_tile);
                // ncnn::Mat out_tile;
                // ex.extract("out", out_tile);
                
                // Simulate output (Gray to RGB conversion simulation)
                // In real code: out_tile.to_pixels(...)
                cv::Mat processed_tile;
                cv::cvtColor(cv_tile, processed_tile, cv::COLOR_GRAY2BGR);

                // 4. Merge back (Handling Overlap)
                // Calculate valid region (center of tile, excluding overlap)
                int valid_x_start = (x == 0) ? 0 : TILE_OVERLAP;
                int valid_y_start = (y == 0) ? 0 : TILE_OVERLAP;
                int valid_x_end = (x_end == w) ? roi.width : roi.width - TILE_OVERLAP;
                int valid_y_end = (y_end == h) ? roi.height : roi.height - TILE_OVERLAP;
                
                cv::Rect valid_roi_local(valid_x_start, valid_y_start, valid_x_end - valid_x_start, valid_y_end - valid_y_start);
                cv::Rect valid_roi_global(x_start + valid_x_start, y_start + valid_y_start, valid_roi_local.width, valid_roi_local.height);
                
                // Ensure bounds are safe
                valid_roi_global &= cv::Rect(0, 0, w, h);
                valid_roi_local.width = valid_roi_global.width;
                valid_roi_local.height = valid_roi_global.height;

                if (valid_roi_global.area() > 0) {
                    processed_tile(valid_roi_local).copyTo(output(valid_roi_global));
                }
            }
        }
        return output;
    }

    void benchmark() {
        printf("🚀 Starting Benchmark (Simulated)...\n");
        cv::Mat input = cv::Mat::zeros(2160, 3840, CV_8UC1);
        
        auto start = std::chrono::high_resolution_clock::now();
        process_frame_tiled(input);
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("⏱️ Time: %.2f ms\n", ms);
    }
};

int main() {
    AI_ISP_Engine engine;
    engine.benchmark();
    return 0;
}
