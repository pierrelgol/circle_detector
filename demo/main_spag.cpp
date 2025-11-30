#include <chrono>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <opencv4/opencv2/opencv.hpp>

#include "../circleDetector.h"

std::size_t getCurrentRSS() {
        std::ifstream stat_stream("/proc/self/statm");
        std::size_t   rss_pages   = 0;
        std::size_t   total_pages = 0;
        stat_stream >> total_pages >> rss_pages;
        return rss_pages * sysconf(_SC_PAGESIZE);
}

int main() {
        bool        show = false;
        const char* path = "input.png";
        if (const char* env = std::getenv("SHOW")) show = (std::atoi(env) != 0);

        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
                std::cerr << "Error: cannot load input.png\n";
                return 1;
        }


        const double min_d           = 8.0;
        const double max_d           = 96.0;

        const double circ_min_aspect = std::clamp(0.80, 0.60, 0.95);


        cv::Mat      yuv;
        cv::cvtColor(img, yuv, cv::COLOR_BGR2YUV_I420);
        const uint8_t*          y = yuv.data;
        const uint8_t*          u = y + img.cols * img.rows;
        const uint8_t*          v = u + (img.cols * img.rows) / 4;

        std::vector< uint8_t >  mask(img.cols * img.rows);
        std::vector< uint8_t >  tmp1(img.cols * img.rows);
        std::vector< uint8_t >  tmp2(img.cols * img.rows);
        std::vector< int >      labels(img.cols * img.rows);
        std::vector< CDCircle > detections(16);

        CDConfig                cfg{};
        cfg.width                   = img.cols;
        cfg.height                  = img.rows;
        cfg.y                       = y;
        cfg.u                       = u;
        cfg.v                       = v;
        cfg.target_u                = 91;
        cfg.target_v                = 240;
        cfg.uv_tol                  = 40;
        cfg.y_min                   = 80;
        cfg.min_d                   = min_d;
        cfg.max_d                   = max_d;
        cfg.aspect_min              = circ_min_aspect;
        cfg.extent_min              = 0.50;
        cfg.max_out                 = static_cast< int >(detections.size());

        const double rss_before     = getCurrentRSS() / 1024.0;

        int          num_components = 0;
        auto         t0             = std::chrono::high_resolution_clock::now();
        int          found          = detectCircles(&cfg,
                                  detections.data(),
                                  cfg.max_out,
                                  mask.data(),
                                  tmp1.data(),
                                  tmp2.data(),
                                  labels.data(),
                                  &num_components);
        auto         t1             = std::chrono::high_resolution_clock::now();
        double       ms             = std::chrono::duration< double, std::milli >(t1 - t0).count();

        const double rss_after      = getCurrentRSS() / 1024.0;

        detections.resize(found);

        std::cout << "C Spaghetti8 Components: " << num_components - 1 << "\n";
        std::cout << "Candidates: " << detections.size() << "\n";
        std::cout << "C Spaghetti8 Time: " << ms << " ms\n";
        std::cout << "Memory before: " << rss_before << " KB, after: " << rss_after << " KB\n";

        if (show) {
                cv::RNG rng(123);
                cv::Mat mask_mat(img.rows, img.cols, CV_8UC1, mask.data());
                cv::Mat bin = mask_mat;
                cv::Mat vis = cv::Mat::zeros(bin.size(), CV_8UC3);
                struct Box {
                        int  x0, y0, x1, y1;
                        bool seen;
                };
                std::vector< Box > boxes(num_components, {img.cols, img.rows, -1, -1, false});
                for (int yb = 0; yb < img.rows; ++yb) {
                        const int* row = labels.data() + yb * img.cols;
                        for (int xb = 0; xb < img.cols; ++xb) {
                                int lbl = row[xb];
                                if (lbl <= 0 || lbl >= num_components) continue;
                                Box& b = boxes[lbl];
                                b.seen = true;
                                if (xb < b.x0) b.x0 = xb;
                                if (yb < b.y0) b.y0 = yb;
                                if (xb > b.x1) b.x1 = xb;
                                if (yb > b.y1) b.y1 = yb;
                        }
                }
                for (int i = 1; i < num_components; ++i) {
                        const Box& b = boxes[i];
                        if (!b.seen) continue;
                        cv::Rect bb(b.x0 - 2, b.y0 - 2, (b.x1 - b.x0 + 1) + 4, (b.y1 - b.y0 + 1) + 4);
                        cv::rectangle(vis, bb, cv::Scalar(255, 255, 255), 1);
                }

                std::vector< cv::Point2f > centers;
                std::vector< float >       radii;
                centers.reserve(std::max< size_t >(detections.size(), 1));
                radii.reserve(std::max< size_t >(detections.size(), 1));

                if (!detections.empty()) {
                        for (const auto& c : detections) {
                                centers.push_back({c.cx, c.cy});
                                radii.push_back(c.r);
                        }
                } else {
                        for (int lab = 1; lab < num_components; ++lab) {
                                const Box& b = boxes[lab];
                                if (!b.seen) continue;
                                float cx = 0.5f * float(b.x0 + b.x1 + 1);
                                float cy = 0.5f * float(b.y0 + b.y1 + 1);
                                float r  = 0.5f * std::min(b.x1 - b.x0 + 1, b.y1 - b.y0 + 1);
                                centers.push_back({cx, cy});
                                radii.push_back(r);
                        }
                }

                for (size_t i = 0; i < centers.size(); ++i) {
                        cv::Scalar col(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                        cv::circle(vis, centers[i], radii[i], col, 2);
                        cv::circle(vis, centers[i], 2, cv::Scalar(0, 255, 255), cv::FILLED);
                        cv::putText(vis,
                                    std::to_string(i),
                                    centers[i] + cv::Point2f(4, -4),
                                    cv::FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    col,
                                    1);
                }

                cv::imshow("Binary", bin);
                cv::imshow("C Spaghetti8", vis);
                cv::waitKey(0);
                cv::destroyAllWindows();
        }
        return 0;
}
