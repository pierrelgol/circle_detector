#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <opencv4/opencv2/opencv.hpp>

std::size_t getCurrentRSS() {
        std::ifstream stat_stream("/proc/self/statm");
        std::size_t   rss_pages   = 0;
        std::size_t   total_pages = 0;
        stat_stream >> total_pages >> rss_pages;
        return rss_pages * sysconf(_SC_PAGESIZE);
}

static inline uchar clamp_u8_int(int v) {
        if (v < 0) return 0;
        if (v > 255) return 255;
        return static_cast< uchar >(v);
}

static inline void rgb_to_yuv(uint8_t r, uint8_t g, uint8_t b, uint8_t& y, uint8_t& u, uint8_t& v) {

        const int yi = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        const int ui = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        const int vi = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        y            = clamp_u8_int(yi);
        u            = clamp_u8_int(ui);
        v            = clamp_u8_int(vi);
}


static cv::Mat threshold_red_yuv420(const cv::Mat& bgr) {
        cv::Mat yuv;
        cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_I420);
        const int      w  = bgr.cols;
        const int      h  = bgr.rows;
        const int      hw = w >> 1;
        const uint8_t* y  = yuv.data;
        const uint8_t* u  = y + w * h;
        const uint8_t* v  = u + (w * h) / 4;

        uint8_t        target_y, target_u, target_v;
        rgb_to_yuv(255, 0, 0, target_y, target_u, target_v);
        (void)target_y;
        const uint8_t uv_tol = 40;
        const uint8_t y_min  = 80;

        cv::Mat       bin(h, w, CV_8UC1);
        for (int j = 0; j < h; ++j) {
                const uint8_t* yrow = y + j * w;
                const uint8_t* up   = u + (j >> 1) * hw;
                const uint8_t* vp   = v + (j >> 1) * hw;
                uint8_t*       dst  = bin.ptr< uint8_t >(j);
                for (int i = 0; i < w; ++i) {
                        const uint8_t uu = up[i >> 1];
                        const uint8_t vv = vp[i >> 1];
                        uint8_t       ok = (static_cast< uint8_t >(std::abs(int(uu) - int(target_u))) <= uv_tol) &
                                     (static_cast< uint8_t >(std::abs(int(vv) - int(target_v))) <= uv_tol);
                        if (ok) ok = (yrow[i] >= y_min);
                        dst[i] = ok ? 255u : 0u;
                }
        }
        return bin;
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

        const double min_area        = CV_PI * (0.5 * min_d) * (0.5 * min_d);
        const double max_area        = CV_PI * (0.5 * max_d) * (0.5 * max_d);

        const double circ_min_aspect = std::clamp(0.80, 0.60, 0.95);
        const double inv_ds          = 1.0;
        const double inv_ds2         = inv_ds * inv_ds;

        struct Cand {
                double      area_full;
                cv::Point2f c_full;
                float       r_full;
        };
        struct DrawElem {
                cv::Point2f c;
                float       r;
        };

        const int            iters = 1;
        cv::Mat              labels, stats, centroids;
        std::vector< Cand >  cands;

        auto                 t0  = std::chrono::high_resolution_clock::now();
        cv::Mat              bin = threshold_red_yuv420(img);


        static const cv::Mat k3  = cv::getStructuringElement(cv::MORPH_CROSS, {3, 3});
        cv::morphologyEx(bin, bin, cv::MORPH_OPEN, k3, {-1, -1}, 1);
        cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, k3, {-1, -1}, 1);

        int num_components = 0;
        for (int i = 0; i < iters; ++i) {
                num_components =
                    cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S, cv::CCL_SPAGHETTI);
                cands.clear();
                for (int lab = 1; lab < num_components; ++lab) {
                        int area_s = stats.at< int >(lab, cv::CC_STAT_AREA);
                        if (area_s < 4) continue;
                        int bb_w = stats.at< int >(lab, cv::CC_STAT_WIDTH);
                        int bb_h = stats.at< int >(lab, cv::CC_STAT_HEIGHT);
                        if (bb_w < 2 || bb_h < 2) continue;
                        double aspect = double(std::min(bb_w, bb_h)) / double(std::max(bb_w, bb_h));
                        if (aspect < circ_min_aspect) continue;
                        double extent = double(area_s) / double(bb_w * bb_h);
                        if (extent < 0.50) continue;
                        double area_full = double(area_s) * inv_ds2;
                        if (area_full < min_area || area_full > max_area) continue;
                        double      cx_s = centroids.at< double >(lab, 0);
                        double      cy_s = centroids.at< double >(lab, 1);
                        cv::Point2f c_full(float(cx_s * inv_ds), float(cy_s * inv_ds));
                        float       r_full = float(std::sqrt(area_full / CV_PI));
                        cands.push_back({area_full, c_full, r_full});
                }
                std::sort(
                    cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.area_full > b.area_full; });
                if (cands.size() > 10) cands.resize(10);
        }
        auto   t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration< double, std::milli >(t1 - t0).count() / iters;

        std::cout << "OpenCV Spaghetti Components: " << num_components - 1 << "\n";
        std::cout << "Candidates: " << cands.size() << "\n";
        std::cout << "OpenCV Spaghetti Avg Time (" << iters << " iters): " << ms << " ms\n";
        std::cout << "Memory used: " << getCurrentRSS() / (1024.0 * 1024.0) << " MB\n";

        if (show) {
                cv::RNG rng(123);
                cv::Mat vis = cv::Mat::zeros(labels.size(), CV_8UC3);
                for (int i = 1; i < num_components; ++i) {
                        int      x = stats.at< int >(i, cv::CC_STAT_LEFT);
                        int      y = stats.at< int >(i, cv::CC_STAT_TOP);
                        int      w = stats.at< int >(i, cv::CC_STAT_WIDTH);
                        int      h = stats.at< int >(i, cv::CC_STAT_HEIGHT);
                        cv::Rect bb(x - 2, y - 2, w + 4, h + 4);
                        cv::rectangle(vis, bb, cv::Scalar(255, 255, 255), 1);
                }


                std::vector< DrawElem > draws;
                if (!cands.empty()) {
                        for (const auto& c : cands) {
                                draws.push_back({c.c_full, c.r_full});
                        }
                } else {
                        for (int i = 1; i < num_components; ++i) {
                                float cx = static_cast< float >(centroids.at< double >(i, 0));
                                float cy = static_cast< float >(centroids.at< double >(i, 1));
                                int   w  = stats.at< int >(i, cv::CC_STAT_WIDTH);
                                int   h  = stats.at< int >(i, cv::CC_STAT_HEIGHT);
                                float r  = 0.5f * std::min(w, h);
                                draws.push_back({cv::Point2f(cx, cy), r});
                        }
                }

                for (size_t i = 0; i < draws.size(); ++i) {
                        const auto& d = draws[i];
                        cv::Scalar  col(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                        cv::circle(vis, d.c, d.r, col, 2);
                        cv::circle(vis, d.c, 2, cv::Scalar(0, 255, 255), cv::FILLED);
                        cv::putText(
                            vis, std::to_string(i), d.c + cv::Point2f(4, -4), cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
                }

                cv::imshow("Binary", bin);
                cv::imshow("OpenCV CCL_SPAGHETTI", vis);
                cv::waitKey(0);
                cv::destroyAllWindows();
        }
        return 0;
}
