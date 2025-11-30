#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
        float  cx;
        float  cy;
        float  r;
        double area;
} CDCircle;

typedef struct {
        int            width;  // full-resolution width (Y plane)
        int            height; // full-resolution height (Y plane)
        const uint8_t* y;      // full-res luma
        const uint8_t* u;      // half-res chroma U
        const uint8_t* v;      // half-res chroma V

        // Target color in YUV (Y ignored by default detection logic, use y_min).
        uint8_t target_u;
        uint8_t target_v;
        uint8_t uv_tol; // allowable absolute diff on U and V
        uint8_t y_min;  // minimum luma (0 to disable)

        double  min_d;      // minimum expected diameter (pixels) at full-res
        double  max_d;      // maximum expected diameter (pixels) at full-res
        double  aspect_min; // minimum aspect ratio to consider circularity
        double  extent_min; // minimum extent
        int     max_out;    // cap on number of outputs
} CDConfig;

// Detect circles from an I420 buffer. Runs threshold -> morphology (3x3 open+close)
// -> 8-connectivity CCL -> filtering. Returns number of detections written to out.
// Caller must provide working buffers:
//  - mask:   width*height bytes, tightly packed.
//  - tmp1/2: width*height bytes scratch.
//  - labels: width*height ints used by CCL.
int detectCircles(const CDConfig* cfg,
                  CDCircle*       out,
                  int             out_cap,
                  uint8_t*        mask,
                  uint8_t*        tmp1,
                  uint8_t*        tmp2,
                  int*            labels,
                  int*            num_components_out);

#ifdef __cplusplus
}
#endif
