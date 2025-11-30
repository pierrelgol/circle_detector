#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "circleDetector.h"


static inline uint8_t clamp_u8(int v) {
        if (v < 0) return 0;
        if (v > 255) return 255;
        return (uint8_t)v;
}


static inline uint8_t abs_u8_diff(uint8_t a, uint8_t b) {
        return (uint8_t)((a > b) ? (a - b) : (b - a));
}

static void make_color_mask_i420_full(const uint8_t* restrict y,
                                      const uint8_t* restrict u,
                                      const uint8_t* restrict v,
                                      int     width,
                                      int     height,
                                      uint8_t target_u,
                                      uint8_t target_v,
                                      uint8_t uv_tol,
                                      uint8_t y_min,
                                      uint8_t* restrict mask) {
        const int hw = width >> 1;
        for (int j = 0; j < height; ++j) {
                const uint8_t* up   = u + (j >> 1) * hw;
                const uint8_t* vp   = v + (j >> 1) * hw;
                const uint8_t* yrow = y + j * width;
                uint8_t*       dst  = mask + j * width;
                int            i    = 0;
                for (; i + 16 <= width; i += 16) {
                        uint8_t mvec[16];
                        for (int k = 0; k < 16; ++k) {
                                const int     x  = i + k;
                                const uint8_t uu = up[x >> 1];
                                const uint8_t vv = vp[x >> 1];
                                uint8_t       ok = ((int)abs_u8_diff(uu, target_u) <= uv_tol) &
                                             (int)(abs_u8_diff(vv, target_v) <= uv_tol);
                                if (ok && y_min) {
                                        ok &= (yrow[x] >= y_min);
                                }
                                mvec[k] = ok ? 255u : 0u;
                        }
                        memcpy(dst + i, mvec, 16);
                }
                for (; i < width; ++i) {
                        const uint8_t uu = up[i >> 1];
                        const uint8_t vv = vp[i >> 1];
                        uint8_t       ok =
                            ((int)abs_u8_diff(uu, target_u) <= uv_tol) & (int)(abs_u8_diff(vv, target_v) <= uv_tol);
                        if (ok && y_min) {
                                ok &= (yrow[i] >= y_min);
                        }
                        dst[i] = ok ? 255u : 0u;
                }
        }
}

static void erode3x3_cross(const uint8_t* restrict src, int w, int h, uint8_t* restrict dst) {
        for (int y = 0; y < h; ++y) {
                const int yw  = y * w;
                const int ym1 = (y - 1) * w;
                const int yp1 = (y + 1) * w;
                for (int x = 0; x < w; ++x) {
                        const int idx = yw + x;
                        if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
                                dst[idx] = 0;
                                continue;
                        }
                        uint8_t m   = src[idx];
                        uint8_t val = src[idx - 1];
                        if (val < m) m = val;
                        val = src[idx + 1];
                        if (val < m) m = val;
                        val = src[ym1 + x];
                        if (val < m) m = val;
                        val = src[yp1 + x];
                        if (val < m) m = val;
                        dst[idx] = m;
                }
        }
}

static void dilate3x3_cross(const uint8_t* restrict src, int w, int h, uint8_t* restrict dst) {
        for (int y = 0; y < h; ++y) {
                const int yw  = y * w;
                const int ym1 = (y - 1) * w;
                const int yp1 = (y + 1) * w;
                for (int x = 0; x < w; ++x) {
                        const int idx = yw + x;
                        uint8_t   m   = src[idx];
                        if (x > 0) {
                                uint8_t val = src[idx - 1];
                                if (val > m) m = val;
                        }
                        if (x + 1 < w) {
                                uint8_t val = src[idx + 1];
                                if (val > m) m = val;
                        }
                        if (y > 0) {
                                uint8_t val = src[ym1 + x];
                                if (val > m) m = val;
                        }
                        if (y + 1 < h) {
                                uint8_t val = src[yp1 + x];
                                if (val > m) m = val;
                        }
                        dst[idx] = m;
                }
        }
}

static void
morph_open_close_3x3(uint8_t* restrict img, int width, int height, uint8_t* restrict tmp1, uint8_t* restrict tmp2) {
        if (!img || width <= 0 || height <= 0) return;
        erode3x3_cross(img, width, height, tmp1);
        dilate3x3_cross(tmp1, width, height, tmp2);
        dilate3x3_cross(tmp2, width, height, tmp1);
        erode3x3_cross(tmp1, width, height, img);
}


typedef struct {
        int      minx, miny, maxx, maxy;
        int      area;
        uint64_t sumx, sumy;
        uint8_t  seen;
} BoxStats;

static int        spaghetti8_label(const uint8_t* img, int width, int height, int* labels_out);
static inline int findRoot(const int* P, int i) {
        int root = i;
        while (P[root] < root) {
                root = P[root];
        }
        return root;
}
static inline void setRoot(int* P, int i, int root) {
        while (P[i] < i) {
                int j = P[i];
                P[i]  = root;
                i     = j;
        }
        P[i] = root;
}
static inline int findUF(int* P, int i) {
        int root = findRoot(P, i);
        setRoot(P, i, root);
        return root;
}
static inline int set_union(int* P, int i, int j) {
        int root = findRoot(P, i);
        if (i != j) {
                int rootj = findRoot(P, j);
                if (root > rootj) {
                        root = rootj;
                }
                setRoot(P, j, root);
        }
        return root;
}
static inline void flattenLParallel(int* P, int start, int nElem, int* k) {
        for (int i = start; i < start + nElem; ++i) {
                if (P[i] < i) {
                        P[i] = P[P[i]];
                } else {
                        P[i] = *k;
                        (*k)++;
                }
        }
}
static inline int stripeFirstLabel8Connectivity(int y, int w) {
        ((void)0);
        return (y / 2) * ((w + 1) / 2) + 1;
}
static int spaghetti8_label(const uint8_t* img, int width, int height, int* labels_out) {
        const int ow    = width;
        const int oh    = height;
        const int w_pad = ow + 4;
        const int h_pad = oh;
        (void)(h_pad);
        const int    max_labels    = ((oh + 1) / 2) * ((ow + 1) / 2) + 4;
        const size_t pad_pixels    = (size_t)w_pad * h_pad;
        const size_t labels_needed = (size_t)ow * oh;

        int*         P_            = (int*)malloc(((size_t)max_labels + 2) * sizeof(int));
        uint8_t*     img_pad       = (uint8_t*)malloc(pad_pixels);
        int*         labels_pad    = (int*)malloc(pad_pixels * sizeof(int));

        if (!P_ || !img_pad || !labels_pad) {
                if (P_) free(P_);
                if (img_pad) free(img_pad);
                if (labels_pad) free(labels_pad);
                return 0;
        }

        memset(P_, 0, ((size_t)max_labels + 2) * sizeof(int));
        memset(img_pad, 0, pad_pixels);
        memset(labels_pad, 0, pad_pixels * sizeof(int));
        for (int y = 0; y < oh; ++y) {
                memcpy(img_pad + y * w_pad + 2, img + y * ow, (size_t)ow);
        }
        memset(labels_out, 0, labels_needed * sizeof(int));
        int       label      = stripeFirstLabel8Connectivity(0, ow);
        const int firstLabel = label;
        const int w          = w_pad;
        if (oh == 1) {
                const uint8_t* const img_row        = img_pad + 2;
                int* const           img_labels_row = labels_pad + 2;
                int                  c              = -2;
        sl_tree_0:
                if ((c += 2) >= w - 2) {
                        if (c > w - 2) {
                                goto sl_break_0_0;
                        } else {
                                goto sl_break_1_0;
                        }
                }
                if (img_row[c] > 0) {
                        if (img_row[c + 1] > 0) {
                                img_labels_row[c] = label;
                                P_[label]         = label;
                                ((void)0);
                                label = label + 1;
                                goto sl_tree_1;
                        } else {
                                img_labels_row[c] = label;
                                P_[label]         = label;
                                ((void)0);
                                label = label + 1;
                                goto sl_tree_0;
                        }
                } else {
                NODE_372:
                        if (img_row[c + 1] > 0) {
                                img_labels_row[c] = label;
                                P_[label]         = label;
                                ((void)0);
                                label = label + 1;
                                goto sl_tree_1;
                        } else {
                                img_labels_row[c] = 0;
                                goto sl_tree_0;
                        }
                }
        sl_tree_1:
                if ((c += 2) >= w - 2) {
                        if (c > w - 2) {
                                goto sl_break_0_1;
                        } else {
                                goto sl_break_1_1;
                        }
                }
                if (img_row[c] > 0) {
                        if (img_row[c + 1] > 0) {
                                img_labels_row[c] = img_labels_row[c - 2];
                                goto sl_tree_1;
                        } else {
                                img_labels_row[c] = img_labels_row[c - 2];
                                goto sl_tree_0;
                        }
                } else {
                        goto NODE_372;
                }
        sl_break_0_0:
                if (img_row[c] > 0) {
                        img_labels_row[c] = label;
                        P_[label]         = label;
                        ((void)0);
                        label = label + 1;
                } else {
                        img_labels_row[c] = 0;
                }
                goto end_sl;
        sl_break_0_1:
                if (img_row[c] > 0) {
                        img_labels_row[c] = img_labels_row[c - 2];
                } else {
                        img_labels_row[c] = 0;
                }
                goto end_sl;
        sl_break_1_0:
                if (img_row[c] > 0) {
                        if (img_row[c + 1] > 0) {
                                img_labels_row[c] = label;
                                P_[label]         = label;
                                ((void)0);
                                label = label + 1;
                        } else {
                                img_labels_row[c] = label;
                                P_[label]         = label;
                                ((void)0);
                                label = label + 1;
                        }
                } else {
                NODE_375:
                        if (img_row[c + 1] > 0) {
                                img_labels_row[c] = label;
                                P_[label]         = label;
                                ((void)0);
                                label = label + 1;
                        } else {
                                img_labels_row[c] = 0;
                        }
                }
                goto end_sl;
        sl_break_1_1:
                if (img_row[c] > 0) {
                        if (img_row[c + 1] > 0) {
                                img_labels_row[c] = img_labels_row[c - 2];
                        } else {
                                img_labels_row[c] = img_labels_row[c - 2];
                        }
                } else {
                        goto NODE_375;
                }
                goto end_sl;
        end_sl:;
        } else {
                {
                        const uint8_t* const img_row        = img_pad + 2;
                        const uint8_t* const img_row_fol    = img_row + w_pad;
                        int* const           img_labels_row = labels_pad + 2;
                        int                  c              = -2;
                fl_tree_0:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto fl_break_0_0;
                                } else {
                                        goto fl_break_1_0;
                                }
                        }
                        if (img_row[c] > 0) {
                        NODE_253:
                                if (img_row[c + 1] > 0) {
                                        img_labels_row[c] = label;
                                        P_[label]         = label;
                                        ((void)0);
                                        label = label + 1;
                                        goto fl_tree_1;
                                } else {
                                        img_labels_row[c] = label;
                                        P_[label]         = label;
                                        ((void)0);
                                        label = label + 1;
                                        goto fl_tree_2;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        goto NODE_253;
                                } else {
                                NODE_255:
                                        if (img_row[c + 1] > 0) {
                                                img_labels_row[c] = label;
                                                P_[label]         = label;
                                                ((void)0);
                                                label = label + 1;
                                                goto fl_tree_1;
                                        } else {
                                                if (img_row_fol[c + 1] > 0) {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                        goto fl_tree_1;
                                                } else {
                                                        img_labels_row[c] = 0;
                                                        goto fl_tree_0;
                                                }
                                        }
                                }
                        }
                fl_tree_1:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto fl_break_0_1;
                                } else {
                                        goto fl_break_1_1;
                                }
                        }
                        if (img_row[c] > 0) {
                        NODE_257:
                                if (img_row[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                        goto fl_tree_1;
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                        goto fl_tree_2;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        goto NODE_257;
                                } else {
                                        goto NODE_255;
                                }
                        }
                fl_tree_2:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto fl_break_0_2;
                                } else {
                                        goto fl_break_1_2;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                        goto NODE_257;
                                } else {
                                        goto NODE_253;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_fol[c - 1] > 0) {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto fl_tree_1;
                                                } else {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                        goto fl_tree_1;
                                                }
                                        } else {
                                                if (img_row_fol[c - 1] > 0) {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto fl_tree_2;
                                                } else {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                        goto fl_tree_2;
                                                }
                                        }
                                } else {
                                        goto NODE_255;
                                }
                        }
                fl_break_0_0:
                        if (img_row[c] > 0) {
                                img_labels_row[c] = label;
                                P_[label]         = label;
                                ((void)0);
                                label = label + 1;
                        } else {
                                if (img_row_fol[c] > 0) {
                                        img_labels_row[c] = label;
                                        P_[label]         = label;
                                        ((void)0);
                                        label = label + 1;
                                } else {
                                        img_labels_row[c] = 0;
                                }
                        }
                        goto end_fl;
                fl_break_0_1:
                        if (img_row[c] > 0) {
                                img_labels_row[c] = img_labels_row[c - 2];
                        } else {
                                if (img_row_fol[c] > 0) {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                } else {
                                        img_labels_row[c] = 0;
                                }
                        }
                        goto end_fl;
                fl_break_0_2:
                        if (img_row[c] > 0) {
                        NODE_266:
                                if (img_row_fol[c - 1] > 0) {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                } else {
                                        img_labels_row[c] = label;
                                        P_[label]         = label;
                                        ((void)0);
                                        label = label + 1;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        goto NODE_266;
                                } else {
                                        img_labels_row[c] = 0;
                                }
                        }
                        goto end_fl;
                fl_break_1_0:
                        if (img_row[c] > 0) {
                        NODE_268:
                                if (img_row[c + 1] > 0) {
                                        img_labels_row[c] = label;
                                        P_[label]         = label;
                                        ((void)0);
                                        label = label + 1;
                                } else {
                                        img_labels_row[c] = label;
                                        P_[label]         = label;
                                        ((void)0);
                                        label = label + 1;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        goto NODE_268;
                                } else {
                                NODE_270:
                                        if (img_row[c + 1] > 0) {
                                                img_labels_row[c] = label;
                                                P_[label]         = label;
                                                ((void)0);
                                                label = label + 1;
                                        } else {
                                                if (img_row_fol[c + 1] > 0) {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                } else {
                                                        img_labels_row[c] = 0;
                                                }
                                        }
                                }
                        }
                        goto end_fl;
                fl_break_1_1:
                        if (img_row[c] > 0) {
                        NODE_272:
                                if (img_row[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        goto NODE_272;
                                } else {
                                        goto NODE_270;
                                }
                        }
                        goto end_fl;
                fl_break_1_2:
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                        goto NODE_272;
                                } else {
                                        goto NODE_268;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_266;
                                        } else {
                                                goto NODE_266;
                                        }
                                } else {
                                        goto NODE_270;
                                }
                        }
                        goto end_fl;
                end_fl:;
                }
                const int e_rows = oh & -2;
                for (int r = 2; r < e_rows; r += 2) {
                        const uint8_t* const img_row                  = img_pad + r * w_pad + 2;
                        const uint8_t* const img_row_prev             = img_row - w_pad;
                        const uint8_t* const img_row_prev_prev        = img_row_prev - w_pad;
                        const uint8_t* const img_row_fol              = img_row + w_pad;
                        int* const           img_labels_row           = labels_pad + r * w_pad + 2;
                        int* const           img_labels_row_prev_prev = img_labels_row - 2 * w_pad;
                        int                  c                        = -2;
                        goto tree_0;
                tree_0:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_0;
                                } else {
                                        goto break_1_0;
                                }
                        }
                        if (img_row[c] > 0) {
                        NODE_1:
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        goto tree_11;
                                } else {
                                        if (img_row[c + 1] > 0) {
                                        NODE_3:
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev[c] > 0) {
                                                        NODE_5:
                                                                if (img_row_prev_prev[c + 1] > 0) {
                                                                        img_labels_row[c] =
                                                                            img_labels_row_prev_prev[c + 2];
                                                                        goto tree_5;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row_prev_prev[c + 2]);
                                                                        goto tree_5;
                                                                }
                                                        } else {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                                goto tree_5;
                                                        }
                                                } else {
                                                        if (img_row_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                goto tree_4;
                                                        } else {
                                                                img_labels_row[c] = label;
                                                                P_[label]         = label;
                                                                ((void)0);
                                                                label = label + 1;
                                                                goto tree_3;
                                                        }
                                                }
                                        } else {
                                                if (img_row_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto tree_10;
                                                } else {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                        goto tree_9;
                                                }
                                        }
                                }
                        } else {
                        NODE_8:
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                        NODE_10:
                                                if (img_row_prev[c + 1] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto tree_6;
                                                } else {
                                                        goto NODE_3;
                                                }
                                        } else {
                                                img_labels_row[c] = label;
                                                P_[label]         = label;
                                                ((void)0);
                                                label = label + 1;
                                                goto tree_7;
                                        }
                                } else {
                                NODE_11:
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_10;
                                        } else {
                                        NODE_12:
                                                if (img_row_fol[c + 1] > 0) {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                        goto tree_2;
                                                } else {
                                                        img_labels_row[c] = 0;
                                                        goto tree_1;
                                                }
                                        }
                                }
                        }
                tree_1:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_1;
                                } else {
                                        goto break_1_1;
                                }
                        }
                        if (img_row[c] > 0) {
                        NODE_13:
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                goto tree_11;
                                        } else {
                                                if (img_row_prev[c - 1] > 0) {
                                                NODE_16:
                                                        if (img_row_prev_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                goto tree_11;
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c - 2],
                                                                              img_labels_row_prev_prev[c]);
                                                                goto tree_11;
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto tree_11;
                                                }
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev[c] > 0) {
                                                                goto NODE_5;
                                                        } else {
                                                                if (img_row_prev[c - 1] > 0) {
                                                                NODE_21:
                                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                                                if (img_row_prev_prev[c] > 0) {
                                                                                        img_labels_row[c] =
                                                                                            img_labels_row_prev_prev[c +
                                                                                                                     2];
                                                                                        goto tree_5;
                                                                                } else {
                                                                                        img_labels_row[c] = set_union(
                                                                                            P_,
                                                                                            img_labels_row_prev_prev[c -
                                                                                                                     2],
                                                                                            img_labels_row_prev_prev
                                                                                                [c + 2]);
                                                                                        goto tree_5;
                                                                                }
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c - 2],
                                                                                    img_labels_row_prev_prev[c + 2]);
                                                                                goto tree_5;
                                                                        }
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            img_labels_row_prev_prev[c + 2];
                                                                        goto tree_5;
                                                                }
                                                        }
                                                } else {
                                                        if (img_row_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                goto tree_4;
                                                        } else {
                                                                if (img_row_prev[c - 1] > 0) {
                                                                        img_labels_row[c] =
                                                                            img_labels_row_prev_prev[c - 2];
                                                                        goto tree_3;
                                                                } else {
                                                                        img_labels_row[c] = label;
                                                                        P_[label]         = label;
                                                                        ((void)0);
                                                                        label = label + 1;
                                                                        goto tree_3;
                                                                }
                                                        }
                                                }
                                        } else {
                                                if (img_row_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto tree_10;
                                                } else {
                                                        if (img_row_prev[c - 1] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                                                goto tree_9;
                                                        } else {
                                                                img_labels_row[c] = label;
                                                                P_[label]         = label;
                                                                ((void)0);
                                                                label = label + 1;
                                                                goto tree_9;
                                                        }
                                                }
                                        }
                                }
                        } else {
                                goto NODE_8;
                        }
                tree_2:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_2;
                                } else {
                                        goto break_1_2;
                                }
                        }
                        if (img_row[c] > 0) {
                        NODE_27:
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                goto tree_11;
                                        } else {
                                                if (img_row_prev[c - 1] > 0) {
                                                        if (img_row_prev_prev[c] > 0) {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c],
                                                                              img_labels_row[c - 2]);
                                                                goto tree_11;
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              set_union(P_,
                                                                                        img_labels_row_prev_prev[c - 2],
                                                                                        img_labels_row_prev_prev[c]),
                                                                              img_labels_row[c - 2]);
                                                                goto tree_11;
                                                        }
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        goto tree_11;
                                                }
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev[c] > 0) {
                                                                if (img_row_prev_prev[c + 1] > 0) {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c + 2],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_5;
                                                                } else {
                                                                        img_labels_row[c] = set_union(
                                                                            P_,
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row_prev_prev[c + 2]),
                                                                            img_labels_row[c - 2]);
                                                                        goto tree_5;
                                                                }
                                                        } else {
                                                                if (img_row_prev[c - 1] > 0) {
                                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                                                if (img_row_prev_prev[c] > 0) {
                                                                                        img_labels_row[c] = set_union(
                                                                                            P_,
                                                                                            img_labels_row_prev_prev[c +
                                                                                                                     2],
                                                                                            img_labels_row[c - 2]);
                                                                                        goto tree_5;
                                                                                } else {
                                                                                        img_labels_row[c] = set_union(
                                                                                            P_,
                                                                                            set_union(
                                                                                                P_,
                                                                                                img_labels_row_prev_prev
                                                                                                    [c - 2],
                                                                                                img_labels_row_prev_prev
                                                                                                    [c + 2]),
                                                                                            img_labels_row[c - 2]);
                                                                                        goto tree_5;
                                                                                }
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    set_union(
                                                                                        P_,
                                                                                        img_labels_row_prev_prev[c - 2],
                                                                                        img_labels_row_prev_prev[c +
                                                                                                                 2]),
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_5;
                                                                        }
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c + 2],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_5;
                                                                }
                                                        }
                                                } else {
                                                        if (img_row_prev[c - 1] > 0) {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c - 2],
                                                                              img_labels_row[c - 2]);
                                                                goto tree_8;
                                                        } else {
                                                        NODE_39:
                                                                if (img_row_prev[c] > 0) {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_4;
                                                                } else {
                                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                                        goto tree_3;
                                                                }
                                                        }
                                                }
                                        } else {
                                                if (img_row_prev[c - 1] > 0) {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]);
                                                        goto tree_12;
                                                } else {
                                                NODE_41:
                                                        if (img_row_prev[c] > 0) {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c],
                                                                              img_labels_row[c - 2]);
                                                                goto tree_10;
                                                        } else {
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                goto tree_9;
                                                        }
                                                }
                                        }
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                        NODE_44:
                                                if (img_row_prev[c + 1] > 0) {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        goto tree_6;
                                                } else {
                                                NODE_45:
                                                        if (img_row_prev[c + 2] > 0) {
                                                                if (img_row_prev_prev[c + 1] > 0) {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c + 2],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_5;
                                                                } else {
                                                                        if (img_row_prev[c] > 0) {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    set_union(
                                                                                        P_,
                                                                                        img_labels_row_prev_prev[c],
                                                                                        img_labels_row_prev_prev[c +
                                                                                                                 2]),
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_5;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_5;
                                                                        }
                                                                }
                                                        } else {
                                                                goto NODE_39;
                                                        }
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_7;
                                        }
                                } else {
                                        goto NODE_11;
                                }
                        }
                tree_3:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_3;
                                } else {
                                        goto break_1_3;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] =
                                            set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                        goto tree_11;
                                } else {
                                        if (img_row[c + 1] > 0) {
                                        NODE_50:
                                                if (img_row_prev[c + 2] > 0) {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                        goto tree_5;
                                                } else {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto tree_8;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_12;
                                        }
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 1] > 0) {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        goto tree_6;
                                                } else {
                                                        goto NODE_50;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_7;
                                        }
                                } else {
                                NODE_54:
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 1] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto tree_6;
                                                } else {
                                                        if (img_row_prev[c + 2] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                                goto tree_5;
                                                        } else {
                                                                img_labels_row[c] = label;
                                                                P_[label]         = label;
                                                                ((void)0);
                                                                label = label + 1;
                                                                goto tree_3;
                                                        }
                                                }
                                        } else {
                                                goto NODE_12;
                                        }
                                }
                        }
                tree_4:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_3;
                                } else {
                                        goto break_1_4;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev_prev[c] > 0) {
                                        NODE_59:
                                                if (img_row_prev_prev[c - 1] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto tree_11;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        goto tree_11;
                                                }
                                        } else {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                goto tree_11;
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                        NODE_61:
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                                if (img_row_prev_prev[c] > 0) {
                                                                NODE_64:
                                                                        if (img_row_prev_prev[c - 1] > 0) {
                                                                                img_labels_row[c] =
                                                                                    img_labels_row_prev_prev[c + 2];
                                                                                goto tree_5;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_5;
                                                                        }
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c + 2],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_5;
                                                                }
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c + 2],
                                                                              img_labels_row[c - 2]);
                                                                goto tree_5;
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto tree_8;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_12;
                                        }
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 1] > 0) {
                                                        if (img_row_prev_prev[c] > 0) {
                                                        NODE_69:
                                                                if (img_row_prev_prev[c - 1] > 0) {
                                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                        goto tree_6;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_6;
                                                                }
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c],
                                                                              img_labels_row[c - 2]);
                                                                goto tree_6;
                                                        }
                                                } else {
                                                        goto NODE_61;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_7;
                                        }
                                } else {
                                        goto NODE_54;
                                }
                        }
                tree_5:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_3;
                                } else {
                                        goto break_1_5;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        goto tree_11;
                                } else {
                                        if (img_row[c + 1] > 0) {
                                        NODE_72:
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                                goto tree_5;
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c + 2],
                                                                              img_labels_row[c - 2]);
                                                                goto tree_5;
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto tree_8;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_12;
                                        }
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 1] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto tree_6;
                                                } else {
                                                        goto NODE_72;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_7;
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 1] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto tree_6;
                                                } else {
                                                        if (img_row_prev[c + 2] > 0) {
                                                                goto NODE_5;
                                                        } else {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                goto tree_4;
                                                        }
                                                }
                                        } else {
                                                goto NODE_12;
                                        }
                                }
                        }
                tree_6:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_3;
                                } else {
                                        goto break_1_6;
                                }
                        }
                        if (img_row[c] > 0) {
                        NODE_80:
                                if (img_row_prev[c + 1] > 0) {
                                NODE_81:
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                goto tree_11;
                                        } else {
                                                if (img_row_prev_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto tree_11;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        goto tree_11;
                                                }
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                        NODE_84:
                                                if (img_row_prev[c + 2] > 0) {
                                                NODE_85:
                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                        NODE_86:
                                                                if (img_row_prev[c] > 0) {
                                                                        img_labels_row[c] =
                                                                            img_labels_row_prev_prev[c + 2];
                                                                        goto tree_5;
                                                                } else {
                                                                        if (img_row_prev_prev[c] > 0) {
                                                                                img_labels_row[c] =
                                                                                    img_labels_row_prev_prev[c + 2];
                                                                                goto tree_5;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_5;
                                                                        }
                                                                }
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c + 2],
                                                                              img_labels_row[c - 2]);
                                                                goto tree_5;
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto tree_8;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_12;
                                        }
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                        NODE_90:
                                                if (img_row_prev[c + 1] > 0) {
                                                NODE_91:
                                                        if (img_row_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                goto tree_6;
                                                        } else {
                                                                if (img_row_prev_prev[c] > 0) {
                                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                        goto tree_6;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_6;
                                                                }
                                                        }
                                                } else {
                                                        goto NODE_84;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_7;
                                        }
                                } else {
                                        goto NODE_11;
                                }
                        }
                tree_7:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_4;
                                } else {
                                        goto break_1_7;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                        goto NODE_27;
                                } else {
                                        goto NODE_13;
                                }
                        } else {
                        NODE_94:
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_fol[c - 1] > 0) {
                                                        goto NODE_44;
                                                } else {
                                                        goto NODE_10;
                                                }
                                        } else {
                                        NODE_97:
                                                if (img_row_fol[c - 1] > 0) {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto tree_7;
                                                } else {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                        goto tree_7;
                                                }
                                        }
                                } else {
                                        goto NODE_11;
                                }
                        }
                tree_8:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_3;
                                } else {
                                        goto break_1_8;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev_prev[c] > 0) {
                                                if (img_row_prev[c - 2] > 0) {
                                                        goto NODE_59;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        goto tree_11;
                                                }
                                        } else {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                goto tree_11;
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                        NODE_102:
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                                if (img_row_prev_prev[c] > 0) {
                                                                        if (img_row_prev[c - 2] > 0) {
                                                                                goto NODE_64;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_5;
                                                                        }
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c + 2],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_5;
                                                                }
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c + 2],
                                                                              img_labels_row[c - 2]);
                                                                goto tree_5;
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto tree_8;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_12;
                                        }
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 1] > 0) {
                                                        if (img_row_prev_prev[c] > 0) {
                                                                if (img_row_prev[c - 2] > 0) {
                                                                        goto NODE_69;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_6;
                                                                }
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c],
                                                                              img_labels_row[c - 2]);
                                                                goto tree_6;
                                                        }
                                                } else {
                                                        goto NODE_102;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto tree_7;
                                        }
                                } else {
                                        goto NODE_54;
                                }
                        }
                tree_9:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_5;
                                } else {
                                        goto break_1_9;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                        if (img_row_prev[c + 1] > 0) {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                goto tree_11;
                                        } else {
                                                if (img_row[c + 1] > 0) {
                                                        goto NODE_45;
                                                } else {
                                                        goto NODE_41;
                                                }
                                        }
                                } else {
                                        goto NODE_1;
                                }
                        } else {
                                goto NODE_94;
                        }
                tree_10:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_6;
                                } else {
                                        goto break_1_10;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                        if (img_row_prev[c + 1] > 0) {
                                        NODE_116:
                                                if (img_row_prev_prev[c - 1] > 0) {
                                                        goto NODE_81;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        goto tree_11;
                                                }
                                        } else {
                                                if (img_row[c + 1] > 0) {
                                                NODE_118:
                                                        if (img_row_prev[c + 2] > 0) {
                                                                if (img_row_prev_prev[c + 1] > 0) {
                                                                NODE_120:
                                                                        if (img_row_prev_prev[c - 1] > 0) {
                                                                                goto NODE_86;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_5;
                                                                        }
                                                                } else {
                                                                        if (img_row_prev[c] > 0) {
                                                                        NODE_122:
                                                                                if (img_row_prev_prev[c - 1] > 0) {
                                                                                        img_labels_row[c] = set_union(
                                                                                            P_,
                                                                                            img_labels_row_prev_prev[c +
                                                                                                                     2],
                                                                                            img_labels_row[c - 2]);
                                                                                        goto tree_5;
                                                                                } else {
                                                                                        img_labels_row[c] = set_union(
                                                                                            P_,
                                                                                            set_union(
                                                                                                P_,
                                                                                                img_labels_row_prev_prev
                                                                                                    [c],
                                                                                                img_labels_row_prev_prev
                                                                                                    [c + 2]),
                                                                                            img_labels_row[c - 2]);
                                                                                        goto tree_5;
                                                                                }
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_5;
                                                                        }
                                                                }
                                                        } else {
                                                                if (img_row_prev[c] > 0) {
                                                                NODE_124:
                                                                        if (img_row_prev_prev[c - 1] > 0) {
                                                                                img_labels_row[c] =
                                                                                    img_labels_row_prev_prev[c];
                                                                                goto tree_4;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c],
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_4;
                                                                        }
                                                                } else {
                                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                                        goto tree_3;
                                                                }
                                                        }
                                                } else {
                                                        if (img_row_prev[c] > 0) {
                                                        NODE_126:
                                                                if (img_row_prev_prev[c - 1] > 0) {
                                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                        goto tree_10;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_10;
                                                                }
                                                        } else {
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                goto tree_9;
                                                        }
                                                }
                                        }
                                } else {
                                        goto NODE_1;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_fol[c - 1] > 0) {
                                                        if (img_row_prev[c + 1] > 0) {
                                                        NODE_131:
                                                                if (img_row_prev_prev[c - 1] > 0) {
                                                                        goto NODE_91;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_6;
                                                                }
                                                        } else {
                                                                goto NODE_118;
                                                        }
                                                } else {
                                                        goto NODE_10;
                                                }
                                        } else {
                                                goto NODE_97;
                                        }
                                } else {
                                        goto NODE_11;
                                }
                        }
                tree_11:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_7;
                                } else {
                                        goto break_1_11;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row[c - 1] > 0) {
                                        goto NODE_80;
                                } else {
                                        if (img_row_fol[c - 1] > 0) {
                                                goto NODE_80;
                                        } else {
                                                if (img_row_prev[c + 1] > 0) {
                                                        if (img_row_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                goto tree_11;
                                                        } else {
                                                                goto NODE_16;
                                                        }
                                                } else {
                                                        if (img_row[c + 1] > 0) {
                                                                if (img_row_prev[c + 2] > 0) {
                                                                        if (img_row_prev[c] > 0) {
                                                                                goto NODE_5;
                                                                        } else {
                                                                                goto NODE_21;
                                                                        }
                                                                } else {
                                                                        if (img_row_prev[c] > 0) {
                                                                                img_labels_row[c] =
                                                                                    img_labels_row_prev_prev[c];
                                                                                goto tree_4;
                                                                        } else {
                                                                                img_labels_row[c] =
                                                                                    img_labels_row_prev_prev[c - 2];
                                                                                goto tree_3;
                                                                        }
                                                                }
                                                        } else {
                                                                if (img_row_prev[c] > 0) {
                                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                        goto tree_10;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            img_labels_row_prev_prev[c - 2];
                                                                        goto tree_9;
                                                                }
                                                        }
                                                }
                                        }
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row[c - 1] > 0) {
                                                        goto NODE_90;
                                                } else {
                                                        if (img_row_fol[c - 1] > 0) {
                                                                if (img_row_prev[c + 1] > 0) {
                                                                        goto NODE_91;
                                                                } else {
                                                                        if (img_row_prev[c + 2] > 0) {
                                                                                goto NODE_85;
                                                                        } else {
                                                                                if (img_row_prev[c] > 0) {
                                                                                        img_labels_row[c] =
                                                                                            img_labels_row_prev_prev[c];
                                                                                        goto tree_4;
                                                                                } else {
                                                                                        img_labels_row[c] =
                                                                                            img_labels_row[c - 2];
                                                                                        goto tree_3;
                                                                                }
                                                                        }
                                                                }
                                                        } else {
                                                                goto NODE_10;
                                                        }
                                                }
                                        } else {
                                                if (img_row_fol[c - 1] > 0) {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto tree_7;
                                                } else {
                                                        if (img_row[c - 1] > 0) {
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                goto tree_7;
                                                        } else {
                                                                img_labels_row[c] = label;
                                                                P_[label]         = label;
                                                                ((void)0);
                                                                label = label + 1;
                                                                goto tree_7;
                                                        }
                                                }
                                        }
                                } else {
                                        goto NODE_11;
                                }
                        }
                tree_12:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto break_0_8;
                                } else {
                                        goto break_1_12;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                        if (img_row_prev[c + 1] > 0) {
                                                if (img_row_prev[c - 2] > 0) {
                                                        goto NODE_116;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        goto tree_11;
                                                }
                                        } else {
                                                if (img_row[c + 1] > 0) {
                                                NODE_154:
                                                        if (img_row_prev[c + 2] > 0) {
                                                                if (img_row_prev_prev[c + 1] > 0) {
                                                                        if (img_row_prev[c - 2] > 0) {
                                                                                goto NODE_120;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_5;
                                                                        }
                                                                } else {
                                                                        if (img_row_prev[c] > 0) {
                                                                                if (img_row_prev[c - 2] > 0) {
                                                                                        goto NODE_122;
                                                                                } else {
                                                                                        img_labels_row[c] = set_union(
                                                                                            P_,
                                                                                            set_union(
                                                                                                P_,
                                                                                                img_labels_row_prev_prev
                                                                                                    [c],
                                                                                                img_labels_row_prev_prev
                                                                                                    [c + 2]),
                                                                                            img_labels_row[c - 2]);
                                                                                        goto tree_5;
                                                                                }
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_5;
                                                                        }
                                                                }
                                                        } else {
                                                                if (img_row_prev[c] > 0) {
                                                                        if (img_row_prev[c - 2] > 0) {
                                                                                goto NODE_124;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c],
                                                                                    img_labels_row[c - 2]);
                                                                                goto tree_4;
                                                                        }
                                                                } else {
                                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                                        goto tree_3;
                                                                }
                                                        }
                                                } else {
                                                        if (img_row_prev[c] > 0) {
                                                                if (img_row_prev[c - 2] > 0) {
                                                                        goto NODE_126;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_10;
                                                                }
                                                        } else {
                                                                img_labels_row[c] = img_labels_row[c - 2];
                                                                goto tree_9;
                                                        }
                                                }
                                        }
                                } else {
                                        goto NODE_1;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_fol[c - 1] > 0) {
                                                        if (img_row_prev[c + 1] > 0) {
                                                                if (img_row_prev[c - 2] > 0) {
                                                                        goto NODE_131;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row[c - 2]);
                                                                        goto tree_6;
                                                                }
                                                        } else {
                                                                goto NODE_154;
                                                        }
                                                } else {
                                                        goto NODE_10;
                                                }
                                        } else {
                                                goto NODE_97;
                                        }
                                } else {
                                        goto NODE_11;
                                }
                        }
                break_0_0:
                        if (img_row[c] > 0) {
                        NODE_168:
                                if (img_row_prev[c] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                } else {
                                        img_labels_row[c] = label;
                                        P_[label]         = label;
                                        ((void)0);
                                        label = label + 1;
                                }
                        } else {
                        NODE_169:
                                if (img_row_fol[c] > 0) {
                                        img_labels_row[c] = label;
                                        P_[label]         = label;
                                        ((void)0);
                                        label = label + 1;
                                } else {
                                        img_labels_row[c] = 0;
                                }
                        }
                        continue;
                break_0_1:
                        if (img_row[c] > 0) {
                        NODE_170:
                                if (img_row_prev[c] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                } else {
                                        if (img_row_prev[c - 1] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                        } else {
                                                img_labels_row[c] = label;
                                                P_[label]         = label;
                                                ((void)0);
                                                label = label + 1;
                                        }
                                }
                        } else {
                                goto NODE_169;
                        }
                        continue;
                break_0_2:
                        if (img_row[c] > 0) {
                        NODE_172:
                                if (img_row_prev[c - 1] > 0) {
                                        img_labels_row[c] =
                                            set_union(P_, img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]);
                                } else {
                                NODE_173:
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        }
                                }
                        } else {
                        NODE_174:
                                if (img_row_fol[c] > 0) {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                } else {
                                        img_labels_row[c] = 0;
                                }
                        }
                        continue;
                break_0_3:
                        if (img_row[c] > 0) {
                                img_labels_row[c] = img_labels_row[c - 2];
                        } else {
                                goto NODE_174;
                        }
                        continue;
                break_0_4:
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                        goto NODE_172;
                                } else {
                                        goto NODE_170;
                                }
                        } else {
                        NODE_176:
                                if (img_row_fol[c] > 0) {
                                NODE_177:
                                        if (img_row_fol[c - 1] > 0) {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        } else {
                                                img_labels_row[c] = label;
                                                P_[label]         = label;
                                                ((void)0);
                                                label = label + 1;
                                        }
                                } else {
                                        img_labels_row[c] = 0;
                                }
                        }
                        continue;
                break_0_5:
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                        goto NODE_173;
                                } else {
                                        goto NODE_168;
                                }
                        } else {
                                goto NODE_176;
                        }
                        continue;
                break_0_6:
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                NODE_180:
                                        if (img_row_prev[c] > 0) {
                                        NODE_181:
                                                if (img_row_prev_prev[c - 1] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        }
                                } else {
                                        goto NODE_168;
                                }
                        } else {
                                goto NODE_176;
                        }
                        continue;
                break_0_7:
                        if (img_row[c] > 0) {
                                if (img_row[c - 1] > 0) {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                } else {
                                        if (img_row_fol[c - 1] > 0) {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        } else {
                                        NODE_184:
                                                if (img_row_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                } else {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                                }
                                        }
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                NODE_186:
                                        if (img_row_fol[c - 1] > 0) {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        } else {
                                                if (img_row[c - 1] > 0) {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                } else {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                }
                                        }
                                } else {
                                        img_labels_row[c] = 0;
                                }
                        }
                        continue;
                break_0_8:
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                NODE_189:
                                        if (img_row_prev[c] > 0) {
                                        NODE_190:
                                                if (img_row_prev[c - 2] > 0) {
                                                        goto NODE_181;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        }
                                } else {
                                        goto NODE_168;
                                }
                        } else {
                                goto NODE_176;
                        }
                        continue;
                break_1_0:
                        if (img_row[c] > 0) {
                        NODE_191:
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                } else {
                                        goto NODE_168;
                                }
                        } else {
                        NODE_192:
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_191;
                                        } else {
                                                img_labels_row[c] = label;
                                                P_[label]         = label;
                                                ((void)0);
                                                label = label + 1;
                                        }
                                } else {
                                NODE_194:
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_191;
                                        } else {
                                        NODE_195:
                                                if (img_row_fol[c + 1] > 0) {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                } else {
                                                        img_labels_row[c] = 0;
                                                }
                                        }
                                }
                        }
                        continue;
                break_1_1:
                        if (img_row[c] > 0) {
                        NODE_196:
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                        } else {
                                                if (img_row_prev[c - 1] > 0) {
                                                NODE_199:
                                                        if (img_row_prev_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c - 2],
                                                                              img_labels_row_prev_prev[c]);
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                }
                                        }
                                } else {
                                        goto NODE_170;
                                }
                        } else {
                                goto NODE_192;
                        }
                        continue;
                break_1_2:
                        if (img_row[c] > 0) {
                        NODE_200:
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                        } else {
                                                if (img_row_prev[c - 1] > 0) {
                                                        if (img_row_prev_prev[c] > 0) {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c],
                                                                              img_labels_row[c - 2]);
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              set_union(P_,
                                                                                        img_labels_row_prev_prev[c - 2],
                                                                                        img_labels_row_prev_prev[c]),
                                                                              img_labels_row[c - 2]);
                                                        }
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                }
                                        }
                                } else {
                                        goto NODE_172;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                        NODE_206:
                                                if (img_row_prev[c + 1] > 0) {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                } else {
                                                        goto NODE_173;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        }
                                } else {
                                        goto NODE_194;
                                }
                        }
                        continue;
                break_1_3:
                        if (img_row[c] > 0) {
                        NODE_207:
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] =
                                            set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_207;
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        }
                                } else {
                                NODE_210:
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 1] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                } else {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                }
                                        } else {
                                                goto NODE_195;
                                        }
                                }
                        }
                        continue;
                break_1_4:
                        if (img_row[c] > 0) {
                        NODE_212:
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev_prev[c] > 0) {
                                                goto NODE_181;
                                        } else {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                        }
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_212;
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        }
                                } else {
                                        goto NODE_210;
                                }
                        }
                        continue;
                break_1_5:
                        if (img_row[c] > 0) {
                        NODE_216:
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_216;
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                        } else {
                                                goto NODE_195;
                                        }
                                }
                        }
                        continue;
                break_1_6:
                        if (img_row[c] > 0) {
                        NODE_220:
                                if (img_row_prev[c + 1] > 0) {
                                NODE_221:
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                        } else {
                                                if (img_row_prev_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                }
                                        }
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_220;
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        }
                                } else {
                                        goto NODE_194;
                                }
                        }
                        continue;
                break_1_7:
                        if (img_row[c] > 0) {
                                if (img_row_fol[c - 1] > 0) {
                                        goto NODE_200;
                                } else {
                                        goto NODE_196;
                                }
                        } else {
                        NODE_226:
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                        NODE_228:
                                                if (img_row_fol[c - 1] > 0) {
                                                        goto NODE_206;
                                                } else {
                                                        goto NODE_191;
                                                }
                                        } else {
                                                goto NODE_177;
                                        }
                                } else {
                                        goto NODE_194;
                                }
                        }
                        continue;
                break_1_8:
                        if (img_row[c] > 0) {
                        NODE_229:
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev_prev[c] > 0) {
                                                goto NODE_190;
                                        } else {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                        }
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_229;
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                        }
                                } else {
                                        goto NODE_210;
                                }
                        }
                        continue;
                break_1_9:
                        if (img_row[c] > 0) {
                                goto NODE_228;
                        } else {
                                goto NODE_226;
                        }
                        continue;
                break_1_10:
                        if (img_row[c] > 0) {
                        NODE_233:
                                if (img_row_fol[c - 1] > 0) {
                                        if (img_row_prev[c + 1] > 0) {
                                        NODE_235:
                                                if (img_row_prev_prev[c - 1] > 0) {
                                                        goto NODE_221;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                }
                                        } else {
                                                goto NODE_180;
                                        }
                                } else {
                                        goto NODE_191;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_233;
                                        } else {
                                                goto NODE_177;
                                        }
                                } else {
                                        goto NODE_194;
                                }
                        }
                        continue;
                break_1_11:
                        if (img_row[c] > 0) {
                                if (img_row[c - 1] > 0) {
                                        goto NODE_220;
                                } else {
                                        if (img_row_fol[c - 1] > 0) {
                                                goto NODE_220;
                                        } else {
                                                if (img_row_prev[c + 1] > 0) {
                                                        if (img_row_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        } else {
                                                                goto NODE_199;
                                                        }
                                                } else {
                                                        goto NODE_184;
                                                }
                                        }
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row[c - 1] > 0) {
                                                        goto NODE_220;
                                                } else {
                                                        if (img_row_fol[c - 1] > 0) {
                                                                if (img_row_prev[c + 1] > 0) {
                                                                        goto NODE_221;
                                                                } else {
                                                                        if (img_row_prev[c] > 0) {
                                                                                img_labels_row[c] =
                                                                                    img_labels_row_prev_prev[c];
                                                                        } else {
                                                                                img_labels_row[c] =
                                                                                    img_labels_row[c - 2];
                                                                        }
                                                                }
                                                        } else {
                                                                goto NODE_191;
                                                        }
                                                }
                                        } else {
                                                goto NODE_186;
                                        }
                                } else {
                                        goto NODE_194;
                                }
                        }
                        continue;
                break_1_12:
                        if (img_row[c] > 0) {
                        NODE_248:
                                if (img_row_fol[c - 1] > 0) {
                                        if (img_row_prev[c + 1] > 0) {
                                                if (img_row_prev[c - 2] > 0) {
                                                        goto NODE_235;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                }
                                        } else {
                                                goto NODE_189;
                                        }
                                } else {
                                        goto NODE_191;
                                }
                        } else {
                                if (img_row_fol[c] > 0) {
                                        if (img_row[c + 1] > 0) {
                                                goto NODE_248;
                                        } else {
                                                goto NODE_177;
                                        }
                                } else {
                                        goto NODE_194;
                                }
                        }
                        continue;
                }
                if (oh & 1) {
                        const int            r                        = oh - 1;
                        const uint8_t* const img_row                  = img_pad + r * w_pad + 2;
                        const uint8_t* const img_row_prev             = img_row - w_pad;
                        const uint8_t* const img_row_prev_prev        = img_row_prev - w_pad;
                        int* const           img_labels_row           = labels_pad + r * w_pad + 2;
                        int* const           img_labels_row_prev_prev = img_labels_row - 2 * w_pad;
                        int                  c                        = -2;
                ll_tree_0:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto ll_break_0_0;
                                } else {
                                        goto ll_break_1_0;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        goto ll_tree_6;
                                } else {
                                        if (img_row[c + 1] > 0) {
                                        NODE_277:
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev[c] > 0) {
                                                        NODE_279:
                                                                if (img_row_prev_prev[c + 1] > 0) {
                                                                        img_labels_row[c] =
                                                                            img_labels_row_prev_prev[c + 2];
                                                                        goto ll_tree_4;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c],
                                                                                      img_labels_row_prev_prev[c + 2]);
                                                                        goto ll_tree_4;
                                                                }
                                                        } else {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                                goto ll_tree_4;
                                                        }
                                                } else {
                                                        if (img_row_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                goto ll_tree_3;
                                                        } else {
                                                                img_labels_row[c] = label;
                                                                P_[label]         = label;
                                                                ((void)0);
                                                                label = label + 1;
                                                                goto ll_tree_2;
                                                        }
                                                }
                                        } else {
                                                if (img_row_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto ll_tree_0;
                                                } else {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                        goto ll_tree_0;
                                                }
                                        }
                                }
                        } else {
                        NODE_282:
                                if (img_row[c + 1] > 0) {
                                        if (img_row_prev[c + 1] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                goto ll_tree_5;
                                        } else {
                                                goto NODE_277;
                                        }
                                } else {
                                        img_labels_row[c] = 0;
                                        goto ll_tree_1;
                                }
                        }
                ll_tree_1:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto ll_break_0_1;
                                } else {
                                        goto ll_break_1_1;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                goto ll_tree_6;
                                        } else {
                                                if (img_row_prev[c - 1] > 0) {
                                                NODE_287:
                                                        if (img_row_prev_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                goto ll_tree_6;
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c - 2],
                                                                              img_labels_row_prev_prev[c]);
                                                                goto ll_tree_6;
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto ll_tree_6;
                                                }
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev[c] > 0) {
                                                                goto NODE_279;
                                                        } else {
                                                                if (img_row_prev[c - 1] > 0) {
                                                                NODE_292:
                                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                                                if (img_row_prev_prev[c] > 0) {
                                                                                        img_labels_row[c] =
                                                                                            img_labels_row_prev_prev[c +
                                                                                                                     2];
                                                                                        goto ll_tree_4;
                                                                                } else {
                                                                                        img_labels_row[c] = set_union(
                                                                                            P_,
                                                                                            img_labels_row_prev_prev[c -
                                                                                                                     2],
                                                                                            img_labels_row_prev_prev
                                                                                                [c + 2]);
                                                                                        goto ll_tree_4;
                                                                                }
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c - 2],
                                                                                    img_labels_row_prev_prev[c + 2]);
                                                                                goto ll_tree_4;
                                                                        }
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            img_labels_row_prev_prev[c + 2];
                                                                        goto ll_tree_4;
                                                                }
                                                        }
                                                } else {
                                                        if (img_row_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                goto ll_tree_3;
                                                        } else {
                                                                if (img_row_prev[c - 1] > 0) {
                                                                        img_labels_row[c] =
                                                                            img_labels_row_prev_prev[c - 2];
                                                                        goto ll_tree_2;
                                                                } else {
                                                                        img_labels_row[c] = label;
                                                                        P_[label]         = label;
                                                                        ((void)0);
                                                                        label = label + 1;
                                                                        goto ll_tree_2;
                                                                }
                                                        }
                                                }
                                        } else {
                                                if (img_row_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto ll_tree_0;
                                                } else {
                                                        if (img_row_prev[c - 1] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                                                goto ll_tree_0;
                                                        } else {
                                                                img_labels_row[c] = label;
                                                                P_[label]         = label;
                                                                ((void)0);
                                                                label = label + 1;
                                                                goto ll_tree_0;
                                                        }
                                                }
                                        }
                                }
                        } else {
                                goto NODE_282;
                        }
                ll_tree_2:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto ll_break_0_2;
                                } else {
                                        goto ll_break_1_2;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] =
                                            set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                        goto ll_tree_6;
                                } else {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 2] > 0) {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
                                                        goto ll_tree_4;
                                                } else {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto ll_tree_7;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto ll_tree_0;
                                        }
                                }
                        } else {
                        NODE_301:
                                if (img_row[c + 1] > 0) {
                                        if (img_row_prev[c + 1] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                goto ll_tree_5;
                                        } else {
                                                if (img_row_prev[c + 2] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                        goto ll_tree_4;
                                                } else {
                                                        img_labels_row[c] = label;
                                                        P_[label]         = label;
                                                        ((void)0);
                                                        label = label + 1;
                                                        goto ll_tree_2;
                                                }
                                        }
                                } else {
                                        img_labels_row[c] = 0;
                                        goto ll_tree_1;
                                }
                        }
                ll_tree_3:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto ll_break_0_2;
                                } else {
                                        goto ll_break_1_3;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev_prev[c] > 0) {
                                        NODE_306:
                                                if (img_row_prev_prev[c - 1] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto ll_tree_6;
                                                } else {
                                                        img_labels_row[c] = set_union(P_,
                                                                                      img_labels_row_prev_prev[c - 2],
                                                                                      img_labels_row_prev_prev[c]);
                                                        goto ll_tree_6;
                                                }
                                        } else {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                goto ll_tree_6;
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                                if (img_row_prev_prev[c] > 0) {
                                                                NODE_311:
                                                                        if (img_row_prev_prev[c - 1] > 0) {
                                                                                img_labels_row[c] =
                                                                                    img_labels_row_prev_prev[c + 2];
                                                                                goto ll_tree_4;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto ll_tree_4;
                                                                        }
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c + 2],
                                                                                      img_labels_row[c - 2]);
                                                                        goto ll_tree_4;
                                                                }
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c + 2],
                                                                              img_labels_row[c - 2]);
                                                                goto ll_tree_4;
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto ll_tree_7;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto ll_tree_0;
                                        }
                                }
                        } else {
                                goto NODE_301;
                        }
                ll_tree_4:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto ll_break_0_2;
                                } else {
                                        goto ll_break_1_4;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                        goto ll_tree_6;
                                } else {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c + 2];
                                                                goto ll_tree_4;
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c + 2],
                                                                              img_labels_row[c - 2]);
                                                                goto ll_tree_4;
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto ll_tree_7;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto ll_tree_0;
                                        }
                                }
                        } else {
                                if (img_row[c + 1] > 0) {
                                        if (img_row_prev[c + 1] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                goto ll_tree_5;
                                        } else {
                                                if (img_row_prev[c + 2] > 0) {
                                                        goto NODE_279;
                                                } else {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto ll_tree_3;
                                                }
                                        }
                                } else {
                                        img_labels_row[c] = 0;
                                        goto ll_tree_1;
                                }
                        }
                ll_tree_5:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto ll_break_0_2;
                                } else {
                                        goto ll_break_1_5;
                                }
                        }
                        if (img_row[c] > 0) {
                        NODE_319:
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                goto ll_tree_6;
                                        } else {
                                                if (img_row_prev_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto ll_tree_6;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        goto ll_tree_6;
                                                }
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                                if (img_row_prev[c] > 0) {
                                                                        img_labels_row[c] =
                                                                            img_labels_row_prev_prev[c + 2];
                                                                        goto ll_tree_4;
                                                                } else {
                                                                        if (img_row_prev_prev[c] > 0) {
                                                                                img_labels_row[c] =
                                                                                    img_labels_row_prev_prev[c + 2];
                                                                                goto ll_tree_4;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto ll_tree_4;
                                                                        }
                                                                }
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c + 2],
                                                                              img_labels_row[c - 2]);
                                                                goto ll_tree_4;
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto ll_tree_7;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto ll_tree_0;
                                        }
                                }
                        } else {
                                goto NODE_282;
                        }
                ll_tree_6:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto ll_break_0_3;
                                } else {
                                        goto ll_break_1_6;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row[c - 1] > 0) {
                                        goto NODE_319;
                                } else {
                                        if (img_row_prev[c + 1] > 0) {
                                                if (img_row_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        goto ll_tree_6;
                                                } else {
                                                        goto NODE_287;
                                                }
                                        } else {
                                                if (img_row[c + 1] > 0) {
                                                        if (img_row_prev[c + 2] > 0) {
                                                                if (img_row_prev[c] > 0) {
                                                                        goto NODE_279;
                                                                } else {
                                                                        goto NODE_292;
                                                                }
                                                        } else {
                                                                if (img_row_prev[c] > 0) {
                                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                        goto ll_tree_3;
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            img_labels_row_prev_prev[c - 2];
                                                                        goto ll_tree_2;
                                                                }
                                                        }
                                                } else {
                                                        if (img_row_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                                goto ll_tree_0;
                                                        } else {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                                                goto ll_tree_0;
                                                        }
                                                }
                                        }
                                }
                        } else {
                                goto NODE_282;
                        }
                ll_tree_7:
                        if ((c += 2) >= w - 2) {
                                if (c > w - 2) {
                                        goto ll_break_0_2;
                                } else {
                                        goto ll_break_1_7;
                                }
                        }
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev_prev[c] > 0) {
                                                if (img_row_prev[c - 2] > 0) {
                                                        goto NODE_306;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                        goto ll_tree_6;
                                                }
                                        } else {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                goto ll_tree_6;
                                        }
                                } else {
                                        if (img_row[c + 1] > 0) {
                                                if (img_row_prev[c + 2] > 0) {
                                                        if (img_row_prev_prev[c + 1] > 0) {
                                                                if (img_row_prev_prev[c] > 0) {
                                                                        if (img_row_prev[c - 2] > 0) {
                                                                                goto NODE_311;
                                                                        } else {
                                                                                img_labels_row[c] = set_union(
                                                                                    P_,
                                                                                    img_labels_row_prev_prev[c + 2],
                                                                                    img_labels_row[c - 2]);
                                                                                goto ll_tree_4;
                                                                        }
                                                                } else {
                                                                        img_labels_row[c] =
                                                                            set_union(P_,
                                                                                      img_labels_row_prev_prev[c + 2],
                                                                                      img_labels_row[c - 2]);
                                                                        goto ll_tree_4;
                                                                }
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c + 2],
                                                                              img_labels_row[c - 2]);
                                                                goto ll_tree_4;
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row[c - 2];
                                                        goto ll_tree_7;
                                                }
                                        } else {
                                                img_labels_row[c] = img_labels_row[c - 2];
                                                goto ll_tree_0;
                                        }
                                }
                        } else {
                                goto NODE_301;
                        }
                ll_break_0_0:
                        if (img_row[c] > 0) {
                        NODE_343:
                                if (img_row_prev[c] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                } else {
                                        img_labels_row[c] = label;
                                        P_[label]         = label;
                                        ((void)0);
                                        label = label + 1;
                                }
                        } else {
                                img_labels_row[c] = 0;
                        }
                        goto ll_end;
                ll_break_0_1:
                        if (img_row[c] > 0) {
                        NODE_344:
                                if (img_row_prev[c] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                } else {
                                        if (img_row_prev[c - 1] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                        } else {
                                                img_labels_row[c] = label;
                                                P_[label]         = label;
                                                ((void)0);
                                                label = label + 1;
                                        }
                                }
                        } else {
                                img_labels_row[c] = 0;
                        }
                        goto ll_end;
                ll_break_0_2:
                        if (img_row[c] > 0) {
                                img_labels_row[c] = img_labels_row[c - 2];
                        } else {
                                img_labels_row[c] = 0;
                        }
                        goto ll_end;
                ll_break_0_3:
                        if (img_row[c] > 0) {
                                if (img_row[c - 1] > 0) {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                } else {
                                NODE_347:
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                        } else {
                                                img_labels_row[c] = img_labels_row_prev_prev[c - 2];
                                        }
                                }
                        } else {
                                img_labels_row[c] = 0;
                        }
                        goto ll_end;
                ll_break_1_0:
                        if (img_row[c] > 0) {
                        NODE_348:
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                } else {
                                        goto NODE_343;
                                }
                        } else {
                        NODE_349:
                                if (img_row[c + 1] > 0) {
                                        goto NODE_348;
                                } else {
                                        img_labels_row[c] = 0;
                                }
                        }
                        goto ll_end;
                ll_break_1_1:
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                        } else {
                                                if (img_row_prev[c - 1] > 0) {
                                                NODE_353:
                                                        if (img_row_prev_prev[c] > 0) {
                                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                                        } else {
                                                                img_labels_row[c] =
                                                                    set_union(P_,
                                                                              img_labels_row_prev_prev[c - 2],
                                                                              img_labels_row_prev_prev[c]);
                                                        }
                                                } else {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                }
                                        }
                                } else {
                                        goto NODE_344;
                                }
                        } else {
                                goto NODE_349;
                        }
                        goto ll_end;
                ll_break_1_2:
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] =
                                            set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                        NODE_355:
                                if (img_row[c + 1] > 0) {
                                        if (img_row_prev[c + 1] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                        } else {
                                                img_labels_row[c] = label;
                                                P_[label]         = label;
                                                ((void)0);
                                                label = label + 1;
                                        }
                                } else {
                                        img_labels_row[c] = 0;
                                }
                        }
                        goto ll_end;
                ll_break_1_3:
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev_prev[c] > 0) {
                                        NODE_359:
                                                if (img_row_prev_prev[c - 1] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                } else {
                                                        img_labels_row[c] = set_union(P_,
                                                                                      img_labels_row_prev_prev[c - 2],
                                                                                      img_labels_row_prev_prev[c]);
                                                }
                                        } else {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                        }
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                                goto NODE_355;
                        }
                        goto ll_end;
                ll_break_1_4:
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                                if (img_row[c + 1] > 0) {
                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                } else {
                                        img_labels_row[c] = 0;
                                }
                        }
                        goto ll_end;
                ll_break_1_5:
                        if (img_row[c] > 0) {
                        NODE_362:
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev[c] > 0) {
                                                img_labels_row[c] = img_labels_row_prev_prev[c];
                                        } else {
                                                if (img_row_prev_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                }
                                        }
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                                goto NODE_349;
                        }
                        goto ll_end;
                ll_break_1_6:
                        if (img_row[c] > 0) {
                                if (img_row[c - 1] > 0) {
                                        goto NODE_362;
                                } else {
                                        if (img_row_prev[c + 1] > 0) {
                                                if (img_row_prev[c] > 0) {
                                                        img_labels_row[c] = img_labels_row_prev_prev[c];
                                                } else {
                                                        goto NODE_353;
                                                }
                                        } else {
                                                goto NODE_347;
                                        }
                                }
                        } else {
                                goto NODE_349;
                        }
                        goto ll_end;
                ll_break_1_7:
                        if (img_row[c] > 0) {
                                if (img_row_prev[c + 1] > 0) {
                                        if (img_row_prev_prev[c] > 0) {
                                                if (img_row_prev[c - 2] > 0) {
                                                        goto NODE_359;
                                                } else {
                                                        img_labels_row[c] = set_union(
                                                            P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                                }
                                        } else {
                                                img_labels_row[c] =
                                                    set_union(P_, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
                                        }
                                } else {
                                        img_labels_row[c] = img_labels_row[c - 2];
                                }
                        } else {
                                goto NODE_355;
                        }
                        goto ll_end;
                ll_end:;
                }
        }
        int k = 1;
        flattenLParallel(P_, firstLabel, label - firstLabel, &k);
        const int nLabels = k;
        for (int r = 0; r < oh; r += 2) {
                for (int c = 0; c < ow; c += 2) {
                        int anchor = labels_pad[r * w_pad + 2 + c];
                        int root   = (anchor > 0) ? P_[anchor] : 0;
                        if (img_pad[r * w_pad + 2 + c]) {
                                labels_out[r * ow + c] = root;
                        } else {
                                labels_out[r * ow + c] = 0;
                        }
                        if (c + 1 < ow) {
                                if (img_pad[r * w_pad + 2 + c + 1]) {
                                        labels_out[r * ow + c + 1] = root;
                                } else {
                                        labels_out[r * ow + c + 1] = 0;
                                }
                        }
                        if (r + 1 < oh) {
                                if (img_pad[(r + 1) * w_pad + 2 + c]) {
                                        labels_out[(r + 1) * ow + c] = root;
                                } else {
                                        labels_out[(r + 1) * ow + c] = 0;
                                }
                        }
                        if (r + 1 < oh && c + 1 < ow) {
                                if (img_pad[(r + 1) * w_pad + 2 + c + 1]) {
                                        labels_out[(r + 1) * ow + c + 1] = root;
                                } else {
                                        labels_out[(r + 1) * ow + c + 1] = 0;
                                }
                        }
                }
        }
        free(P_);
        free(img_pad);
        free(labels_pad);
        return nLabels;
}

static int cmp_area_desc(const void* a, const void* b) {
        const CDCircle* ca = (const CDCircle*)a;
        const CDCircle* cb = (const CDCircle*)b;
        if (ca->area < cb->area) return 1;
        if (ca->area > cb->area) return -1;
        return 0;
}

int detectCircles(const CDConfig* cfg,
                  CDCircle*       out,
                  int             out_cap,
                  uint8_t*        mask,
                  uint8_t*        tmp1,
                  uint8_t*        tmp2,
                  int*            labels,
                  int*            num_components_out) {
        if (!cfg || !out || out_cap <= 0 || !mask || !tmp1 || !tmp2 || !labels) return 0;
        const int width  = cfg->width;
        const int height = cfg->height;
        if (width <= 0 || height <= 0 || (width & 1) || (height & 1)) return 0;

        make_color_mask_i420_full(
            cfg->y, cfg->u, cfg->v, width, height, cfg->target_u, cfg->target_v, cfg->uv_tol, cfg->y_min, mask);

        morph_open_close_3x3(mask, width, height, tmp1, tmp2);

        const int num_components = spaghetti8_label(mask, width, height, labels);
        if (num_components_out) *num_components_out = num_components;
        if (num_components <= 1) return 0;

        BoxStats* stats = (BoxStats*)malloc(num_components * sizeof(BoxStats));
        if (!stats) return 0;

        for (int i = 0; i < num_components; ++i) {
                stats[i].minx = width;
                stats[i].miny = height;
                stats[i].maxx = -1;
                stats[i].maxy = -1;
                stats[i].area = 0;
                stats[i].sumx = 0;
                stats[i].sumy = 0;
                stats[i].seen = 0;
        }


        for (int y = 0; y < height; ++y) {
                const int* restrict row = labels + y * width;
                for (int x = 0; x < width; ++x) {
                        const int lbl = row[x];
                        if (lbl <= 0 || lbl >= num_components) continue;
                        BoxStats* s = &stats[lbl];
                        s->seen     = 1;
                        if (x < s->minx) s->minx = x;
                        if (y < s->miny) s->miny = y;
                        if (x > s->maxx) s->maxx = x;
                        if (y > s->maxy) s->maxy = y;
                        s->area++;
                        s->sumx += (uint64_t)x;
                        s->sumy += (uint64_t)y;
                }
        }

        const double min_area = M_PI * (0.5 * cfg->min_d) * (0.5 * cfg->min_d);
        const double max_area = M_PI * (0.5 * cfg->max_d) * (0.5 * cfg->max_d);

        int          found    = 0;
        for (int lab = 1; lab < num_components; ++lab) {
                const BoxStats* s = &stats[lab];
                if (!s->seen) continue;
                if (s->area < 4) continue;
                const int bb_w = s->maxx - s->minx + 1;
                const int bb_h = s->maxy - s->miny + 1;
                if (bb_w < 2 || bb_h < 2) continue;
                const double aspect = (double)(bb_w < bb_h ? bb_w : bb_h) / (double)(bb_w > bb_h ? bb_w : bb_h);
                if (aspect < cfg->aspect_min) continue;
                const double extent = (double)s->area / (double)(bb_w * bb_h);
                if (extent < cfg->extent_min) continue;
                const double area_full = (double)s->area;
                if (area_full < min_area || area_full > max_area) continue;
                const double cx_s = (double)s->sumx / (double)s->area;
                const double cy_s = (double)s->sumy / (double)s->area;
                CDCircle     c;
                c.cx   = (float)(cx_s);
                c.cy   = (float)(cy_s);
                c.area = area_full;
                c.r    = (float)sqrt(area_full / M_PI);
                if (found < out_cap && found < cfg->max_out) {
                        out[found] = c;
                        ++found;
                }
        }

        if (found > 1) {
                qsort(out, (size_t)found, sizeof(CDCircle), cmp_area_desc);
        }
        free(stats);
        return found;
}
