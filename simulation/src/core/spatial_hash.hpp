#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <functional>

// Open-addressing spatial hash for O(1) amortized neighbor queries.
// CELL_SIZE: size of each grid cell in world units (metres).
// Designed for ~4096 objects; resize BUCKET_COUNT if needed.
template<typename T, int CELL_SIZE = 20>
class SpatialHash {
public:
    static constexpr int BUCKET_COUNT = 8192;  // power of two for fast modulo

    struct Entry {
        T*       obj  = nullptr;
        uint64_t key  = 0;       // cell key (to handle collisions)
        bool     used = false;
    };

    SpatialHash() { buckets_.fill({}); }

    void clear() { buckets_.fill({}); }

    void insert(T* obj, float x, float y) {
        uint64_t key = cell_key(x, y);
        uint32_t idx = key & (BUCKET_COUNT - 1);
        for (int i = 0; i < BUCKET_COUNT; ++i) {
            Entry& e = buckets_[(idx + i) & (BUCKET_COUNT - 1)];
            if (!e.used) {
                e = {obj, key, true};
                return;
            }
        }
    }

    void remove(T* obj) {
        for (auto& e : buckets_) {
            if (e.used && e.obj == obj) {
                e.used = false;
                return;
            }
        }
    }

    void update(T* obj, float old_x, float old_y, float new_x, float new_y) {
        uint64_t old_key = cell_key(old_x, old_y);
        uint64_t new_key = cell_key(new_x, new_y);
        if (old_key == new_key) return;
        remove(obj);
        insert(obj, new_x, new_y);
    }

    // Collect all objects within radius r of (cx, cy)
    void query_radius(float cx, float cy, float r, std::vector<T*>& out) const {
        int x0 = cell_coord(cx - r);
        int x1 = cell_coord(cx + r);
        int y0 = cell_coord(cy - r);
        int y1 = cell_coord(cy + r);
        float r2 = r * r;

        for (int gx = x0; gx <= x1; ++gx) {
            for (int gy = y0; gy <= y1; ++gy) {
                uint64_t key = make_key(gx, gy);
                uint32_t idx = key & (BUCKET_COUNT - 1);
                for (int i = 0; i < 16; ++i) {
                    const Entry& e = buckets_[(idx + i) & (BUCKET_COUNT - 1)];
                    if (!e.used) break;
                    if (e.key == key) {
                        T* obj = e.obj;
                        float dx = obj->x - cx;
                        float dy = obj->y - cy;
                        if (dx * dx + dy * dy <= r2) out.push_back(obj);
                    }
                }
            }
        }
    }

private:
    std::array<Entry, BUCKET_COUNT> buckets_;

    static int cell_coord(float v) {
        return static_cast<int>(std::floor(v / CELL_SIZE));
    }

    static uint64_t make_key(int gx, int gy) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(gx)) * 73856093ULL)
             ^ (static_cast<uint64_t>(static_cast<uint32_t>(gy)) * 19349663ULL);
    }

    static uint64_t cell_key(float x, float y) {
        return make_key(cell_coord(x), cell_coord(y));
    }
};
