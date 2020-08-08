#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <cstring>
#include <utility>

#include <Godot.hpp>
#include <Reference.hpp>
#include <PoolArrays.hpp>

using namespace godot;

/// All methods are optimised for maximum speed. They take arrays and assume the 
/// right dimension for them. If the inputs aren't right they'll crash. Third input
/// is used for the answer, preallocated. There are no conditional branchings. 
/// Just use the method appropriate to the situation. The names are coded to reflect
/// that. s = scalar, v = vector and m = matrix. So for example

/// dot_vm 

/// is a dot product between vector and matrix (in that order). Wherever the in_place
/// argument is provided, it is possible to perform the operation on the object
/// itself instead of instantiating a new one (this too optimises performance).
struct LinAlg : public Reference {
    GODOT_CLASS(LinAlg, Reference);

    static void _register_methods();

    /// Inititalise a vector
    static PoolRealArray init_v(int n, float v0) {
        PoolRealArray ans;
        ans.resize(n);
        real_t *ans_write_ptr = ans.write().ptr();
        std::memset(ans_write_ptr, v0, (n * sizeof(real_t)));
        // for (int i = 0; i < n; ++i) {
        //     ans_write_ptr[i] = v0;
        // }

        return ans;
    }

    /// Initialise a matrix
    static PoolRealArray init_m(int m, int n, float m0) {
        PoolRealArray ans;
        ans.resize(m * n);
        real_t *ans_write_ptr = ans.write().ptr();

        // Initialise as contiguous space
        std::memset(ans_write_ptr, m0, (m * n * sizeof(real_t)));
        // for (int i = 0; i < m * n; ++i) {
        //     ans_write_ptr[i] = m0;
        // }

        return ans;
    }

    /// Identity matrix
    static PoolRealArray eye(const int n) {
        PoolRealArray ans;
        ans.resize(n * n);
        real_t *ans_write_ptr = ans.write().ptr();

        // Use row-major notation for accessing index
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // look ma, no branches
                ans_write_ptr[n * i + j] = real_t(i == j);
            }
        }

        return ans;
    }

    /// Diagonal matrix
    /// TODO consider making this sparse
    static PoolRealArray diag(const PoolRealArray &v) {
        int n = v.size();
        PoolRealArray ans;
        ans.resize(n * n);
        real_t *ans_write_ptr = ans.write().ptr();

        const real_t *v_read_ptr = v.read().ptr();
        for (int i = 0; i < n; ++i) {
            // can't use v[i] as that operator expects a const int
            real_t vi = v_read_ptr[i];
            for (int j = 0; j < n; ++j) {
                ans_write_ptr[n * i + j] = real_t((i == j) * vi);
            }
        }

        return ans;
    }

    /// Dyadic matrix
    static PoolRealArray dyadic(const PoolRealArray &v) {
        int n = v.size();
        PoolRealArray ans;
        ans.resize(n * n);
        real_t *ans_write_ptr = ans.write().ptr();

        const real_t *v_read_ptr = v.read().ptr();
        for (int i = 0; i < n; ++i) {
            real_t vi = v_read_ptr[i];
            PoolRealArray row(v);
            real_t *row_write_ptr = row.write().ptr();
            for (int j = 0; j < n; ++j) {
                // can't use v[i] as that operator doesn't return a reference
                row_write_ptr[j] *= vi;
            }
            ans_write_ptr[n * i] = row_write_ptr[i];
        }

        return ans;
    }

    /// Transpose in-place
    static void transpose(const PoolRealArray &M, int n) {

    }
};

#endif