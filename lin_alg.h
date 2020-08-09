#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <Godot.hpp>
#include <PoolArrays.hpp>
#include <Reference.hpp>

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
	static PoolRealArray init_v(int n, real_t v0 = real_t(0));

	/// Initialise a matrix
	static PoolRealArray init_m(int m, int n, real_t m0 = real_t(0));

	/// Identity matrix
	static PoolRealArray eye(const int n);

	/// Diagonal matrix
	/// TODO consider making this sparse
	static PoolRealArray diag(const PoolRealArray &v);

	/// Dyadic matrix
	static PoolRealArray dyadic(const PoolRealArray &v);

	/// Transpose in-place, needs matrix row size
	static void transpose_in_place(PoolRealArray &M, int n);

	/// Transpose, needs matrix row size
	static PoolRealArray transpose(const PoolRealArray &M, int n);

	/// Householder matrix from vector
	/// (https://en.wikipedia.org/wiki/Householder_transformation)
	static PoolRealArray householder(const PoolRealArray &v);

	/// Random vector
	static PoolRealArray rand_v(int n, real_t s = real_t(1));

	/// Random matrix
	static PoolRealArray rand_m(int m, int n, real_t s = real_t(1));

	/// Element-wise: vector plus scalar in-place
	static void ewise_vs_add_in_place(PoolRealArray &v, real_t s);

	/// Element-wise: vector plus scalar
	static PoolRealArray ewise_vs_add(const PoolRealArray &v, real_t s);

	/// Element-wise: vector times scalar in-place
	static void ewise_vs_mul_in_place(PoolRealArray &v, real_t s);

	/// Element-wise: vector times scalar
	static PoolRealArray ewise_vs_mul(const PoolRealArray &v, real_t s);

	/// Element-wise: vector plus vector in-place
	static void ewise_vv_add_in_place(PoolRealArray &v1, const PoolRealArray &v2);

	/// Element-wise: vector plus vector
	static PoolRealArray ewise_vv_add(const PoolRealArray &v1, const PoolRealArray &v2);

	/// Element-wise: vector times vector in-place
	static void ewise_vv_mul_in_place(PoolRealArray &v1, const PoolRealArray &v2);

	/// Element-wise: vector times vector
	static PoolRealArray ewise_vv_mul(const PoolRealArray &v1, const PoolRealArray &v2);

	/// Element-wise: matrix plus scalar in-place
	static void ewise_ms_add_in_place(PoolRealArray &M, real_t s);

	/// Element-wise: matrix plus scalar
	static PoolRealArray ewise_ms_add(const PoolRealArray &M, real_t s);

	/// Element-wise: matrix times scalar in-place
	static void ewise_ms_mul_in_place(PoolRealArray &M, real_t s);

	/// Element-wise: matrix times scalar
	static PoolRealArray ewise_ms_mul(const PoolRealArray &M, real_t s);

	/// Element-wise: matrix plus matrix in-place
	static void ewise_mm_add_in_place(PoolRealArray &M1, int n1, const PoolRealArray &M2, int n2);

	/// Element-wise: matrix plus matrix
	static PoolRealArray ewise_mm_add(const PoolRealArray &M1, int n1, const PoolRealArray &M2, int n2);

	/// Element-wise: matrix times matrix in-place
	static void ewise_mm_mul_in_place(PoolRealArray &M1, int n1, const PoolRealArray &M2, int n2);

	/// Element-wise: matrix times matrix
	static PoolRealArray ewise_mm_mul(const PoolRealArray &M1, int n1, const PoolRealArray &M2, int n2);

	/// Norm^2 of vector
	static real_t norm2_v(const PoolRealArray &v);

	/// Norm of vector
	static real_t norm_v(const PoolRealArray &v);

	/// Normalise in-place
	static void normalize_in_place(PoolRealArray &v);

	/// Normalise
	static PoolRealArray normalize(const PoolRealArray &v);

	/// Dot product: vector times vector
	static real_t dot_vv(const PoolRealArray &v1, const PoolRealArray &v2);

	/// Dot product: matrix times matrix
	static PoolRealArray dot_mm(const PoolRealArray &M1, int n1, const PoolRealArray &M2, int n2);
};

#endif
