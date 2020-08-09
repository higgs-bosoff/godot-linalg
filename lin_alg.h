#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <Dictionary.hpp>
#include <Godot.hpp>
#include <PoolArrays.hpp>
#include <Reference.hpp>

using namespace godot;

/// All methods exchange PoolRealArrays or Dictionaries.
/// Conditional branches are avoided where possible.
/// The names are coded to reflect the type of data used: s = scalar, v = vector and m = matrix. So
///
/// dot_mv
///
/// is a dot product between matrix and vector (in that order).
/// In-place function variants do not instantiate new PoolRealArrays (but may resize them).
///
/// Matrices are represented as Dictionaries of PoolRealArray, real_t, real_t.
///
/// This is *NOT* a replacement for proper BLAS libraries and routines.
struct LinAlg : public Reference {
	GODOT_CLASS(LinAlg, Reference);

	static void _register_methods();

	/// Read/write a matrix element
	static real_t &m_ij(const Dictionary &M, int i, int j, bool column_major = false, bool check = true);

	/// Inititalise a vector
	static PoolRealArray init_v(int n, real_t v0 = real_t(0));

	/// Initialise a matrix
	static Dictionary init_m(int m, int n, real_t m0 = real_t(0));

	/// Identity matrix
	static Dictionary eye(int n);

	/// Diagonal matrix
	/// TODO consider making this sparse
	static Dictionary diag(const PoolRealArray &v);

	/// Dyadic matrix
	static Dictionary dyadic(const PoolRealArray &v);

	/// Transpose in-place, needs matrix row size
	static void transpose_in_place(Dictionary &M, bool check = true);

	/// Transpose, needs matrix row size
	static Dictionary transpose(const Dictionary &M, bool check = true);

	/// Householder matrix from vector
	/// (https://en.wikipedia.org/wiki/Householder_transformation)
	static Dictionary householder(const PoolRealArray &v);

	/// Random vector
	static PoolRealArray rand_v(int n, real_t s = real_t(1));

	/// Random matrix
	static Dictionary rand_m(int m, int n, real_t s = real_t(1));

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
	static void ewise_ms_add_in_place(Dictionary &M, real_t s, bool check = true);

	/// Element-wise: matrix plus scalar
	static Dictionary ewise_ms_add(const Dictionary &M, real_t s, bool check = true);

	/// Element-wise: matrix times scalar in-place
	static void ewise_ms_mul_in_place(Dictionary &M, real_t s, bool check = true);

	/// Element-wise: matrix times scalar
	static Dictionary ewise_ms_mul(const Dictionary &M, real_t s, bool check = true);

	/// Element-wise: matrix plus matrix in-place
	static void ewise_mm_add_in_place(Dictionary &M1, const Dictionary &M2, bool check = true);

	/// Element-wise: matrix plus matrix
	static Dictionary ewise_mm_add(const Dictionary &M1, const Dictionary &M2, bool check = true);

	/// Element-wise: matrix times matrix in-place
	static void ewise_mm_mul_in_place(Dictionary &M1, const Dictionary &M2, bool check = true);

	/// Element-wise: matrix times matrix
	static Dictionary ewise_mm_mul(const Dictionary &M1, const Dictionary &M2, bool check = true);

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

	/// Dot product: matrix times vector
	static PoolRealArray dot_mv(const Dictionary &M, const PoolRealArray &v, bool check = true);

	/// Dot product: matrix times matrix
	static Dictionary dot_mm(const Dictionary &M1, const Dictionary &M2, bool check = true);

	/// QR decomposition
	/// returns {Q: m, R: m}
	static Dictionary qr(const Dictionary &M, bool check = true);

	/// Eigenvalues by power iteration for symmetric matrices
	/// returns {evals: v, evecs: m}
	static Dictionary eigs_powerit(const Dictionary &M, real_t tolerance = 1e-5, bool check = true);
};

#endif
