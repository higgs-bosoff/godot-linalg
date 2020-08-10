#ifndef LIN_ALG_HPP
#define LIN_ALG_HPP

#include <Dictionary.hpp>
#include <Godot.hpp>
#include <Node.hpp>
#include <PoolArrays.hpp>

using namespace godot;

/// All methods exchange `PoolRealArrays` or `Dictionaries` and operate on `real_t`
/// (synonymous with `float` in most cases).
/// Conditional branches are avoided where possible.
///
/// The names are coded to reflect the type of data used: s = scalar, v = vector and m = matrix. So
///
///     dot_mv(M, v)
///
/// is a dot product between a matrix and a vector (in that order).
/// In-place function variants do not instantiate new `PoolRealArray`s (but may resize them).
///
/// Matrices are represented as Dictionaries of `PoolRealArray`, `real_t`, `real_t`:
/// a two-dimensional matrix in one-dimensional row-major order and its dimensions, m and n.
///
/// This is ***NOT*** a replacement for proper BLAS libraries and routines.
/// **Do not use this in production code.**
/// There is no sparse matrix implementation, and none of these functions work in parallel
/// or use special vector instructions.
/// Consider supporting the author in writing a wrapper for your favourite Linear Algebra library!
class LinAlg : public Node {
	GODOT_CLASS(LinAlg, Node);

public:
	static void _register_methods();

	/// Read/write a matrix element
	real_t &m_ij(const Dictionary M, int i, int j, bool column_major = false, bool check = true);

	/// Inititalise a vector
	PoolRealArray init_v(int n, real_t v0 = real_t(0));

	/// Initialise a matrix
	Dictionary init_m(int m, int n, real_t m0 = real_t(0));

	/// Identity matrix
	Dictionary eye(int n);

	/// Diagonal matrix
	/// TODO consider making this sparse
	Dictionary diag(const PoolRealArray v);

	/// Dyadic matrix
	Dictionary dyadic(const PoolRealArray v);

	/// Transpose in-place, needs matrix row size
	void transpose_in_place(Dictionary M, bool check = true);

	/// Transpose, needs matrix row size
	Dictionary transpose(const Dictionary M, bool check = true);

	/// Householder matrix from vector
	/// (https://en.wikipedia.org/wiki/Householder_transformation)
	Dictionary householder(const PoolRealArray v);

	/// Random vector
	PoolRealArray rand_v(int n, real_t s = real_t(1));

	/// Random matrix
	Dictionary rand_m(int m, int n, real_t s = real_t(1));

	/// TODO negation operator, boolean operators

	/// Element-wise: vector plus scalar in-place
	void ewise_vs_add_in_place(PoolRealArray v, real_t s);

	/// Element-wise: vector plus scalar
	PoolRealArray ewise_vs_add(const PoolRealArray v, real_t s);

	/// Element-wise: vector times scalar in-place
	void ewise_vs_mul_in_place(PoolRealArray v, real_t s);

	/// Element-wise: vector times scalar
	PoolRealArray ewise_vs_mul(const PoolRealArray v, real_t s);

	/// Element-wise: vector plus vector in-place
	void ewise_vv_add_in_place(PoolRealArray v1, const PoolRealArray v2);

	/// Element-wise: vector plus vector
	PoolRealArray ewise_vv_add(const PoolRealArray v1, const PoolRealArray v2);

	/// Element-wise: vector times vector in-place
	void ewise_vv_mul_in_place(PoolRealArray v1, const PoolRealArray v2);

	/// Element-wise: vector times vector
	PoolRealArray ewise_vv_mul(const PoolRealArray v1, const PoolRealArray v2);

	/// Element-wise: matrix plus scalar in-place
	void ewise_ms_add_in_place(Dictionary M, real_t s, bool check = true);

	/// Element-wise: matrix plus scalar
	Dictionary ewise_ms_add(const Dictionary M, real_t s, bool check = true);

	/// Element-wise: matrix times scalar in-place
	void ewise_ms_mul_in_place(Dictionary M, real_t s, bool check = true);

	/// Element-wise: matrix times scalar
	Dictionary ewise_ms_mul(const Dictionary M, real_t s, bool check = true);

	/// Element-wise: matrix plus matrix in-place
	void ewise_mm_add_in_place(Dictionary M1, const Dictionary M2, bool check = true);

	/// Element-wise: matrix plus matrix
	Dictionary ewise_mm_add(const Dictionary M1, const Dictionary M2, bool check = true);

	/// Element-wise: matrix times matrix in-place
	void ewise_mm_mul_in_place(Dictionary M1, const Dictionary M2, bool check = true);

	/// Element-wise: matrix times matrix
	Dictionary ewise_mm_mul(const Dictionary M1, const Dictionary M2, bool check = true);

	/// Norm^2 of vector
	real_t norm2_v(const PoolRealArray v);

	/// Norm of vector
	real_t norm_v(const PoolRealArray v);

	/// Normalise in-place
	void normalize_in_place(PoolRealArray v);

	/// Normalise
	PoolRealArray normalize(const PoolRealArray v);

	/// Dot product: vector times vector
	real_t dot_vv(const PoolRealArray v1, const PoolRealArray v2);

	/// Dot product: matrix times vector
	PoolRealArray dot_mv(const Dictionary M, const PoolRealArray v, bool check = true);

	/// Dot product: matrix times matrix
	Dictionary dot_mm(const Dictionary M1, const Dictionary M2, bool check = true);

	/// QR decomposition
	/// returns {Q: m, R: m}
	Dictionary qr(const Dictionary M, bool check = true);

	/// Eigenvalues by power iteration for symmetric matrices, in-place
	/// returns {evals: array of s, evecs: array of v}
	Dictionary eigs_powerit_in_place(const Dictionary M, real_t tolerance = real_t(1e-5f), bool check = true);

	/// Eigenvalues by power iteration for symmetric matrices, in-place
	/// returns {evals: array of s, evecs: array of v}
	Dictionary eigs_powerit(const Dictionary M, real_t tolerance = real_t(1e-5f), bool check = true);

	/// Eigenvalues by QR decomposition
	/// Dictionary eigs_qr(const Dictionary M, real_t tol = real_t(1e-8f));
};

#endif // LIN_ALG_HPP
