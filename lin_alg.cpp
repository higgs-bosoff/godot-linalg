#include "lin_alg.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <Array.hpp>
#include <RandomNumberGenerator.hpp>

using namespace godot;

/*
Some design decisions
=====================
- Bypassing PoolRealArray::Read and PoolRealArray::Write
  While incredibly useful for manipulating the raw pointers with
  std:: functions, quite unreadable outside that use case, but
  perhaps fastest. #include "donald-knuth-quote.inc".

- Treating matrices as contiguous row-major PoolRealArrays.
  Many vector functions can simply be reused for matrices this way.

- Wrapping matrices in Dictionaries carrying both dimensions
  instead of just n, where m = M.size() / n

- Permitting custom key names on matrix-dictionaries for ease-of-use.
  Internally, the value list is extracted where needed.

- Adding a way to validate any dictionary that represents a matrix
  and making validation an integral but optional part of every 
  function that takes matrix-dictionaries as arguments.

- Macros for repetitive definitions such as the element-wise operations.

- Internal functions for
	- matrix-dictionary validation
	- wrapping PoolRealArray matrices in matrix-dictionaries
	- initialising vectors
	- initialising PoolRealArray matrices (just uses the vector version)
	- in-place transpose
	- stretching a vector to fit a larger vector
	- stretching a matrix to fit a larger matrix
	- matrix-matrix dot product (which works for matrix-vector)
  these allow matrices to be operated on in either form. 
*/

void LinAlg::_register_methods() {
	register_method("init_v", &LinAlg::init_v);
	register_method("init_m", &LinAlg::init_m);
	register_method("eye", &LinAlg::eye);
	register_method("diag", &LinAlg::diag);
	register_method("dyadic", &LinAlg::dyadic);
	register_method("transpose_in_place", &LinAlg::transpose_in_place);
	register_method("transpose", &LinAlg::transpose);
	register_method("householder", &LinAlg::householder);
	register_method("rand_v", &LinAlg::rand_v);
	register_method("rand_m", &LinAlg::rand_m);
	register_method("ewise_vs_add_in_place", &LinAlg::ewise_vs_add_in_place);
	register_method("ewise_vs_add", &LinAlg::ewise_vs_add);
	register_method("ewise_vs_mul_in_place", &LinAlg::ewise_vs_mul_in_place);
	register_method("ewise_vs_mul", &LinAlg::ewise_vs_mul);
	register_method("ewise_vv_add_in_place", &LinAlg::ewise_vv_add_in_place);
	register_method("ewise_vv_add", &LinAlg::ewise_vv_add);
	register_method("ewise_vv_mul_in_place", &LinAlg::ewise_vv_mul_in_place);
	register_method("ewise_vv_mul", &LinAlg::ewise_vv_mul);
	register_method("ewise_ms_add_in_place", &LinAlg::ewise_ms_add_in_place);
	register_method("ewise_ms_add", &LinAlg::ewise_ms_add);
	register_method("ewise_ms_mul_in_place", &LinAlg::ewise_ms_mul_in_place);
	register_method("ewise_ms_mul", &LinAlg::ewise_ms_mul);
	register_method("ewise_mm_add_in_place", &LinAlg::ewise_mm_add_in_place);
	register_method("ewise_mm_add", &LinAlg::ewise_mm_add);
	register_method("ewise_mm_mul_in_place", &LinAlg::ewise_mm_mul_in_place);
	register_method("ewise_mm_mul", &LinAlg::ewise_mm_mul);
	register_method("norm2_v", &LinAlg::norm2_v);
	register_method("norm_v", &LinAlg::norm_v);
	register_method("normalize_in_place", &LinAlg::normalize_in_place);
	register_method("normalize", &LinAlg::normalize);
	register_method("dot_vv", &LinAlg::dot_vv);
	register_method("dot_mv", &LinAlg::dot_mv);
	register_method("dot_mm", &LinAlg::dot_mm);
	register_method("qr", &LinAlg::qr);
	register_method("eigs_powerit_in_place", &LinAlg::eigs_powerit_in_place);
	register_method("eigs_powerit", &LinAlg::eigs_powerit);
	// register_method("eigs_qr", &LinAlg::eigs_qr);
}

const char *_not_m_msg = "This Dictionary isn't a matrix. Make sure that there are 3 values of type PoolRealArray, float, float.";

#define M_CHECK(M)                 \
	if (check) {                   \
		if (!_is_m((M))) {         \
			ERR_PRINT(_not_m_msg); \
			ERR_FAIL();            \
		}                          \
	}

#define M_CHECK_V(M, ret)          \
	if (check) {                   \
		if (!_is_m((M))) {         \
			ERR_PRINT(_not_m_msg); \
			ERR_FAIL_V((ret));     \
		}                          \
	}

inline bool _is_m(const Dictionary &M) {
	if (M.size() != 3) {
		return false;
	}
	const Array M_values = M.values();
	if (M_values[0].get_type() != Variant::POOL_REAL_ARRAY) {
		return false;
	} else if (M_values[1].get_type() != Variant::REAL) {
		return false;
	} else if (M_values[2].get_type() != Variant::REAL) {
		return false;
	} else if (((int)M_values[1] < 0) || ((int)M_values[2] < 0)) {
		return false;
	} else {
		return true;
	}
}

inline Dictionary _make_m(const PoolRealArray &M, int m, int n) {
	return Dictionary::make("M", M, "m", m, "n", n);
}

inline Dictionary _make_m() {
	return ::_make_m(PoolRealArray(), 0, 0);
}

// The most useful thing I've devised, a macro to do what tuples do in C++17!
#define expand_m(M)                                      \
	const Array M##_values = (M).values();               \
	PoolRealArray *_##M = &(PoolRealArray)M##_values[0]; \
	int m_##M = M##_values[1];                           \
	int n_##M = M##_values[2]

constexpr real_t _REAL_SIGNALING_NAN = std::numeric_limits<real_t>::signaling_NaN();

inline real_t &LinAlg::m_ij(const Dictionary &M, int i, int j, bool column_major = false, bool check = true) {
	M_CHECK_V(M, const_cast<real_t &>(_REAL_SIGNALING_NAN));

	expand_m(M);
	real_t *M_write_ptr = _M->write().ptr();
	return column_major ? M_write_ptr[m_M * j + i] : M_write_ptr[n_M * i + j];
}

inline PoolRealArray _init_v(int n, real_t v0 = real_t(0)) {
	PoolRealArray ans;
	ans.resize(n);
	real_t *ans_write_ptr = ans.write().ptr();
	std::memset(ans_write_ptr, v0, (n * sizeof(real_t)));
	// for (int i = 0; i < n; ++i) {
	//     ans_write_ptr[i] = v0;
	// }

	return ans;
}

inline PoolRealArray _init_m(int m, int n, real_t m0 = real_t(0)) {
	return _init_v(m * n, m0);
}

PoolRealArray LinAlg::init_v(int n, real_t v0 = real_t()) {
	return ::_init_v(n, v0);
}

Dictionary LinAlg::init_m(int m, int n, real_t m0 = real_t()) {
	return ::_make_m(::_init_m(m, n, m0), m, n);
}

Dictionary LinAlg::eye(int n) {
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

	return ::_make_m(ans, n, n);
}

Dictionary LinAlg::diag(const PoolRealArray &v) {
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

	return ::_make_m(ans, n, n);
}

Dictionary LinAlg::dyadic(const PoolRealArray &v) {
	int n = v.size();
	PoolRealArray ans;
	ans.resize(n * n);
	real_t *ans_write_ptr = ans.write().ptr();

	const real_t *v_read_ptr = v.read().ptr();
	for (int i = 0; i < n; ++i) {
		// can't use v[i] as v is passed as a const &
		real_t vi = v_read_ptr[i];
		PoolRealArray row(v);

		real_t *row_write_ptr = row.write().ptr();
		for (int j = 0; j < n; ++j) {
			row_write_ptr[j] *= vi;
		}
		ans_write_ptr[n * i] = row_write_ptr[i];
	}

	return ::_make_m(ans, n, n);
}

// Implementation for transforming non-square matrices from
// https://www.geeksforgeeks.org/inplace-m-x-n-size-matrix-transpose/
inline void transpose_in_place(PoolRealArray &M, int m, int n) {
	/*
		m   = 2   = 3
		n   = 3   = 2
		[1 2 3] [1 4] [1 2 3;4 5 6] [1 4;2 5;3 6]
		[4 5 6] [2 5]
				[3 6]
	*/
	real_t *M_write_ptr = M.write().ptr();
	int size = m * n - 1;
	// Use a vector<bool> instead of a bitset
	std::vector<bool> marked(m * n);

	marked[0] = marked[size] = true;

	int i = 1;
	while (i < size) {
		int cycle_start = i;
		real_t t = M_write_ptr[i];
		do {
			int next = (i * m) % size;
			std::swap(M_write_ptr[next], t);
			marked[i] = true;
			i = next;
		} while (i != cycle_start);

		for (i = 1; i < size && marked[i]; i++)
			;
	}
}

void LinAlg::transpose_in_place(Dictionary &M, bool check = true) {
	M_CHECK(M);

	const Array M_values = M.values();
	::transpose_in_place((PoolRealArray)M_values[0], M_values[1], M_values[2]);
}

Dictionary LinAlg::transpose(const Dictionary &M, bool check = true) {
	M_CHECK_V(M, Dictionary());

	const Array M_values = M.values();
	PoolRealArray ans((PoolRealArray)M_values[0]);
	::transpose_in_place(ans, M_values[1], M_values[2]);
	return _make_m(ans, M_values[1], M_values[2]);
}

Dictionary LinAlg::householder(const PoolRealArray &v) {
	int n = v.size();
	PoolRealArray ans;
	ans.resize(n * n);
	real_t *ans_write_ptr = ans.write().ptr();

	const real_t *v_read_ptr = v.read().ptr();
	for (int i = 0; i < n; ++i) {
		real_t vi = -v_read_ptr[i] * 2;
		PoolRealArray row(v);

		real_t *row_write_ptr = row.write().ptr();
		for (int j = 0; j < n; ++j) {
			row_write_ptr[j] *= vi;
			row_write_ptr[j] += real_t(i == j);
		}
		ans_write_ptr[n * i] = row_write_ptr[i];
	}

	return ::_make_m(ans, n, n);
}

PoolRealArray LinAlg::rand_v(int n, real_t s = real_t(1)) {
	PoolRealArray ans;
	ans.resize(n);
	real_t *ans_write_ptr = ans.write().ptr();

	Ref<RandomNumberGenerator> rand = RandomNumberGenerator::_new();
	for (int i = 0; i < n; ++i) {
		ans_write_ptr[i] = std::fmod(rand->randf(), s);
	}

	return ans;
}

Dictionary LinAlg::rand_m(int m, int n, real_t s = real_t(1)) {
	return ::_make_m(rand_v(m * n, s), m, n);
}

#define ewise_vs_op_in_place(op, __)                                    \
	void LinAlg::ewise_vs_##op##_in_place(PoolRealArray &v, real_t s) { \
		real_t *v_write_ptr = v.write().ptr();                          \
                                                                        \
		for (int i = 0; i < v.size(); ++i) {                            \
			v_write_ptr[i] __ s;                                        \
		}                                                               \
	}

#define ewise_vs_op(op)                                                       \
	PoolRealArray LinAlg::ewise_vs_##op##(const PoolRealArray &v, real_t s) { \
		PoolRealArray ans(v);                                                 \
		ewise_vs_##op##_in_place(ans, s);                                     \
		return ans;                                                           \
	}

ewise_vs_op_in_place(add, +=);
ewise_vs_op_in_place(mul, *=);

ewise_vs_op(add);
ewise_vs_op(mul);

inline void _stretch_to_fit_v(PoolRealArray &to_stretch, const PoolRealArray &to_fit) {
	if (to_stretch.size() < to_fit.size()) {
		to_stretch.resize(to_fit.size());
	}
}

// It is up to the user to avoid costly stretching by supplying a longer v1.
// The implementation must promise that only the first argument is modified.
// This is desirable if the user wants to make this vector larger for further manipulation.
#define ewise_vv_op_in_place(op, __)                                                    \
	void LinAlg::ewise_vv_##op##_in_place(PoolRealArray &v1, const PoolRealArray &v2) { \
		::_stretch_to_fit_v(v1, v2);                                                    \
                                                                                        \
		real_t *v1_write_ptr = v1.write().ptr();                                        \
		const real_t *v2_read_ptr = v2.read().ptr();                                    \
		/* ##op## all elements based on v2's length */                                  \
		for (int i = 0; i < v2.size(); ++i) {                                           \
			v1_write_ptr[i] __ v2_read_ptr[i];                                          \
		}                                                                               \
	}

// This needn't be the case if it's not in-place
#define compare_vv_and_ewise_vv_op(op)                                                      \
	PoolRealArray LinAlg::ewise_vv_##op(const PoolRealArray &v1, const PoolRealArray &v2) { \
		bool v1_gt_v2 = v1.size() > v2.size();                                              \
		const PoolRealArray *small = !v1_gt_v2 ? &v1 : &v2;                                 \
		const PoolRealArray *large = v1_gt_v2 ? &v1 : &v2;                                  \
                                                                                            \
		PoolRealArray ans(*large);                                                          \
		ewise_vv_##op##_in_place(ans, *small);                                              \
		return ans;                                                                         \
	}

ewise_vv_op_in_place(add, +=);
ewise_vv_op_in_place(mul, *=);

compare_vv_and_ewise_vv_op(add);
compare_vv_and_ewise_vv_op(mul);

// internals are alike, just reuse the functions

#define ewise_ms_op_in_place(op)                                                        \
	void LinAlg::ewise_ms_##op##_in_place(Dictionary &M, real_t s, bool check = true) { \
		M_CHECK(M);                                                                     \
		ewise_vs_##op##_in_place((PoolRealArray)M.values()[0], s);                      \
	}

#define ewise_ms_op(op)                                                                  \
	Dictionary LinAlg::ewise_ms_##op(const Dictionary &M, real_t s, bool check = true) { \
		M_CHECK_V(M, ::_make_m());                                                       \
                                                                                         \
		expand_m(M);                                                                     \
		PoolRealArray ans(*_M);                                                          \
		ewise_vs_##op##_in_place(ans, s);                                                \
		return ::_make_m(ans, m_M, n_M);                                                 \
	}

ewise_ms_op_in_place(add);
ewise_ms_op_in_place(mul);

ewise_ms_op(add);
ewise_ms_op(mul);

/*
Stretch to fit: matrix edition!
==============================

row-major:
    row increases: insert cells at the end of each stride
    column increases: append new cells
    both: insert then append

column-major:
    inverse.

problems:
	append > insert
	append_array > n * append
	
solutions:
	-  { insert } -> append_array(init_v) -> append (O(n^2))
		- many resizes
	++ new -> resize -> { v[i] = row[j] or 0 } (O(n))
		+ one resize
	+  { resize -> transpose } twice (O(n)) <==
		+ in-place
		+ resize twice
		+ transpose implemented, optional
		- not smart but functional

* is + or -.
                                           n
[1 2 3 -] [1 2 3 -;4 5 6 -;+ + + *] [1 4 +]1
[4 5 6 -] [1 4 +;2 5 +;3 6 +;- - *] [2 5 +]2
[+ + + *]                           [3 6 +]3
                                    [- - *]4
								   m 1 2 3 

*/

inline void _stretch_to_fit_m(PoolRealArray &to_stretch, int m1, int n1, const PoolRealArray &to_fit, int m2, int n2) {
	if (m1 > m2 || n1 > n2) {
		return;
	}
	// now n1 <= n2 and m1 <= n2

	// only stretch rows if possible
	if (n1 < n2 /* && m1 == m2*/) {
		// [1 2 3;4 5 6] -> [1 2 3;4 5 6;0 0 0]
		// to_stretch.resize(to_fit.size());
		to_stretch.append_array(_init_v(m2 * (n2 - n1)));
	}

	// stretch columns
	if (m1 < m2) {
		// [1 2 3;4 5 6] -> [1 4;2 5;3 6]
		::transpose_in_place(to_stretch, m1, n2);
		// [1 4;2 5;3 6] -> [1 4;2 5;3 6;0 0]
		// to_stretch.resize(to_fit.size());
		to_stretch.append_array(_init_v(n2 * (m2 - m1)));
		// [1 4;2 5;3 6;0 0] -> [1 2 3 0;4 5 6 0]
		::transpose_in_place(to_stretch, n2, m2);
	}
}

// It is up to the user to avoid costly stretching by supplying a longer v1.
// The implementation must promise that only the first argument is modified.
// This is desirable if the user wants to make this matrix larger for further manipulation.
// This is even more significant due to the added complexity of resizing a matrix.
#define ewise_mm_op_in_place(op, __)                                                                 \
	void LinAlg::ewise_mm_##op##_in_place(Dictionary &M1, const Dictionary &M2, bool check = true) { \
		M_CHECK(M1);                                                                                 \
		M_CHECK(M2);                                                                                 \
                                                                                                     \
		expand_m(M1);                                                                                \
		expand_m(M2);                                                                                \
		::_stretch_to_fit_m(*_M1, m_M1, n_M1, *_M2, m_M2, n_M2);                                     \
                                                                                                     \
		real_t *M1_write_ptr = _M1->write().ptr();                                                   \
		const real_t *M2_read_ptr = _M2->read().ptr();                                               \
		/* ##op## all elements based on M2's length */                                               \
		for (int i = 0; i < _M2->size(); ++i) {                                                      \
			M1_write_ptr[i] __ M2_read_ptr[i];                                                       \
		}                                                                                            \
	}

// This consideration can be overlooked by making sure that the larger matrix becomes the first arg.
#define compare_mm_and_ewise_mm_op(op)                                                                  \
	Dictionary LinAlg::ewise_mm_##op##(const Dictionary &M1, const Dictionary &M2, bool check = true) { \
		M_CHECK_V(M1, ::_make_m());                                                                     \
		M_CHECK_V(M2, ::_make_m());                                                                     \
                                                                                                        \
		PoolRealArray *_M1 = &(PoolRealArray)M1.values()[0];                                            \
		PoolRealArray *_M2 = &(PoolRealArray)M2.values()[0];                                            \
                                                                                                        \
		bool M1_gt_M2 = _M1->size() > _M2->size();                                                      \
		const Dictionary *small = !M1_gt_M2 ? &M1 : &M2;                                                \
		const Dictionary *large = M1_gt_M2 ? &M1 : &M2;                                                 \
                                                                                                        \
		/* TODO check if this is a deep copy */                                                         \
		Dictionary ans(*large);                                                                         \
		ewise_mm_##op##_in_place(ans, *small, check);                                                   \
		return ans;                                                                                     \
	}

ewise_mm_op_in_place(add, +=);
ewise_mm_op_in_place(mul, *=);

compare_mm_and_ewise_mm_op(add);
compare_mm_and_ewise_mm_op(mul);

real_t LinAlg::norm2_v(const PoolRealArray &v) {
	real_t ans = real_t(0);

	const real_t *v_read_ptr = v.read().ptr();
	for (int i = 0; i < v.size(); ++i) {
		real_t vi = v_read_ptr[i];
		ans += vi * vi;
	}

	return ans;
}

real_t LinAlg::norm_v(const PoolRealArray &v) {
	return std::sqrt(norm2_v(v));
}

void LinAlg::normalize_in_place(PoolRealArray &v) {
	ewise_vs_mul_in_place(v, real_t(1) / norm_v(v));
}

PoolRealArray LinAlg::normalize(const PoolRealArray &v) {
	PoolRealArray ans(v);
	normalize_in_place(ans);
	return ans;
}

real_t LinAlg::dot_vv(const PoolRealArray &v1, const PoolRealArray &v2) {
	if (v1.size() != v2.size()) {
		ERR_PRINT("Arguments to dot_vv must be equally long.");
		return real_t();
	}

	real_t ans = real_t();

	const real_t *v1_read_ptr = v1.read().ptr();
	const real_t *v2_read_ptr = v2.read().ptr();
	for (int i = 0; i < v1.size(); ++i) {
		ans += v1_read_ptr[i] * v2_read_ptr[i];
	}

	return ans;
}

PoolRealArray _dot_mm(const PoolRealArray &M1, int m1, int n1, const PoolRealArray &M2, int m2, int n2) {
	if (m2 != n1) {
		ERR_PRINT("There should be as many columns in M1 as there are rows in M2.");
		ERR_FAIL_V(PoolRealArray());
	}

	// TODO check if this hack really avoids copy-on-write
	PoolRealArray *ans = &_init_v(m1 * n2);
	real_t *ans_write_ptr = ans->write().ptr();
	const real_t *M1_read_ptr = M1.read().ptr();
	const real_t *M2_read_ptr = M2.read().ptr();

	for (int i = 0; i < m1; ++i) {
		for (int j = 0; j < n2; ++j) {
			real_t sum = real_t();
			for (int k = 0; k < m2; ++k) {
				sum += M1_read_ptr[n1 * i + k] * M2_read_ptr[n2 * k + j];
			}
			ans_write_ptr[n2 * i + j] = sum;
		}
	}

	return *ans;

	/*
	Rough work for the matrix-vector case.

	m2 = v.size
	n2 = 1 => j = 0

	for (int i = 0; i < m1; ++i) {
		real_t sum = real_t();
		for (int k = 0; k < v.size; ++k) {
			sum += M1_read_ptr[n1 * i + k] * M2_read_ptr[1 * k + 0];
		}
		ans_write_ptr[1 * i + 0] = sum;
	}

	== (k -> j) and if loop unrolls

	for (int i = 0; i < m1; ++i) {
		real_t sum = real_t();
		for (int j = 0; j < v.size; ++j) {
			sum += M1_read_ptr[n1 * i + j] * M2_read_ptr[j];
		}
		ans_write_ptr[i] = sum;
	}
	*/
}

PoolRealArray LinAlg::dot_mv(const Dictionary &M, const PoolRealArray &v, bool check = true) {
	M_CHECK_V(M, PoolRealArray());

	expand_m(M);
	// A vector is a 1D matrix
	PoolRealArray *ans = &::_dot_mm(*_M, m_M, n_M, v, v.size(), 1);
	return *ans; // size is m1
}

Dictionary LinAlg::dot_mm(const Dictionary &M1, const Dictionary &M2, bool check = true) {
	M_CHECK_V(M1, ::_make_m());
	M_CHECK_V(M2, ::_make_m());

	expand_m(M1);
	expand_m(M2);
	PoolRealArray *ans = &::_dot_mm(*_M1, m_M1, n_M1, *_M2, m_M2, n_M2);
	return ::_make_m(*ans, m_M1, n_M2);
}

void _minor(const PoolRealArray &M, int m, int n, int d, PoolRealArray &out_ans, int m_ans, int n_ans) {
	const real_t *M_read_ptr = M.read().ptr();
	real_t *out_ans_write_ptr = out_ans.write().ptr();

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			real_t x = real_t(i == j);

			bool i_or_j_lt_d = (i < d) || (j < d);
			x = (i_or_j_lt_d * x) + (!i_or_j_lt_d * M_read_ptr[n * i + j]);

			out_ans_write_ptr[n_ans * i + j] = x;
		}
	}
}

void _copy_column(const PoolRealArray &M, int m, int n, PoolRealArray &out_v, int j) {
	const real_t *M_read_ptr = M.read().ptr();
	real_t *out_v_write_ptr = out_v.write().ptr();

	for (int i = 0; i < m; ++i) {
		out_v_write_ptr[i] = M_read_ptr[n * i + j];
	}
}

Dictionary LinAlg::qr(const Dictionary &M, bool check = true) {
	M_CHECK_V(M, Dictionary());

	// The beauty of this technique is that you can interchange between
	// internal and wrapped matrix functions as required
	expand_m(M);
	const real_t *M_read_ptr = _M->read().ptr();
	int k_max = n_M < (m_M - 1) ? n_M : (m_M - 1);

	PoolRealArray *e = &init_v(m_M);
	real_t *e_write_ptr = e->write().ptr();
	PoolRealArray *x = &init_v(m_M);
	const real_t *x_read_ptr = x->read().ptr();
	// M used in PoolRealArray form
	PoolRealArray Z(*_M);
	PoolRealArray *Z1 = &_init_m(m_M, n_M);

	// Array of matrices
	Array vq;
	vq.resize(k_max);

	for (int k = 0; k < k_max; ++k) {
		_minor(Z, m_M, n_M, k, *Z1, m_M, n_M);
		_copy_column(*Z1, m_M, n_M, *x, k);

		real_t a = norm_v(*x);
		bool diag_gt_0 = M_read_ptr[k * (n_M + 1)]; // M[n * k + k]
		a = (diag_gt_0 * -a) + (!diag_gt_0 * a);

		for (int i = 0; i < m_M; ++i) {
			e_write_ptr[i] = x_read_ptr[i];
			e_write_ptr[i] += real_t((i == k) * a);
		}

		normalize_in_place(*e);
		vq[k] = householder(*e);
		Dictionary vq_k = ((Dictionary)vq[k]);
		// Z1 is in its PoolRealArray form, so use the internal _dot_mm call
		expand_m(vq_k);
		Z = _dot_mm(*_vq_k, m_vq_k, n_vq_k, *Z1, m_M, n_M);

		// Courtesy Variant::operator Dictionary()
		Dictionary Q = vq[0];
		for (int i = 1; i < k_max; ++i) {
			// Since I got a dictionary, why not just use Dictionaries?
			Q = dot_mm(vq[i], Q);
		}

		// M used in Dictionary form
		Dictionary R = dot_mm(Q, M);
		transpose_in_place(Q);

		return Dictionary::make("Q", Q, "R", R);
	}
}

Dictionary LinAlg::eigs_powerit_in_place(const Dictionary &M, real_t tol = real_t(1e-5), bool check = true) {
	M_CHECK_V(M, Dictionary());

	// Not ideal since m_M is unused
	expand_m(M);
	real_t *M_write_ptr = _M->write().ptr();

	Array evals, evecs;

	evecs.resize(n_M);
	evals.resize(n_M);

	for (int k = 0; k < n_M; ++k) {
		// Start with a random vector
		PoolRealArray v0 = rand_v(n_M);
		real_t e0 = real_t(0);
		real_t e1 = real_t();

		for (int t = 0; t < 100; ++t) {
			PoolRealArray *v1 = &dot_mv(M, v0);
			e1 = norm_v(*v1);
			ewise_vs_mul_in_place(*v1, real_t(1) / e1);

			if (std::fabsf(e1 - e0) < tol) {
				// Sign fix
				e1 *= dot_vv(v0, *v1);
				break;
			}
			e0 = e1;
			v0 = *v1;
		}

		evals[k] = e1;
		evecs[k] = v0;

		// Shift
		for (int i = 0; i < n_M; ++i) {
			// Can be used this way since it's not a const &
			real_t vi = v0[i];
			for (int j = 0; j < n_M; ++j) {
				M_write_ptr[n_M * i + j] -= e1 * vi * v0[j];
			}
		}
	}

	return Dictionary::make("evals", evals, "evecs", evecs);
}

Dictionary LinAlg::eigs_powerit(const Dictionary &M, real_t tol = real_t(1e-5), bool check = true) {
	M_CHECK_V(M, Dictionary());

	expand_m(M);
	Dictionary ans = ::_make_m(PoolRealArray(*_M), m_M, n_M);
	return eigs_powerit(ans, tol, check);
}
