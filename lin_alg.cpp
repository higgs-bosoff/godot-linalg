#include "lin_alg.h"

#include <cmath>
#include <cstring>
#include <utility>

#include <RandomNumberGenerator.hpp>

using namespace godot;

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
	register_method("dot_mm", &LinAlg::dot_mm);
}

inline PoolRealArray init_v(int n, real_t v0 = real_t(0)) {
	PoolRealArray ans;
	ans.resize(n);
	real_t *ans_write_ptr = ans.write().ptr();
	std::memset(ans_write_ptr, v0, (n * sizeof(real_t)));
	// for (int i = 0; i < n; ++i) {
	//     ans_write_ptr[i] = v0;
	// }

	return ans;
}

PoolRealArray LinAlg::init_v(int n, real_t v0 = real_t()) {
	return ::init_v(n, v0);
}

PoolRealArray LinAlg::init_m(int m, int n, real_t m0 = real_t()) {
	return ::init_v(m * n, m0);
}

PoolRealArray LinAlg::eye(const int n) {
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

PoolRealArray LinAlg::diag(const PoolRealArray &v) {
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

PoolRealArray LinAlg::dyadic(const PoolRealArray &v) {
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

inline void transpose_in_place(PoolRealArray &M, int n) {
	real_t *M_write_ptr = M.write().ptr();

	for (int i = 0; i < n - 1; ++i) {
		for (int j = i + 1; j < n; ++j) {
			std::swap(M_write_ptr[n * j + i], M_write_ptr[n * i + j]);
		}
	}
}

void LinAlg::transpose_in_place(PoolRealArray &M, int n) {
	::transpose_in_place(M, n);
}

PoolRealArray LinAlg::transpose(const PoolRealArray &M, int n) {
	PoolRealArray ans(M);
	::transpose_in_place(ans, n);
	return ans;
}

PoolRealArray LinAlg::householder(const PoolRealArray &v) {
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

	return ans;
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

PoolRealArray LinAlg::rand_m(int m, int n, real_t s = real_t(1)) {
	return rand_v(m * n, s);
}

void LinAlg::ewise_vs_add_in_place(PoolRealArray &v, real_t s) {
	real_t *v_write_ptr = v.write().ptr();

	for (int i = 0; i < v.size(); ++i) {
		v_write_ptr[i] += s;
	}
}

PoolRealArray LinAlg::ewise_vs_add(const PoolRealArray &v, real_t s) {
	PoolRealArray ans(v);
	ewise_vs_add_in_place(ans, s);
	return ans;
}

void LinAlg::ewise_vs_mul_in_place(PoolRealArray &v, real_t s) {
	real_t *v_write_ptr = v.write().ptr();

	for (int i = 0; i < v.size(); ++i) {
		v_write_ptr[i] *= s;
	}
}

PoolRealArray LinAlg::ewise_vs_mul(const PoolRealArray &v, real_t s) {
	PoolRealArray ans(v);
	ewise_vs_mul_in_place(ans, s);
	return ans;
}

inline void stretch_to_fit_v(PoolRealArray &to_stretch, const PoolRealArray &to_fit) {
	// For being absolutely branchless
	//bool to_stretch_lt_to_fit = to_stretch.size() < to_fit.size();
	// resizes if to_fit is longer
	//int new_size = to_stretch_lt_to_fit * to_fit.size() + !to_stretch_lt_to_fit * to_stretch.size();
	//to_stretch.resize(new_size);

	// this ought to be easy to do (and is probably safer) with
	if (to_stretch.size() < to_fit.size()) {
		to_stretch.resize(to_fit.size());
	}
}

// It is up to the user to avoid costly stretching by supplying a longer v1
// The implementation must promise that only the first argument is modified
// Probably desirable if user expects to make this vector larger
void LinAlg::ewise_vv_add_in_place(PoolRealArray &v1, const PoolRealArray &v2) {
	::stretch_to_fit_v(v1, v2);
	real_t *v1_write_ptr = v1.write().ptr();
	const real_t *v2_read_ptr = v2.read().ptr();

	// add all elements based on v2's length
	for (int i = 0; i < v2.size(); ++i) {
		v1_write_ptr[i] += v2_read_ptr[i];
	}
}

// This needn't be the case if it's not in-place
PoolRealArray LinAlg::ewise_vv_add(const PoolRealArray &v1, const PoolRealArray &v2) {
	bool v1_gt_v2 = v1.size() > v2.size();
	const PoolRealArray *small = !v1_gt_v2 ? &v1 : &v2;
	const PoolRealArray *large = v1_gt_v2 ? &v1 : &v2;

	PoolRealArray ans(*large);
	ewise_vv_add_in_place(ans, *small);
	return ans;
}

// Similarly,

// It is up to the user to avoid costly stretching by supplying a longer v1
// The implementation must promise that only the first argument is modified
// Probably desirable if user expects to make this vector larger
void LinAlg::ewise_vv_mul_in_place(PoolRealArray &v1, const PoolRealArray &v2) {
	::stretch_to_fit_v(v1, v2);
	real_t *v1_write_ptr = v1.write().ptr();
	const real_t *v2_read_ptr = v2.read().ptr();

	// mul all elements based on v2's length
	for (int i = 0; i < v2.size(); ++i) {
		v1_write_ptr[i] *= v2_read_ptr[i];
	}
}

// Again,

// This needn't be the case if it's not in-place
PoolRealArray LinAlg::ewise_vv_mul(const PoolRealArray &v1, const PoolRealArray &v2) {
	bool v1_gt_v2 = v1.size() > v2.size();
	const PoolRealArray *small = !v1_gt_v2 ? &v1 : &v2;
	const PoolRealArray *large = v1_gt_v2 ? &v1 : &v2;

	PoolRealArray ans(*large);
	ewise_vv_mul_in_place(ans, *small);
	return ans;
}

// internals are alike, just reuse the functions

void LinAlg::ewise_ms_add_in_place(PoolRealArray &M, real_t s) {
	ewise_vs_add_in_place(M, s);
}

PoolRealArray LinAlg::ewise_ms_add(const PoolRealArray &M, real_t s) {
	PoolRealArray ans(M);
	ewise_ms_add_in_place(ans, s);
	return ans;
}

void LinAlg::ewise_ms_mul_in_place(PoolRealArray &M, real_t s) {
	ewise_vs_mul_in_place(M, s);
}

PoolRealArray LinAlg::ewise_ms_mul(const PoolRealArray &M, real_t s) {
	PoolRealArray ans(M);
	ewise_ms_mul_in_place(ans, s);
	return ans;
}

/*
stretch to fit 2d
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

inline void stretch_to_fit_m(PoolRealArray &to_stretch, int n1, const PoolRealArray &to_fit, int n2) {
	// Num columns
	int m1 = to_stretch.size() / n1;
	int m2 = to_fit.size() / n2;

	if (m1 > m2 || n1 > n2) {
		return;
	}
	// now n1 <= n2 and m1 <= n2

	// only stretch rows if possible
	if (n1 < n2 /* && m1 == m2*/) {
		// [1 2 3;4 5 6] -> [1 2 3;4 5 6;0 0 0]
		// to_stretch.resize(to_fit.size());
		to_stretch.append_array(init_v(m2 * (n2 - n1)));
	}

	// stretch columns
	if (m1 < m2) {
		// [1 2 3;4 5 6] -> [1 4;2 5;3 6]
		transpose_in_place(to_stretch, n2);
		// [1 4;2 5;3 6] -> [1 4;2 5;3 6;0 0]
		// to_stretch.resize(to_fit.size());
		to_stretch.append_array(init_v(n2 * (m2 - m1)));
		// [1 4;2 5;3 6;0 0] -> [1 2 3 0;4 5 6 0]
		transpose_in_place(to_stretch, m2);
	}
}

void LinAlg::ewise_mm_add_in_place(PoolRealArray &M1, int n1, const PoolRealArray &M2, int n2) {
	::stretch_to_fit_m(M1, n1, M2, n2);
	real_t *M1_write_ptr = M1.write().ptr();
	const real_t *M2_read_ptr = M2.read().ptr();

	// add all elements based on M2's length
	for (int i = 0; i < M2.size(); ++i) {
		M1_write_ptr[i] += M2_read_ptr[i];
	}
}

PoolRealArray LinAlg::ewise_mm_add(const PoolRealArray &M1, int n1, const PoolRealArray &M2, int n2) {
	bool n1_gt_n2 = n1 > n2;
	int small_n = !n1_gt_n2 ? n1 : n2;
	int large_n = n1_gt_n2 ? n1 : n2;

	bool M1_gt_M2 = M1.size() > M2.size();
	const PoolRealArray *small = !M1_gt_M2 ? &M1 : &M2;
	const PoolRealArray *large = M1_gt_M2 ? &M1 : &M2;

	PoolRealArray ans(*large);
	ewise_mm_add_in_place(ans, large_n, *small, small_n);
	return ans;
}

void LinAlg::ewise_mm_mul_in_place(PoolRealArray &M1, int n1, const PoolRealArray &M2, int n2) {
	::stretch_to_fit_m(M1, n1, M2, n2);
	real_t *M1_write_ptr = M1.write().ptr();
	const real_t *M2_read_ptr = M2.read().ptr();

	// mul all elements based on M2's length
	for (int i = 0; i < M2.size(); ++i) {
		M1_write_ptr[i] *= M2_read_ptr[i];
	}
}

PoolRealArray LinAlg::ewise_mm_mul(const PoolRealArray &M1, int n1, const PoolRealArray &M2, int n2) {
	bool n1_gt_n2 = n1 > n2;
	int small_n = !n1_gt_n2 ? n1 : n2;
	int large_n = n1_gt_n2 ? n1 : n2;

	bool M1_gt_M2 = M1.size() > M2.size();
	const PoolRealArray *small = !M1_gt_M2 ? &M1 : &M2;
	const PoolRealArray *large = M1_gt_M2 ? &M1 : &M2;

	PoolRealArray ans(*large);
	ewise_mm_mul_in_place(ans, large_n, *small, small_n);
	return ans;
}

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

PoolRealArray LinAlg::dot_mm(const PoolRealArray &M1, int n1, const PoolRealArray &M2, int n2) {
	int m2 = M2.size() / n2;
	if (m2 != n1) {
		ERR_PRINT("There should be as many columns in lhs as there are rows in rhs.");
		return PoolRealArray();
	}

	int m1 = M1.size() / n1;
	// TODO check if this hack really avoids copy-on-write
	PoolRealArray *ans = &init_m(m1, n2);
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
}
