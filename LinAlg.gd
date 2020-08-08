class_name LinAlg

# Linear Algebra library class

# All methods are optimised for maximum speed. They take arrays and assume the 
# right dimension for them. If the inputs aren't right they'll crash. Third input
# is used for the answer, preallocated. There are no conditional branchings. 
# Just use the method appropriate to the situation. The names are coded to reflect
# that. s = scalar, v = vector and m = matrix. So for example

# dot_vm 

# is a dot product between vector and matrix (in that order). Wherever the in_place
# argument is provided, it is possible to perform the operation on the object
# itself instead of instantiating a new one (this too optimises performance).


# Initialise a vector
static func init_v(n: int, v0: float=0.0)->Array:
	var ans = []
	ans.resize(n)
	
	for i in range(n):
		ans[i] = v0
			
	return ans


# Initialise a matrix
static func init_m(m: int, n: int, m0: float=0.0)->Array:
	var ans = []
	ans.resize(m)
	
	for i in range(m):
		var row = []
		row.resize(n)
		for j in range(n):
			row[j] = m0
		ans[i] = row
	
	return ans


# Identity matrix
static func eye(n: int)->Array:
	var ans = []
	ans.resize(n)
	
	for i in range(n):
		var row = []
		row.resize(n)
		for j in range(n):
			row[j] = 1 if i == j else 0
		ans[i] = row

	return ans


# Diagonal matrix
static func diag(v: Array)->Array:
	var n = len(v)
	var ans = []
	ans.resize(n)
	
	for i in range(n):
		var row = []
		row.resize(n)
		var vi = v[i]
		for j in range(n):
			row[j] = vi if i == j else 0
		ans[i] = row

	return ans


# Dyadic matrix
static func dyadic(v: Array)->Array:
	var n = len(v)
	var ans = []
	ans.resize(n)
	
	for i in range(n):
		var vi = v[i]
		var row = v.duplicate()
		for j in range(n):
			row[j] *= vi
		ans[i] = row
	
	return ans


# Transpose
static func transpose(M: Array, in_place: bool=false)->Array:
	var n = len(M)
	var ans
	if not in_place:
		ans = M.duplicate(true)
	else:
		ans = M
	
	for i in range(n-1):
		var row = ans[i]
		for j in range(i+1,n):
			var dummy = row[j]
			row[j] = ans[j][i]
			ans[j][i] = dummy
	
	return ans


# Householder matrix from vector
# (https://en.wikipedia.org/wiki/Householder_transformation)
static func householder(v: Array)->Array:
	var n = len(v)
	var ans = []
	ans.resize(n)
	
	for i in range(n):
		var vi = -v[i]*2
		var row = v.duplicate()
		for j in range(n):
			row[j] *= vi
			if i == j:
				row[j] += 1
		ans[i] = row
	
	return ans


# Random vector
static func rand_v(n: int, s: float=1)->Array:
	var ans = []
	ans.resize(n)
	
	for i in range(n):
		ans[i] = randf()*s
	
	return ans


# Random matrix
static func rand_m(m: int, n: int, s: float=1)->Array:
	var ans = []
	ans.resize(n)
	
	for i in range(m):
		var row = []
		row.resize(n)
		for j in range(n):
			row[j] = randf()*s
		ans[i] = row
	
	return ans


# Element-wise: vector plus scalar
static func ewise_vs_add(v: Array, s: float, in_place: bool=false)->Array:
	
	var n = len(v)
	var ans
	if in_place:
		ans = v
	else:
		ans = init_v(n)
	
	for i in range(n):
		ans[i] = v[i]+s
	
	return ans


# Element-wise: vector times scalar
static func ewise_vs_mul(v: Array, s: float, in_place: bool=false)->Array:
	
	var n = len(v)
	var ans
	if in_place:
		ans = v
	else:
		ans = init_v(n)
	
	for i in range(n):
		ans[i] = v[i]*s
	
	return ans


# Element-wise: vector plus vector
static func ewise_vv_add(v: Array, v2: Array, in_place: bool=false)->Array:
	
	var n = len(v)
	var ans
	if in_place:
		ans = v
	else:
		ans = init_v(n)
	
	for i in range(n):
		ans[i] = v[i]+v2[i]
	
	return ans
	
# Element-wise: vector times vector
static func ewise_vv_mul(v: Array, v2: Array, in_place: bool=false)->Array:
	
	var n = len(v)
	var ans
	if in_place:
		ans = v
	else:
		ans = init_v(n)
	
	for i in range(n):
		ans[i] = v[i]*v2[i]
	
	return ans


# Element-wise: matrix plus scalar
static func ewise_ms_add(M: Array, s: float, in_place: bool=false)->Array:
	
	var m = len(M)
	var n = len(M[0])
	var ans
	if in_place:
		ans = M
	else:
		ans = init_m(m, n)
	
	for i in range(m):
		var rowin = M[i]
		var rowout = ans[i]
		for j in range(n):
			rowout[j] = rowin[j]+s
			
	return ans


# Element-wise: matrix times scalar
static func ewise_ms_mul(M: Array, s: float, in_place: bool=false)->Array:
	
	var m = len(M)
	var n = len(M[0])
	var ans
	if in_place:
		ans = M
	else:
		ans = init_m(m, n)
	
	for i in range(m):
		var rowin = M[i]
		var rowout = ans[i]
		for j in range(n):
			rowout[j] = rowin[j]*s	
			
	return ans


# Element-wise: matrix plus matrix
static func ewise_mm_add(M: Array, M2: Array, in_place: bool=false)->Array:
	
	var m = len(M)
	var n = len(M[0])
	var ans
	if in_place:
		ans = M
	else:
		ans = init_m(m, n)
	
	for i in range(m):
		var row1 = M[i]
		var row2 = M2[i]
		var rowout = ans[i]
		for j in range(n):
			rowout[j] = row1[j]+row2[j]
			
	return ans


# Element-wise: matrix times matrix
static func ewise_mm_mul(M: Array, M2: Array, in_place: bool=false)->Array:
	
	var m = len(M)
	var n = len(M[0])
	var ans
	if in_place:
		ans = M
	else:
		ans = init_m(m, n)
	
	for i in range(m):
		var row1 = M[i]
		var row2 = M2[i]
		var rowout = ans[i]
		for j in range(n):
			rowout[j] = row1[j]*row2[j]
			
	return ans


# Norm^2 of vector
static func norm2_v(v: Array)->float:
	var ans = 0.0
	
	for i in range(len(v)):
		ans += pow(v[i], 2)
	
	return ans


# Norm of vector
static func norm_v(v: Array)->float:
	return sqrt(norm2_v(v))


# Normalize
static func normalized_v(v: Array, in_place: bool=false)->Array:
	var norm = norm_v(v)
	return ewise_vs_mul(v, 1.0/norm, in_place)


# Dot product: matrix times vector
static func dot_mv(M: Array, v: Array)->Array:
	var m = len(M)
	var n = len(v)
	var ans = init_v(m)
	
	for i in range(m):
		var tot = 0.0
		var row = M[i]
		for j in range(n):
			tot += row[j]*v[j]
		ans[i] = tot
		
	return ans


# Dot product: matrix times matrix
static func dot_mm(M: Array, M2: Array)->Array:
	var m = len(M)
	var n = len(M2[0])
	var nn = len(M2)
	var ans = init_m(m, n)
	
	for i in range(m):
		var row = M[i]
		var rowout = ans[i]
		for j in range(n):
			var tot = 0.0
			for k in range(nn):
				tot += row[k]*M2[k][j]
			rowout[j] = tot
		
	return ans


# Dot product: vector times vector
static func dot_vv(v: Array, v2: Array)->float:
	var n = len(v)
	var ans = 0.0
	
	for i in range(n):
		ans += v[i]*v2[i]
	
	return ans


# Utilities for QR: Extract minor
static func _minor(M: Array, d: int, ans: Array)->void:
	var n = len(M)
	
	for i in range(n):
		var row = M[i]
		var rowout = ans[i]
		rowout.resize(n)
		for j in range(n):
			var x = 1 if i == j else 0
			x = x if (i < d or j < d) else row[j]
			rowout[j] = x


# Utilities for QR: copy column
static func _copycol(M: Array, v: Array, j: int)->void:
	var m = len(M)
	
	for i in range(m):
		v[i] = M[i][j]


# QR decomposition
static func qr(M: Array)->Array:
	var m = len(M) 
	var n = len(M[0])
	var kmax = min(n, m-1)
	
	var e = init_v(m)
	var x = init_v(m)
	var z = M.duplicate(true)
	var z1 = init_m(m, n)
	
	var vq = []
	vq.resize(kmax)
	
	for k in range(kmax):
		var a
		
		# Compute minor
		_minor(z, k, z1)
		
		# Extract column
		_copycol(z1, x, k)
		
		a = norm_v(x)
		a = -a if M[k][k] > 0 else a
		
		for i in range(m):
			e[i] = x[i]
			if i == k:
				e[i] += a
		
		normalized_v(e, true)
		vq[k] = householder(e)
		z = dot_mm(vq[k], z1)
	
	var Q = vq[0]
	for i in range(1, kmax):
		Q = dot_mm(vq[i], Q)
	
	var R = dot_mm(Q, M)
	transpose(Q, true)
	
	return [Q, R]


# Eigenvalues by power iteration for symmetric matrices
static func eigs_powerit(M: Array, tol: float=1e-5, in_place: bool=false)->Array:
	var n = len(M)
	if not in_place:
		M = M.duplicate(true)
	
	var evals = []
	var evecs = []
	
	evals.resize(n)
	evecs.resize(n)
	
	for k in range(n):
		# Start with a random vector
		var v0 = rand_v(n)
		var e0 = 0
		var v1
		var e1
		for t in range(100):
			v1 = dot_mv(M, v0)
			e1 = norm_v(v1)
			ewise_vs_mul(v1, 1.0/e1, true)
			if abs(e1-e0) < tol:
				# Sign fix
				e1 *= dot_vv(v0, v1)
				break
			e0 = e1
			v0 = v1
		evals[k] = e1
		evecs[k] = v0
		
		# Shift
		for i in range(n):
			var row = M[i]
			var vi = v0[i]
			for j in range(n):
				row[j] -= e1*vi*v0[j]
	
	return [evals, evecs]


# Eigenvalues by QR decomposition (still in development, commented out for now)

# static func eigs_qr(M: Array, tol: float=1e-8)->Array:
# 	var n = len(M)
	
# 	var evals = []
# 	var evecs = eye(n)
	
# 	evals.resize(n)
	
# 	var A = M.duplicate(true)
	
# 	for t in range(100):
# 		# Compute the Wilkinson shift
# 		var a = A[n-2][n-2]
# 		var b = A[n-1][n-2]
# 		var c = A[n-1][n-1]
# 		var del = (a-c)/2.0
# 		var ws = c-(sign(del) if del != 0 else 1)*b*b/(abs(del)+sqrt(del*del+a*a))
# 		for i in range(n):
# 			A[i][i] -= ws
# 		var QR = qr(A)
# 		A = dot_mm(QR[1], QR[0])
# 		for i in range(n):
# 			A[i][i] += ws
# 		evecs = dot_mm(evecs, QR[0])
# 		if abs(A[n-1][n-2]) < tol:
# 			break 
	
# 	for i in range(n):
# 		evals[i] = A[i][i]
	
# 	transpose(evecs, true)
		
# 	return [evals, evecs]

