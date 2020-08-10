extends Control

func get_v1():
	var v1 = Expression.new()
	v1.parse($Inputs/Vector1.text, [])
	return v1.execute()

func get_v2():
	var v2 = Expression.new()
	v2.parse($Inputs/Vector2.text, [])
	return v2.execute()

func get_M():
	var M = Expression.new()
	M.parse($Inputs/Matrix.text, [])
	return M.execute()

func _on_ButtonTestVector1_pressed():
	var v1 = get_v1()
	$TestVector1/Norm.text = str(LinAlg.norm_v(v1))


func _on_ButtonTestVector2_pressed():
	var v1 = get_v1()
	var v2 = get_v2()

	$TestVector2/Ewise.text = str(LinAlg.ewise_vv_add(v1, v2))
	$TestVector2/Dot.text = str(LinAlg.dot_vv(v1, v2))
	$TestVector2/Dyadic.text = str(LinAlg.dyadic(v2))


func _on_ButtonTestMatrix_pressed():
	var v1 = get_v1()
	var v2 = get_v2()
	var M = get_M()

	$TestMatrix/Dot.text = str(LinAlg.dot_mv(M, v1))
	$TestMatrix/Square.text = str(LinAlg.dot_mm(M, M))
	var QR = LinAlg.qr(M)
	$TestMatrix/Q.text = str(QR[0])
	$TestMatrix/R.text = str(QR[1])
	$TestMatrix/invQR.text = str(LinAlg.dot_mm(QR[0], QR[1]))

	var eig_pi = LinAlg.eigs_powerit(M.duplicate(true))

	$TestMatrix/EvalsPowIt.text = str(eig_pi[0])
	$TestMatrix/EvecsPowIt.text = str(eig_pi[1])
	$TestMatrix/EtestPowIt.text = ""
	for i in range(len(M)):
		var e1 = eig_pi[1][i]
		$TestMatrix/EtestPowIt.text += str(LinAlg.dot_vv(e1, LinAlg.dot_mv(M, e1)))
		$TestMatrix/EtestPowIt.text += ", "

