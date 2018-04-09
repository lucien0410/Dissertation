reference=[[str(i) for i in range(10)]]
for k in range(1,11):
	print k
	candidate =[str(1) for k in range(k)]+['2' for o in range(9-k)]
	score=sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
	print reference
	print candidate
	print score
	print '\n'
