class PL2(metapy.index.RankingFunction):
	def __init__(self, param=0.5):
		self.param = param
		# You *must* call the base class constructor here!
		super(PL2, self).__init__()

	def score_one(self, sd):
		lamda = sd.num_docs/sd.corpus_term_count
		tfn = sd.doc_term_count * math.log((1 + self.param * (sd.avg_dl/sd.doc_size)), 2)
		if lamda < 1 or tfn <= 0 :
			return 0
		result = tfn * math.log((tfn * lamda), 2)
		result += math.log(math.exp(1), 2) * ((1/lamda)-tfn)
		result += 0.5 * math.log((2*math.pi*tfn), 2)
		result /= (tfn + 1)
		result *= sd.query_term_weight
		return result

def load_ranker(cfg_file):
	return PL2(5.0)

