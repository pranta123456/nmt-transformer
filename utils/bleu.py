import math
from collections import Counter
import numpy as np

class BLEU:
    def __init__(self, n_gram):
        self.n_gram = n_gram
    
    def create_ngram(self, sent):

        return [tuple(sent[i:i+self.n_gram]) for i in range(len(sent)-self.n_gram+1)]

    def modified_precision(self, reference, candidate):
        ref_ngarm = Counter(self.create_ngram(reference))
        cand_ngarm = Counter(self.create_ngram(candidate))

        count_clip = sum(min(ref_ngarm[ng], cand_ngarm[ng]) for ng in cand_ngarm)
        total_count = sum(cand_ngarm.values())

        # return float(count_clip) / total_count if total_count > 0 else 0
        return count_clip, total_count

    def brevity_penalty(self, reference, candidate):
        ref_len = len(reference)
        cand_len = len(candidate)

        if cand_len > ref_len:
            return 1
        elif cand_len == 0:
            return 0
        else:
            return math.exp(1 - float(ref_len) / cand_len)

    def bleu_score(self, reference, candidate):
        bp = self.brevity_penalty(reference, candidate)
        precision = self.modified_precision(reference, candidate)

        return bp * precision
    

def combined_bleu_score(reference, candidate):
    bleu_1 = BLEU(1)
    bleu_2 = BLEU(2)
    bleu_3 = BLEU(3)
    bleu_4 = BLEU(4)
    
    # p1, p2, p3, p4 
    t1, t2, t3, t4= bleu_1.modified_precision(reference, candidate), bleu_2.modified_precision(reference, candidate), bleu_3.modified_precision(reference, candidate), bleu_4.modified_precision(reference, candidate)
    
    precisions = [t1[0], t1[1], t2[0], t2[1], t3[0], t3[1], t4[0], t4[1]] #[p1, p2, p3, p4]
    if len(list(filter(lambda x : x == 0, precisions))) > 0:
        return 0
    
    GAP = math.exp(sum([math.log(float(x) / y) for x, y in zip(precisions[0::2], precisions[1::2])]) / 4.)
    bp = bleu_1.brevity_penalty(reference, candidate)
    
    return bp * GAP * 100


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)