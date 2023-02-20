import numpy as np
    
import matplotlib.pyplot as plt

from sklearn import metrics


def pfbeta(labels, predictions, beta=1.0):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0
    
def optimal_f1(labels, predictions):

    thres = np.linspace(0.001, 0.5, 20)
#     thres = np.linspace(0.001, 0.999, 30)
    #f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    f1s = []
    for thr in thres:
        pred = (np.array(predictions) > thr).astype(np.int8).tolist()
        try:
            f1s.append(float(pfbeta(labels, pred)))
        except:
            f1s.append(0.0)
        
    idx = np.argmax(f1s)

    return f1s[idx], thres[idx]
    




# ref https://storage.googleapis.com/kaggle-forum-message-attachments/2101688/18584/run_all_valid1.py

def np_binary_cross_entropy_loss(probability, truth):
    probability = probability.astype(np.float64)
    probability = np.nan_to_num(probability, nan=1, posinf=1, neginf=0)

    p = np.clip(probability, 1e-5, 1 - 1e-5)
    y = truth

    loss = -y * np.log(p) - (1 - y) * np.log(1 - p)
    loss = loss.mean()
    return loss

def get_f1score(probability, truth, threshold = np.linspace(0, 1, 50)):
    f1score = []
    precision=[]
    recall=[]
    for t in threshold:
        predict = (probability > t).astype(np.float32)

        tp = ((predict >= 0.5) & (truth >= 0.5)).sum()
        fp = ((predict >= 0.5) & (truth < 0.5)).sum()
        fn = ((predict < 0.5) & (truth >= 0.5)).sum()

        r = tp / (tp + fn + 1e-3)
        p = tp / (tp + fp + 1e-3)
        f1 = 2 * r * p / (r + p + 1e-3)
        f1score.append(f1)
        precision.append(p)
        recall.append(r)
    f1score = np.array(f1score)
    precision = np.array(precision)
    recall = np.array(recall)
    return f1score, precision, recall, threshold

def plot_auc(cancer_p, cancer_t, figure_num):
    plt.figure(figure_num)
    spacing=50
    cancer_t = cancer_t.astype(int)
    pos, bin = np.histogram(cancer_p[cancer_t == 1], np.linspace(0, 1, spacing))
    neg, bin = np.histogram(cancer_p[cancer_t == 0], np.linspace(0, 1, spacing))
    pos = pos / (cancer_t == 1).sum()
    neg = neg / (cancer_t == 0).sum()
    #print(pos)
    #print(neg)
    # plt.plot(bin[1:],neg, alpha=1)
    # plt.plot(bin[1:],pos, alpha=1)
    bin = (bin[1:] + bin[:-1]) / 2
    plt.bar(bin, neg, width=1/spacing, label='neg', alpha=0.5)
    plt.bar(bin, pos, width=1/spacing, label='pos', alpha=0.5)
    plt.legend()
    #plt.yscale('log')
    # if is_show:
    #     plt.show()
    # return  plt.gcf()

def compute_metric(cancer_p, cancer_t):
    try:
        fpr, tpr, thresholds = metrics.roc_curve(cancer_t, cancer_p)
    except ValueError:
        fpr = 0.0
        tpr = 0.0
        thresholds = 0.0
        print("Input contains NaN")

    auc = metrics.auc(fpr, tpr)

    f1score, precision, recall, threshold = get_f1score(cancer_p, cancer_t)
    i = f1score.argmax()
    f1score, precision, recall, threshold = f1score[i], precision[i], recall[i], threshold[i]

    specificity = ((cancer_p < threshold ) & ((cancer_t <= 0.5))).sum() / (cancer_t <= 0.5).sum()
    sensitivity = ((cancer_p >= threshold) & ((cancer_t >= 0.5))).sum() / (cancer_t >= 0.5).sum()
    pf1,pthr = optimal_f1(cancer_t, cancer_p)
    return {
        'auc': auc,
        'threshold': threshold,
        'f1score': f1score,
        'pf1':pf1,
        'pthr':pthr,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }

def compute_pfbeta(labels, predictions, beta=1):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
            #cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp+1e-8)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def print_all_metric(valid_df):

	print(f'{"    ": <16}    \tauc      @th     f1      | 	prec    recall  | 	sens    spec ')
	#log.write(f'{"    ": <16}    \t0.77902	0.44898	0.28654 | 	0.32461	0.25726 | 	0.25726	0.98794\n')
	for site_id in [0,1,2]:

		#log.write(f'*** site_id [{site_id}] ***\n')
		#log.write(f'\n')

		if site_id>0:
			site_df = valid_df[valid_df.site_id == site_id].reset_index(drop=True)
		else:
			site_df = valid_df
		# ---

		gb = site_df
		m = compute_metric(gb.cancer_p, gb.cancer_t)
		text = f'{"single image": <16} [{site_id}]'
		text += f'\t{m["auc"]:0.5f}'
		text += f'\t{m["threshold"]:0.5f}'
		text += f'\t{m["f1score"]:0.5f} | '
		text += f'\t{m["precision"]:0.5f}'
		text += f'\t{m["recall"]:0.5f} | '
		text += f'\t{m["sensitivity"]:0.5f}'
		text += f'\t{m["specificity"]:0.5f}'
		#text += '\n'
		print(text)


		# ---

		gb = site_df[['patient_id', 'laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).mean()
		m = compute_metric(gb.cancer_p, gb.cancer_t)
		text = f'{"groupby mean()": <16} [{site_id}]'
		text += f'\t{m["auc"]:0.5f}'
		text += f'\t{m["threshold"]:0.5f}'
		text += f'\t{m["f1score"]:0.5f} | '
		text += f'\t{m["precision"]:0.5f}'
		text += f'\t{m["recall"]:0.5f} | '
		text += f'\t{m["sensitivity"]:0.5f}'
		text += f'\t{m["specificity"]:0.5f}'
		#text += '\n'
		print(text)

		# ---
		gb = site_df[['patient_id', 'laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).max()
		m = compute_metric(gb.cancer_p, gb.cancer_t)
		text = f'{"groupby max()": <16} [{site_id}]'
		text += f'\t{m["auc"]:0.5f}'
		text += f'\t{m["threshold"]:0.5f}'
		text += f'\t{m["f1score"]:0.5f} | '
		text += f'\t{m["precision"]:0.5f}'
		text += f'\t{m["recall"]:0.5f} | '
		text += f'\t{m["sensitivity"]:0.5f}'
		text += f'\t{m["specificity"]:0.5f}'
		#text += '\n'
		print(text)
		print(f'--------------\n')

