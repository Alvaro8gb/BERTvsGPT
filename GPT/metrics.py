def precision(tp, fp):
    if tp + fp == 0:
        return 1
    else:
        return tp/(tp+fp)


def recall(tp, fn):
    if tp + fn == 0:
        return 1
    else:
        return tp/(tp+fn)


def fscore(precision, recall):
    return (2*precision*recall) / (precision+recall)

def fscore_base(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)

    if p == 0 and r == 0:
        f = 0
    else:
        f = fscore(p, r)

    return f

def metrics(tp, tn, fp, fn):
    """
    In NER True negative is not used, so on accuracy also not
    """

    p = precision(tp, fp)
    r = recall(tp, fn)

    if p == 0 and r == 0:
        f = 0
    else:
        f = fscore(p, r)

    return p, r, f


def calculate_metrics_text_only(y_true, y_pred):
    y_true_texts = [item[0] for item in y_true]
    y_pred_texts = [item[0] for item in y_pred]

    true_positive_text = 0
    false_positive_text = 0

    for pred_text in y_pred_texts:
        if pred_text in y_true_texts:
            true_positive_text += 1
            y_true_texts.remove(pred_text)  # Delete to avoid duplicates
        else:
            false_positive_text += 1

    false_negative_text = len(y_true_texts)

    return true_positive_text, false_negative_text, false_positive_text


def calculate_metrics_start_end(y_true, y_pred):
    true_positive_start_end = 0
    false_positive_start_end = 0
    false_negative_start_end = 0

    y_pred_set = set(y_pred)
    y_true_set = set(y_true)

    true_positive_start_end = len(y_pred_set & y_true_set)

    if len(y_pred) == 0:
        false_negative_start_end = len(y_true)
    else:
        false_positive_start_end = len(y_pred_set - y_true_set)

    return true_positive_start_end, false_negative_start_end, false_positive_start_end


def calculate_metrics_start(y_true, y_pred):
    y_true_copy = y_true[:]
    true_positive_start = 0
    false_negative_start = 0
    false_positive_start = 0

    for pred_item in y_pred:
        pred_text, pred_start, pred_end = pred_item
        for true_item in y_true_copy:
            true_text, true_start, true_end = true_item
            if pred_start == true_start:
                true_positive_start += 1
                y_true_copy.remove(true_item)
            else:
                false_positive_start += 1
    if len(y_pred) == 0:
        false_negative_start = len(y_true)

    return true_positive_start, false_negative_start, false_positive_start


def calculate_metrics_end(y_true, y_pred):
    y_true_copy = y_true[:]
    true_positive_end = 0
    false_negative_end = 0
    false_positive_end = 0

    for pred_item in y_pred:
        pred_text, pred_start, pred_end = pred_item
        for true_item in y_true_copy:
            true_text, true_start, true_end = true_item
            if pred_end == true_end:
                true_positive_end += 1
                y_true_copy.remove(true_item)
            else:
                false_positive_end += 1
    if len(y_pred) == 0:
        false_negative_end = len(y_true)

    return true_positive_end, false_negative_end, false_positive_end


def calculate_metrics_soft(y_true, y_pred):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    y_true_texts = {yt[0] for yt in y_true}
    y_pred_texts = {yp[0] for yp in y_pred}

    for pred in y_pred:
        if pred[0] in y_true_texts:
            true_positives += 1
        else:
            false_positives += 1

    for true in y_true:
        if true[0] not in y_pred_texts:
            false_negatives += 1

    return true_positives, false_positives, false_negatives
 
def calculate_mc(results, calcualte_metrics):
    tp = 0
    fn = 0
    fp = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for result in results:
        y_pred, y_true = result

        tp, fn, fp = calcualte_metrics(y_true, y_pred)

        total_tp += tp
        total_fn += fn
        total_fp += fp

    print("Total True Positive:", total_tp)
    print("Total False Positive:", total_fp)
    print("Total False Negative:", total_fn)

    return total_tp, total_fn, total_fp


def show_eval(tp, fn, fp):
    p, r, f = metrics(tp, 0, fp, fn)
    print("Precision: {:.2f}".format(p))
    print("Recall: {:.2f}".format(r))
    print("F1 Score: {:.2f}".format(f))
    print("-"*20)

    return p, r, f


def process_results(results, calculate_metrics_function, label):
    metrics = []
    for m, sets in results.items():
        for s, types_promt in sets.items():
            for promt, info in types_promt.items():
                #print(m, s, promt, "Money:", info["money"])
                total_tp, total_fp, total_fn = calculate_mc(
                    info["results"], calculate_metrics_function)
                p, r, f = show_eval(total_tp, total_fp, total_fn)
                
                metrics.append(
                    {"Label": label, "Model": m, "Set": s, "Promt": promt,  "p": p, "r": r, "f": f, "tp":total_tp, "fp":total_fp, "fn":total_fn})

    return metrics
