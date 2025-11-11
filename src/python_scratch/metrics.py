import math

def scr_root_mean_squared_error(y_true, y_pred) -> float:
    rmse = (sum((y_t - y_p)**2 for y_t, y_p in zip(y_true, y_pred)) / len(y_true)) **0.5
    return float(rmse)

def scr_root_mean_squared_log_error(y_true, y_pred) -> float:
    rmsle = (sum((math.log(y_t + 1) - math.log(y_p + 1))**2 for y_t, y_p in zip(y_true, y_pred)) / len(y_true)) **0.5
    return float(rmsle)

def scr_r2_score(y_true, y_pred) -> float:
    mean = sum(y_true) / len(y_true)
    ssr = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    sst = sum((y_t - mean) ** 2 for y_t in y_true)
    
    if sst == 0:
        return 0.0
    
    return float(1 - ssr / sst)

def scr_d2_absolute_error_score(y_true, y_pred) -> float:
    y_true_sorted = sorted(y_true)
    
    if len(y_true_sorted) % 2 == 1:
        median = y_true_sorted[len(y_true_sorted) // 2]
    else:
        median = (y_true_sorted[len(y_true_sorted) // 2 ] + y_true_sorted[(len(y_true_sorted) - 1) // 2 ]) / 2
        
    ssr = sum(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred))
    ssm = sum(abs(y_t - median) for y_t in y_true)
    
    if ssm == 0:
        return 0.0
    
    return float(1 - ssr / ssm)

def scr_mean_absolute_error(y_true, y_pred) -> float:
    mae = (sum(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred))) / len(y_true)
    return float(mae)

def scr_symmetric_mean_absolute_percentage_error(y_true, y_pred) -> float:
    n = len(y_true)
    smape_sum = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        denom = abs(y_t) + abs(y_p)
        if denom == 0:
            continue
        smape_sum += 2 * abs(y_t - y_p) / denom

    return float(100 * smape_sum / n)

def scr_max_error(y_true, y_pred) -> float:
    max_error = max(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred))
    
    return float(max_error)

def scr_median_absolute_error(y_true, y_pred) -> float:
    y_sorted = sorted(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred))
    
    if len(y_sorted) % 2 == 1:
        median = y_sorted[len(y_sorted) // 2]
    else:
        median = (y_sorted[len(y_sorted) // 2] + y_sorted[len(y_sorted)  // 2 - 1]) / 2
    
    return float(median)

def scr_explained_variance_score(y_true, y_pred) -> float:
    n = len(y_true)

    mean_y = sum(y_true) / n
    var_y = sum((y - mean_y) ** 2 for y in y_true) / n

    if var_y == 0:
        return 0.0

    errors = [y_t - y_p for y_t, y_p in zip(y_true, y_pred)]
    mean_err = sum(errors) / n
    var_err = sum((e - mean_err) ** 2 for e in errors) / n

    return float(1 - var_err / var_y)