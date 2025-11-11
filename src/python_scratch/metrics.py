import math

def scr_root_mean_squared_error(y_true, y_pred):
    rmse = (sum((y_t - y_p)**2 for y_t, y_p in zip(y_true, y_pred)) / len(y_true)) **0.5
    return float(rmse)

def scr_root_mean_squared_log_error(y_true, y_pred):
    rmsle = (sum((math.log(y_t + 1) - math.log(y_p + 1))**2 for y_t, y_p in zip(y_true, y_pred)) / len(y_true)) **0.5
    return float(rmsle)

def scr_r2_score(y_true, y_pred):
    mean = sum(y_true) / len(y_true)
    ssr = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    sst = sum((y_t - mean) ** 2 for y_t in y_true)
    
    if sst == 0:
        return 0.0
    
    return float(1 - ssr / sst)

def scr_d2_absolute_error_score(y_true, y_pred):
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