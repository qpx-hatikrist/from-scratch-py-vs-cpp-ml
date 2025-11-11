import math
from typing import Sequence

def scr_root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    '''
    Compute Root Mean Squared Error (RMSE).
    
    Вычисляет корень из средней квадратичной ошибки (RMSE).
    
    Formula | Формула:
        RMSE = sqrt( (1 / n) * sum( (y_true - y_pred)^2 ) )

    Equivalent to | Эквивалентно:
        sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)
    '''
    rmse = (sum((y_t - y_p)**2 for y_t, y_p in zip(y_true, y_pred)) / len(y_true)) **0.5
    return float(rmse)

def scr_root_mean_squared_log_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    '''
    Compute Root Mean Squared Logarithmic Error (RMSLE).

    Uses log(y + 1). Assumes y_true and y_pred are >= 0.

    Equivalent to | Эквивалентно:
        math.sqrt(sklearn.metrics.mean_squared_log_error(y_true, y_pred))
        
    Вычисляет корень из средней квадратичной логарифмической ошибки (RMSLE).
    
    Использует log(y + 1). Предполагается, что y_true и y_pred >= 0.
    '''
    rmsle = (sum((math.log(y_t + 1) - math.log(y_p + 1))**2 for y_t, y_p in zip(y_true, y_pred)) / len(y_true)) **0.5
    return float(rmsle)

def scr_r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    '''
    Compute coefficient of determination R².
    
    Вычисляет коэффициент детерминации R².

    Formula | Формула:
    R² = 1 - SS_res / SS_tot.
    
    Equivalent to | Эквивалентно:
        sklearn.metrics.r2_score(y_true, y_pred)
    '''
    mean = sum(y_true) / len(y_true)
    ssr = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    sst = sum((y_t - mean) ** 2 for y_t in y_true)
    
    if sst == 0:
        return 0.0
    
    return float(1 - ssr / sst)

def scr_d2_absolute_error_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    '''
    Compute D² score based on absolute error.
    
    Вычисляет метрику D² на основе абсолютной ошибки.

    Formula | Формула:
    D² = 1 - sum(|y_true - y_pred|) / sum(|y_true - median(y_true)|)
    
    Equivalent to | Эквивалентно:
        sklearn.metrics.d2_absolute_error_score(y_true, y_pred)
    '''
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

def scr_mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    '''
    Compute Mean Absolute Error (MAE).

    Average absolute difference between y_true and y_pred.
    
    Equivalent to | Эквивалентно:
        sklearn.metrics.mean_absolute_error(y_true, y_pred)
        
    Вычисляет среднюю абсолютную ошибку (MAE).
    
    Среднее значение |y_true - y_pred|.
    '''
    mae = (sum(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred))) / len(y_true)
    return float(mae)

def scr_symmetric_mean_absolute_percentage_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    '''
    Compute Symmetric Mean Absolute Percentage Error (sMAPE) in %.

    Formula | Формула:
    sMAPE = (100 / n) * sum( 2 * |y - y_hat| / (|y| + |y_hat|) ).
    
    Note: there is no direct equivalent in sklearn.
    
    Вычисляет симметричную среднюю абсолютную процентную ошибку (sMAPE) в процентах.
    
    Прямого эквивалента в sklearn нет.
    '''
    n = len(y_true)
    smape_sum = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        denom = abs(y_t) + abs(y_p)
        if denom == 0:
            continue
        smape_sum += 2 * abs(y_t - y_p) / denom

    return float(100 * smape_sum / n)

def scr_max_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    '''
    Compute max error.

    Maximum absolute difference between y_true and y_pred.
    
    Equivalent to | Эквивалентно:
        sklearn.metrics.max_error(y_true, y_pred)
        
    Вычисляет максимальную ошибку (max error).
    
    Максимальное значение |y_true - y_pred| по всем объектам.
    '''
    max_error = max(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred))
    
    return float(max_error)

def scr_median_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    '''
    Compute Median Absolute Error (MedAE).

    Median of absolute errors |y_true - y_pred|.
    
    Equivalent to | Эквивалентно:
        sklearn.metrics.median_absolute_error(y_true, y_pred)
        
    Вычисляет медианную абсолютную ошибку (MedAE).
    
    Медиана значений |y_true - y_pred|.
    '''
    y_sorted = sorted(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred))
    
    if len(y_sorted) % 2 == 1:
        median = y_sorted[len(y_sorted) // 2]
    else:
        median = (y_sorted[len(y_sorted) // 2] + y_sorted[len(y_sorted)  // 2 - 1]) / 2
    
    return float(median)

def scr_explained_variance_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    '''
    Compute explained variance score (EVS).
    
    Вычисляет долю объяснённой дисперсии (Explained Variance Score, EVS).

    Formula | Формула:
    EVS = 1 - Var(y - y_hat) / Var(y).
    
    Equivalent to | Эквивалентно:
        sklearn.metrics.explained_variance_score(y_true, y_pred)
    '''
    n = len(y_true)

    mean_y = sum(y_true) / n
    var_y = sum((y - mean_y) ** 2 for y in y_true) / n

    if var_y == 0:
        return 0.0

    errors = [y_t - y_p for y_t, y_p in zip(y_true, y_pred)]
    mean_err = sum(errors) / n
    var_err = sum((e - mean_err) ** 2 for e in errors) / n

    return float(1 - var_err / var_y)