import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from itertools import product
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import add_months
from arima import MyARIMA

df = pd.read_csv('dataset.csv', index_col=['Book ID'])

sorted_by_date_df = df.sort_values(['Date'])

arr = np.array(sorted_by_date_df['Date'])
filter_arr = []
for element in arr:
    try:
        filter_arr.append(not math.isnan(element))
    except TypeError:
        filter_arr.append(True)
arr = arr[filter_arr]

arr = np.array([datetime.strptime(el, '%Y-%m-%d') for el in arr])


def count_bestsellers(arr, month_step, should_plot=True):
    min_month = arr[0].replace(day=1)
    end_month = add_months(min_month, month_step)

    count_of_bestsellers_by_month = []
    months = []

    current_count = 0
    for element in arr:
        if min_month < element < end_month:
            current_count += 1
        else:
            while element > end_month:
                count_of_bestsellers_by_month.append(current_count)
                months.append(end_month)
                current_count = 0
                min_month = add_months(min_month, month_step)
                end_month = add_months(end_month, month_step)
            current_count = 1
    if current_count > 0:
        count_of_bestsellers_by_month.append(current_count)
        months.append(end_month)

    if should_plot:
        plt.figure(figsize=(20, 9))
        plt.plot(count_of_bestsellers_by_month)
        plt.xticks(rotation='vertical')
        plt.xlabel('Segment End Date', rotation=0, labelpad=30)
        plt.ylabel('Count', rotation=0, labelpad=30)
        plt.title('Best sellers by month segments, month step={}'.format(month_step))

    return months, count_of_bestsellers_by_month


months, bestsellers_count = count_bestsellers(arr, 2, False)

# plt.figure()
# plt.plot(bestsellers_count)

shift_5_arr = np.full(len(bestsellers_count), 5)
data = bestsellers_count + shift_5_arr
data_df = pd.DataFrame({'Original': bestsellers_count}, index=months)

log_data = np.log10(data)
data_df['Log_Data'] = log_data
data_df['Log_Diff_Data'] = np.concatenate(([log_data[0]], np.diff(log_data)))

# plt.figure(figsize=(10, 5))
# plt.title('Log Data')
# plt.plot(data_df['Log_Data'])
# plt.figure(figsize=(10, 5))
# plt.title('Log Diff Data')
# plt.plot(data_df['Log_Diff_Data'])

# decomp = seasonal_decompose(data_df['Log_Diff_Data'])
# plt.figure()
# plt.title('Trend')
# plt.plot(decomp.trend)
# plt.figure()
# plt.title('Seasonal')
# plt.plot(decomp.seasonal)
# plt.figure()
# plt.title('Residual')
# plt.plot(decomp.resid)

# p_value = adfuller(data_df['Log_Diff_Data'])[1]
# rounded_p_value = round(p_value, 4)
# print(f'Полученный уровень значимости (p-value): {rounded_p_value}.')
# if rounded_p_value > 0.05:
#     print(f'{rounded_p_value} > 0.05. Ряд не стационарен.')
# else:
#     print(f'{rounded_p_value} < 0.05. Ряд стационарен!')


plot_acf(data_df['Log_Diff_Data'], lags=50)
plt.show()

def unlog_unshift(data):
    shift_arr = np.full(len(data), 5)
    unshifted_data = 10 ** data - shift_arr
    return np.array([max(val, 0) for val in unshifted_data])


train_data_len = int(len(data_df['Log_Data']) * 0.85)
train, test = data_df['Log_Data'][:train_data_len], data_df['Log_Data'][train_data_len:]


####################################
# Using SARIMAX
# ----------------------------------
d = 1
qs = [5]
ps = [5]
parameters = product(ps, qs)
parameters_list = list(parameters)
len(parameters_list)


results = []
best_aic = float("inf")
best_model = None

for param in tqdm(parameters_list):
    try:
        model = sm.tsa.statespace.SARIMAX(
            train,
            order=(param[0], d, param[1]),
            seasonal_order=(0, 0, 0, 0)
        ).fit(disp=-1, method='lbfgs')
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
    results.append([param, model.aic])


result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())

sarimax_forecast = best_model.forecast(len(test))

print(best_model.summary())



# results_my = []
# best_aic_my = float("inf")
# best_model_my = None

# for param in tqdm(parameters_list):
#     try:
#         model = MyARIMA(
#             data_df['Log_Value'],
#             order=(param[0], d, param[1]),
#         )
#         model.fit()
#     except ValueError:
#         print('wrong parameters:', param)
#         continue
#     if param[0] == param[1] == 5:
#         best_model_my = model
#         # best_aic_my = aic
#     results_my.append([param, 0])


####################################
# Using MyARIMA
# ----------------------------------
my_model = MyARIMA(
    train,
    order=(5, 1, 5),
)
my_model.fit()

my_model_forecast = my_model.forecast(len(test))

# Графики
plt.figure()
plt.plot(train.index, unlog_unshift(train), label='Train', color='blue')
plt.plot(train.index, unlog_unshift(best_model.fittedvalues), label='SARIMAX fitted Train', color='red')
plt.plot(test.index, unlog_unshift(test), label='Test', color='orange')
plt.plot(test.index, unlog_unshift(sarimax_forecast), label='SARIMAX Predicted', color='red', linestyle='--')
plt.title('SARIMAX model')
plt.xlabel('Date')
plt.ylabel('Count of Bestsellers for past month')
plt.legend()
plt.show()

plt.figure()
plt.plot(train.index, unlog_unshift(train), label='Train', color='blue')
plt.plot(train.index, unlog_unshift(my_model.fittedvalues), label='MyARIMA fitted Train', color='purple')
plt.plot(test.index, unlog_unshift(test), label='Test', color='orange')
plt.plot(test.index, unlog_unshift(my_model_forecast), label='MyARIMA Predicted', color='purple', linestyle='--')
plt.title('MyARIMA model')
plt.xlabel('Date')
plt.ylabel('Count of Bestsellers for past month')
plt.legend()
plt.show()


##################################################
# Метрики
# ----------------------
def calc_metrics(data, data_pred):
    mae = mean_absolute_error(data, data_pred)
    rmse = np.sqrt(mean_squared_error(data, data_pred))
    mape = np.mean(np.abs((data - data_pred) / data)) * 100
    return mae, rmse, mape


mae_custom, rmse_custom, mape_custom = calc_metrics(unlog_unshift(test), unlog_unshift(my_model_forecast))
print(f'----- Custom Model -----')
print(f'MAE: {np.round(mae_custom, 4)}')
print(f'RMSE: {np.round(rmse_custom, 4)}')
print(f'MAPE: {np.round(mape_custom, 4)} %')
print()

mae_prophet, rmse_prophet, mape_prophet = calc_metrics(unlog_unshift(test), unlog_unshift(sarimax_forecast))
print(f'----- SARIMAX -----')
print(f'MAE: {np.round(mae_prophet, 4)}')
print(f'RMSE: {np.round(rmse_prophet, 4)}')
print(f'MAPE: {np.round(mape_prophet, 4)} %')
