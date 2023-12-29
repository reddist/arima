import numpy as np
from scipy.optimize import minimize


class MyARIMA:
    def __init__(self, data, order):
        self.p, self.d, self.q = order
        self.data = data
        self.diff_start_values = np.zeros(self.d)
        self.diff_data = self.difference(data, n=self.d)
        self.params = np.zeros(self.p + self.q)
        self.fittedvalues = np.zeros(len(data))
        self.e = np.zeros(len(self.diff_data))
        self.sigma2 = 0

    def difference(self, data, n=1):
        diff_data = np.array(data)
        for i in range(n):
            self.diff_start_values[i] = diff_data[0]
            diff_data = np.diff(diff_data)
        return diff_data

    def undifference(self, diff_data, n=1):
        undiff_data = diff_data
        for i in range(n):
            undiff_data = np.concatenate(([self.diff_start_values[n - i - 1]], undiff_data)).cumsum()
        return undiff_data

    def calc_arma(self, ar_params, ma_params, data, e, t, with_e_t=False):
        ar_t = np.dot(ar_params[::-1], data[t - self.p:t]) if t >= self.p else np.dot(ar_params[:t][::-1], data[:t])
        ma_t = np.dot(ma_params[::-1], e[t - self.q:t]) if t >= self.q else np.dot(ma_params[:t][::-1], e[:t])
        e_t = e[t] if with_e_t else 0
        return ar_t + ma_t + e_t

    def fit(self):
        def optimScoreFunc(params, data):
            ar_params = params[:self.p]
            ma_params = params[self.p:]
            n = len(data)
            e = np.zeros(n)

            # Calculate residuals
            for i in range(n):
                y_i_pred = self.calc_arma(ar_params, ma_params, data, e, i)
                e[i] = data[i] - y_i_pred

            self.e = e
            self.sigma2 = np.sum(e**2) / len(data)

            # Calculate score (negative log-likelihood)
            # score = 0.5 * np.sum(e ** 2)
            score = 0.5 * (len(data) * np.log(2 * np.pi * self.sigma2) + np.sum(e**2) / self.sigma2)
            return score

        initial_params = np.zeros(self.p + self.q)
        result = minimize(optimScoreFunc,
                          initial_params,
                          method='L-BFGS-B',
                          args=(self.diff_data),
                          bounds=np.full((self.p + self.q, 2), [-1.5, 1.5]))

        if result.success:
            self.params = result.x
            # print(self.params[:self.p])
            # print(self.params[self.p:])
            # print(self.sigma2)

            # Calculate fittedValues
            ar_params = self.params[:self.p]
            ma_params = self.params[self.p:]
            diff_len = len(self.diff_data)
            diff_predicted = np.zeros(diff_len)
            for i in range(diff_len):
                diff_predicted[i] = self.calc_arma(ar_params, ma_params, diff_predicted, self.e, i, True)
            self.fittedvalues = self.undifference(diff_predicted, n=self.d)
        else:
            raise ValueError("Model fitting did not converge.")

    def forecast(self, steps):
        ar_params = self.params[:self.p]
        ma_params = self.params[self.p:]
        n = self.p + self.q

        # Use the last n values from the differenced data to initialize the forecast
        forecast_diff = np.zeros(n)
        forecast_diff[:n] = self.diff_data[-n:]

        # Generate forecast for the remaining steps
        for i in range(n, steps + n):
            forecast_diff = np.append(
                forecast_diff,
                self.calc_arma(ar_params, ma_params, forecast_diff, self.e, i, True)
            )

        # Reverse the differencing to obtain the final forecast
        forecast = self.undifference(np.concatenate((self.diff_data, forecast_diff[n:])), n=self.d)[len(self.data):]

        return forecast
