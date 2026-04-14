import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

class MetabolicTest:
    def __init__(self, test_file):       
        self.test_file = test_file
        self.raw_df = None
        self.exercise_df = None
        self.exercise_df_edited = None
        self.rer_cutoff_vo2 = None
        
        all_headers = pd.read_excel(self.test_file, header=0, nrows=0).columns.tolist()
        data_df = pd.read_excel(self.test_file, skiprows=2, names=all_headers)
        
        # Extract and Rename Columns
        target_cols = ["Time", "t", "Rf", "Ve", "VE", "VO2", "VCO2", "RQ", "VE/VO2", "VE/VCO2", "HR", "Phase", "PetO2", "PetCO2", "Fat"]
        available_cols = [c for c in target_cols if c in data_df.columns]
        self.raw_df = data_df[available_cols].copy()
        
        variable_dict = {'RQ': 'RER', 't': 'Time', 'VE': 'Ve'}
        for k, v in variable_dict.items():
            if k in self.raw_df.columns:
                self.raw_df.rename(columns={k: v}, inplace=True)
        
        # Remove any extra rows
        self.raw_df = self.raw_df.dropna(subset=['Time']).reset_index(drop=True)
        
        # Convert VO2 and VCO2 to L/min
        self.raw_df['VO2'] = self.raw_df['VO2'] / 1000
        self.raw_df['VCO2'] = self.raw_df['VCO2'] / 1000
        
        # excess CO2 and Excess VE
        self.raw_df['excess_co2'] = self.raw_df["VCO2"]**2 / (self.raw_df["VO2"] + 1e-8) - self.raw_df["VCO2"]
        self.raw_df['excess_Ve'] = self.raw_df["Ve"]**2 / (self.raw_df["VCO2"] + 1e-8) - self.raw_df["Ve"]
        
        # Filter for Exercise phase
        self.exercise_df = self.raw_df[self.raw_df['Phase'].astype(str).str.contains('Exercise', case=False, na=False)].copy()
        
        # Find RER cutoff VO2
        if 'RER' in self.exercise_df.columns:
            below_one = self.exercise_df[self.exercise_df['RER'] < 1.0]
            if not below_one.empty:
                self.rer_cutoff_vo2 = below_one.iloc[-1]['VO2']    
        
        # Initial Sort and Trim
        self.exercise_df = self.exercise_df.sort_values(by='VO2').reset_index(drop=True)
        if not self.exercise_df.empty and 'RER' in self.exercise_df.columns:
            min_rer_idx = self.exercise_df['RER'].idxmin()
            self.exercise_df = self.exercise_df.loc[min_rer_idx:].copy()
            self.exercise_df = self.exercise_df[self.exercise_df['RER'] <= 1.07].reset_index(drop=True)
        
        # Calculated normalized errors for each method
        self.exercise_df['FatMaxMask'] = (self.exercise_df.index <= self.exercise_df['Fat'].idxmax()).astype(float)
        self.exercise_df['RER>1.0Mask'] = (self.exercise_df['VO2'] > self.rer_cutoff_vo2).astype(float) if self.rer_cutoff_vo2 else 0.0
        self.exercise_df['RER=0.85'] = self._normalize_errors((self.exercise_df['RER'] - 0.85).abs())
        self.exercise_df['V-Slope'] = self._detect_vt1_vslope_1986(self.exercise_df)
        self.exercise_df['VCO2vs.VO2'] = self._segmented_regression('VCO2', self.exercise_df)
        self.exercise_df['VE/VO2vs.VO2'] = self._segmented_regression('VE/VO2', self.exercise_df)
        self.exercise_df['ExcessCO2vs.VO2'] = self._segmented_regression('excess_co2', self.exercise_df)
        if 'PetO2' in self.exercise_df.columns:
            self.exercise_df['PetO2vs.VO2'] = self._segmented_regression('PetO2', self.exercise_df)
        
        self.exercise_df_edited = self.exercise_df.copy()
    
    def smooth_series(self, series, filter_type, smooth_val):
        if filter_type == 'None':
            return series
        if filter_type == 'Rolling':
            if smooth_val <= 0:
                smooth_val = 3
            return series.rolling(window=int(smooth_val), center=True, min_periods=1).mean()
        if filter_type == 'Gaussian':
            if smooth_val <= 0:
                smooth_val = 0.25
            return pd.Series(gaussian_filter1d(series.values, sigma=smooth_val), index=series.index)
        return series
    
    def _uniform_weights(self, method_cols):
        """Equal weight for every method in the combination."""
        w = 1.0 / len(method_cols)
        return {col: w for col in method_cols} 

    def compute_bayesian_ensemble(self, 
                                  method_cols, 
                                  smooth_type, 
                                  smooth_val, 
                                  smooth_scope, 
                                  T, 
                                  weights=None):
        self.exercise_df_edited = self.exercise_df.copy() 
        df = self.exercise_df_edited        
        avail_method_cols = [col for col in method_cols if col in df.columns]
        
        if not avail_method_cols:
            return np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
        
        # Optional: Individual smoothing of MSE errors
        if smooth_type != "None" and smooth_scope in ['Individual', 'Both (Individual + Average)']:
            cols_to_smooth = [col for col in avail_method_cols if 'Mask' not in col]
            for col in cols_to_smooth:
                df[col] = self.smooth_series(df[col], smooth_type, smooth_val)
        
        if weights is None:
            weights = self._uniform_weights(avail_method_cols)
        
        # Apply negative log softmax with temperature
        log_probs_list = []
        for col in avail_method_cols:
            z = -df[col].values.astype(float) / T  # Higher error -> more negative value
            finite_mask = np.isfinite(z)
            if finite_mask.any():
                lse = logsumexp(z[finite_mask])
                log_p = np.full(len(df), -np.inf)
                log_p[finite_mask] = z[finite_mask] - lse
                log_probs_list.append(log_p * weights.get(col, 0.0))

        if not log_probs_list:
            return np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))

        # Sum results for each method: i.e., Product of Experts
        combined_log_p = np.sum(log_probs_list, axis=0)
        
        # Optional: Smooth the combined log-result
        if smooth_type != "None" and smooth_scope in ['Average', 'Both (Individual + Average)']:
            valid_mask = np.isfinite(combined_log_p)
            if valid_mask.any():
                # Fill -inf with a very small number for smoothing instead of 0 
                # to avoid artificial probability spikes at the edges
                fill_val = np.nanmin(combined_log_p[valid_mask]) - 10 
                temp_series = pd.Series(np.where(valid_mask, combined_log_p, fill_val))
                smooth_vals = self.smooth_series(temp_series, smooth_type, smooth_val).values
                combined_log_p = np.where(valid_mask, smooth_vals, -np.inf)
        
        # Exponentiate and Normalize
        finite_final = np.isfinite(combined_log_p)
        if not finite_final.any():
            return combined_log_p, np.zeros(len(df)), np.zeros(len(df))
        final_lse = logsumexp(combined_log_p[finite_final])
        posterior = np.where(finite_final, np.exp(combined_log_p - final_lse), 0.0)
        cdf = np.nancumsum(posterior)
        
        return combined_log_p, posterior, cdf

    def _normalize_errors(self, results):
        res_arr = np.array(results)
        valid = res_arr[res_arr >= 0]
        if len(valid) == 0: return [0.0] * len(results)
        e_min, e_max = np.min(valid), np.max(valid)
        processed = np.where(res_arr >= 0, res_arr, e_max)
        if e_max > e_min:
            return ((processed - e_min) / (e_max - e_min)).tolist()
        return [0.0] * len(results)
    
    def _segmented_regression(self, y_name, df):
        # Reshape data
        x = df["VO2"].values.reshape(-1, 1)
        y = df[y_name].values.reshape(-1, 1)      
        
        # Iterate possible breakpoints
        results = []
        for i in range(len(x)):
            # Skip first and last four data points
            if (i < 4) or (i > len(x) - 5):
                results.append(-1.0)
                continue
                
            # Split data into two groups
            x1, y1 = x[:i], y[:i]
            x2, y2 = x[i:], y[i:]
            
            # Fit lines
            model1 = LinearRegression().fit(x1, y1)
            model2 = LinearRegression().fit(x2, y2)
            error = mean_squared_error(y1, model1.predict(x1)) + mean_squared_error(y2, model2.predict(x2))
            results.append(error)

        return self._normalize_errors(results)
    
    def _detect_vt1_vslope_1986(self, df):
        """
        Detects VT1 using the V-Slope method described by:
        Beaver, W. L., Karlman Wasserman, and B. J. Whipp. A New Method for Detecting Anaerobic Threshold by Gas Exchange. 
        Journal of Applied Physiology, vol. 60, no. 6, 1986, pp. 2020–2027, doi:10.1152/jappl.1986.60.6.2020.
        """        
        # Reshape data
        vo2 = df['VO2'].values.reshape(-1, 1)
        vco2 = df['VCO2'].values.reshape(-1, 1)

        # Fit single regression line through all data
        global_regr = LinearRegression().fit(vo2, vco2)
        global_regr_vco2 = global_regr.predict(vo2)
        m_global = global_regr.coef_[0][0]
        b_global = global_regr.intercept_[0]

        # Fit regression lines above and below all possible threshold points
        results = []
        for i in range(len(vco2)):
            # Skip first and last four data points and points above the global regression line
            if (i < 4) or (i > len(vco2) - 5) or vco2[i] > global_regr_vco2[i]:
                results.append(-1.0)
                continue
            
            # Get VO2 & VCO2 above and below the threshold point
            vo2_below  = vo2[:i]
            vco2_below = vco2[:i]
            vo2_above  = vo2[i:] 
            vco2_above = vco2[i:]

            # Fit regression lines above and below threshold point
            regr_below = LinearRegression().fit(vo2_below, vco2_below)
            regr_above = LinearRegression().fit(vo2_above, vco2_above)
            
            # Line equation: y = m*x + b
            m_below = regr_below.coef_[0][0]
            b_below = regr_below.intercept_[0]
            m_above = regr_above.coef_[0][0]
            b_above = regr_above.intercept_[0]

            # Upper regression line must be steeper than the lower regression line by >0.1
            if (m_above - m_below) <= 0.1:
                results.append(-1.0)
                continue
                
            # Find intersection of the lower and upper regression lines
            vo2_int = (b_above - b_below) / (m_below - m_above)
            vco2_int = m_below * vo2_int + b_below
            
            # Find distance from intersection of lower/upper regression lines to the global regression line
            distance = abs(m_global * vo2_int - vco2_int + b_global) / np.sqrt(m_global**2 + 1)
            
            # Calculate RMSE for lower and upper regression lines
            rmse_below = np.sqrt(mean_squared_error(vco2_below, regr_below.predict(vo2_below)))
            rmse_above = np.sqrt(mean_squared_error(vco2_above, regr_above.predict(vo2_above)))
            
            # Store the error: distance / sum of RMSE
            results.append(distance / (rmse_below + rmse_above + 1e-8))
        
        return self._normalize_errors(results)
        