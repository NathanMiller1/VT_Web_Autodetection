import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d

class MetabolicTest:
    def __init__(self, test_file):       
        self.test_file = test_file
        self.raw_df = None
        self.exercise_df = None
        self.exercise_df_edited = None
        self.load_data()

    def load_data(self):
        all_headers = pd.read_excel(self.test_file, header=0, nrows=0).columns.tolist()
        data_df = pd.read_excel(self.test_file, skiprows=2, names=all_headers)
        
        # Extract and Rename Columns
        target_cols = ["Time", "t", "Rf", "Ve", "VE", "VO2", "VCO2", "RQ", "VE/VO2", "VE/VCO2", "HR", "Phase", "PetO2", "PetCO2", "Fat"]
        available_cols = [c for c in target_cols if c in data_df.columns]
        self.raw_df = data_df[available_cols].copy()
        
        variable_dict = {'RQ': 'RER', 't': 'Time', 'VE': 'Ve'}
        for k, v in variable_dict.items():
            if k in self.raw_df.columns:
                print(f'k:v: {k}: {v}')
                self.raw_df.rename(columns={k: v}, inplace=True)
        
        # Remove any extra rows
        self.raw_df = self.raw_df.dropna(subset=['Time']).reset_index(drop=True)
        
        # Convert VO2 and VCO2 to L/min
        self.raw_df['VO2'] = self.raw_df['VO2'] / 1000
        self.raw_df['VCO2'] = self.raw_df['VCO2'] / 1000
        
        # Calculate excess CO2 and excess VE
        self.raw_df['excess_co2'] = self.raw_df["VCO2"]**2 / (self.raw_df["VO2"] + 1e-8) - self.raw_df["VCO2"]
        self.raw_df['excess_Ve'] = self.raw_df["Ve"]**2 / (self.raw_df["VCO2"] + 1e-8) - self.raw_df["Ve"]
        
        # Filter for Exercise phase
        self.exercise_df = self.raw_df[self.raw_df['Phase'].astype(str).str.contains('Exercise', case=False, na=False)].copy()
        
        # Sort by VO2
        self.exercise_df = self.exercise_df.reset_index().sort_values(by='VO2', ascending=True).reset_index(drop=True)
        
        # Trim by RER
        min_rer_idx = self.exercise_df['RER'].idxmin()
        self.exercise_df = self.exercise_df.loc[min_rer_idx:].copy()
        self.exercise_df = self.exercise_df[self.exercise_df['RER'] <= 1.05].reset_index(drop=True)
        
        # Update error values for the base exercise set
        self._calculate_error_values(self.exercise_df)
        
        # Initialize edited_df
        self.exercise_df_edited = self.exercise_df.copy()
    
    def apply_smoothing(self, method, value, selected_methods):
        assert(method in ["None", "Rolling", "Gaussian"])
        
        # Fresh copy of original data
        df = self.exercise_df.copy()
        
        if method != "None":
            available_cols = [col for col in selected_methods if col in df.columns]
            if method == "Rolling":
                df[available_cols] = df[available_cols].rolling(window=int(value), center=True).mean().ffill().bfill()
            elif method == "Gaussian":
                df[available_cols] = gaussian_filter1d(df[available_cols], sigma=float(value), axis=0)

        self.exercise_df_edited = df.reset_index(drop=True)
    
    def _calculate_error_values(self, df):
        # FatMax Mask
        df['FatMaxMask'] = (df.index <= df['Fat'].idxmax()).astype(float)
        
        # RER > 1.0 Mask
        df['RER>1.0Mask'] = (df['RER'] > 1.0).astype(float)
        
        # RER=0.85
        df['RER=0.85'] = self._normalize_errors((df['RER'] - 0.85).abs())
        
        # V-Slope
        df['V-Slope'] = self._detect_vt1_vslope_1986(df)
        
        # VCO2 vs. VO2
        df['VCO2vs.VO2'] = self._segmented_regression('VCO2', df)
        
        # VE/VO2 vs. VO2
        df['VE/VO2vs.VO2'] = self._segmented_regression('VE/VO2', df)
        
        # Excess CO2
        df['ExcessCO2vs.VO2'] = self._segmented_regression('excess_co2', df)
        
        # PetO2 vs. VO2
        if 'PetO2' in df.columns:
            df['PetO2vs.VO2'] = self._segmented_regression('PetO2', df)
    
    def _normalize_errors(self, results):
        # --- NORMALIZE ERROR (0 to 1) ---
        e_min = min(x for x in results if x > 0)
        e_max = max(results)
        
        # Set skipped rows to max error (first/last 4 rows, V-Slope points above global regression line)
        results = [x if x >= 0 else e_max for x in results]
        
        if e_max > e_min:
            results = [((x - e_min) / (e_max - e_min)) for x in results]
        else:
            # Handle edge case where all errors are equal
            results = [0.0] * len(results)
        
        return results
    
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
        