import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from MetabolicTest import MetabolicTest
from helpers import increment_idx, decrement_idx

def load_threshold_tab():
    if 'met_test' in st.session_state:
        populate_threshold_tab(st.session_state['met_test'])
    else:
        st.warning("Please upload a file.")

def apply_ensemble_logic(met_test):    
    active_smooth = st.session_state.get('active_smoothing', 'None')
    smooth_val = st.session_state.get('roll_val' if active_smooth == "Rolling" else 'gauss_val', 0)
    smooth_scope = st.session_state.get('smooth_scope')
    
    # Selected threshold methods
    all_selected = st.session_state.get('selected_options', [])
    
    # Ensure selected methods exist in the dataset
    all_selected = [col for col in all_selected if col in met_test.exercise_df.columns]
    
    # Remove masks for smoothing
    all_selected_no_mask = [item for item in all_selected if "Mask" not in item]
    
    if smooth_scope == "Both (Individual + Average)":
        # This updates met_test.exercise_df_edited and all its error columns
        met_test.apply_smoothing(active_smooth, smooth_val, all_selected_no_mask)
        df = met_test.exercise_df_edited
    else:
        met_test.exercise_df_edited = met_test.exercise_df.copy()
        df = met_test.exercise_df_edited

    df['Ensemble_Error'] = df[all_selected].mean(axis=1)

    # Final Smoothing pass on the Ensemble error
    if active_smooth == "Rolling":
        df['Ensemble_Error'] = df['Ensemble_Error'].rolling(window=int(st.session_state.roll_val), center=True).mean().ffill().bfill()
    elif active_smooth == "Gaussian":
        df['Ensemble_Error'] = gaussian_filter1d(df['Ensemble_Error'], sigma=float(st.session_state.gauss_val))
    
    df['Ensemble_Error'] = met_test._normalize_errors(df['Ensemble_Error'])
    
def populate_threshold_tab(met_test):
    # --- INITIALIZE DEFAULTS ---
    if 'active_smoothing' not in st.session_state:
        st.session_state.active_smoothing = "Gaussian"
    if 'smooth_scope' not in st.session_state:
        st.session_state.smooth_scope = "Both (Individual + Average)"
    if 'use_gauss' not in st.session_state:
        st.session_state.use_gauss = True
    if 'use_roll' not in st.session_state:
        st.session_state.use_roll = False
    
    # Keep track of the settings for the ensemble method
    if 'last_settings_hash' not in st.session_state:
        st.session_state.last_settings_hash = None
    
    # --- UI CONTROLS ------------------------------------------------------------------------------------------------
    c1, c2 = st.columns(2)
    
    with c1:
        use_roll = st.checkbox("Rolling Average", key="use_roll", disabled=st.session_state.get("use_gauss", False))
        use_gauss = st.checkbox("Gaussian Smooth", key="use_gauss", disabled=st.session_state.get("use_roll", False))
        
        # Sync state
        if use_roll: st.session_state.active_smoothing = "Rolling"
        elif use_gauss: st.session_state.active_smoothing = "Gaussian"
        else: st.session_state.active_smoothing = "None"

        if use_roll: st.select_slider("Window Size", [3, 5, 7, 9, 11], 5, key="roll_val")
        if use_gauss: st.slider("Sigma (Width)", 0.25, 3.0, 1.0, 0.25, key="gauss_val")
        if use_roll or use_gauss: st.radio("Apply Smoothing To:", ["Final Average Only", "Both (Individual + Average)"], key="smooth_scope", index=0)
        
    with c2:
        # Match names to the keys created in MetabolicTest.update_error_values
        opts = ["FatMaxMask", "RER>1.0Mask", "V-Slope", "VCO2vs.VO2", "VE/VO2vs.VO2", "ExcessCO2vs.VO2", "RER=0.85", "PetO2vs.VO2"]
        default_opts = ["FatMaxMask", "RER>1.0Mask", "VCO2vs.VO2", "ExcessCO2vs.VO2", "RER=0.85"]
        st.multiselect("Ensemble Components", options=opts, default=default_opts, key="selected_options")
        
    st.divider()
    
    # Generate a "hash" of current settings to see if we should auto-recalculate
    current_settings = (
        getattr(met_test, 'test_file', None),
        st.session_state.active_smoothing,
        st.session_state.get('roll_val', 0),
        st.session_state.get('gauss_val', 0),
        st.session_state.selected_options,
        st.session_state.get('smooth_scope')
    )
    
    apply_ensemble_logic(met_test)
    df = met_test.exercise_df_edited
    
    settings_changed = current_settings != st.session_state.last_settings_hash
    
    if 'manual_vt1_idx' not in st.session_state or settings_changed:
        best_vt1_label = df['Ensemble_Error'].idxmin()
        best_vt2_label = int(len(df)/3*2)
        st.session_state.manual_vt1_idx = df.index.get_loc(best_vt1_label)
        st.session_state.manual_vt2_idx = best_vt2_label
        st.session_state.last_settings_hash = current_settings
    
    # Helper for safe plotting values
    vt1_idx = st.session_state.manual_vt1_idx
    vt2_idx = st.session_state.manual_vt2_idx
    
    # --- 
        
    # Navigation Buttons
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    # VT1
    with col1:
        st.markdown("## 🟢 VT1")
        st.button("⬅️ Previous", on_click=decrement_idx, args=("manual_vt1_idx",), key="manual_vt1_prev")
        st.button("Next ➡️", on_click=increment_idx, args=(len(df) - 1, "manual_vt1_idx"), key="manual_vt1_next")
    with col2:
        st.write(f"**Time:** {str(df.iloc[vt1_idx]['Time']).split('days ')[-1].split('.')[0]}")
        st.write(f"**VO2:** {df.iloc[vt1_idx]['VO2']:.3f} L/min")
        st.write(f"**% VO2Max:** {((df.iloc[vt1_idx]['VO2'] / df["VO2"].max()) * 100):.1f} %")
        st.write(f"**VCO2:** {df.iloc[vt1_idx]['VCO2']:.3f} L/min")
    with col3:
        st.write(f"**RER:** {df.iloc[vt1_idx]['VCO2'] / df.iloc[vt1_idx]["VO2"]:.2f}")
        st.write(f"**HR:** {df.iloc[vt1_idx]['HR']:.0f}")
        st.write(f"**% HRMax:** {((df.iloc[vt1_idx]['HR'] / df["HR"].max()) * 100):.1f} %")
    
    # VT2
    with col4:
        st.markdown("## 🟠 VT2")
        st.button("⬅️ Previous", on_click=decrement_idx, args=("manual_vt2_idx",), key="manual_vt2_prev")
        st.button("Next ➡️", on_click=increment_idx, args=(len(df) - 1, "manual_vt2_idx"), key="manual_vt2_next")
    with col5:
        st.write(f"**Time:** {str(df.iloc[vt2_idx]['Time']).split('days ')[-1].split('.')[0]}")
        st.write(f"**VO2:** {df.iloc[vt2_idx]['VO2']:.3f} L/min")
        st.write(f"**% VO2Max:** {((df.iloc[vt2_idx]['VO2'] / df["VO2"].max()) * 100):.1f} %")
        st.write(f"**VCO2:** {df.iloc[vt2_idx]['VCO2']:.3f} L/min")
    with col6:
        st.write(f"**RER:** {df.iloc[vt2_idx]['VCO2'] / df.iloc[vt2_idx]["VO2"]:.2f}")
        st.write(f"**HR:** {df.iloc[vt2_idx]['HR']:.0f}")
        st.write(f"**% HRMax:** {((df.iloc[vt2_idx]['HR'] / df["HR"].max()) * 100):.1f} %")
    
    # --- PLOTTING ------------------------------------------------------------------------------------------------------
    DATA_SIZE = 8
    INDICATOR_SIZE = 10
    
    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=("V-Slope (VCO2 vs VO2)", "Ventilatory Equivalents", "Excess CO2", "End-Tidal Gases", "Ensemble Error Profile"),
                        horizontal_spacing=0.08, vertical_spacing=0.08,
                        # Enable secondary_y for the rows where you want two axes
                        specs=[[{}, {}], [{"secondary_y": True}, {"secondary_y": True}], [{"colspan": 2}, None]])
    
    # --- ROW 1, COL 1: V-SLOPE -----------------------------------------------------------------------------------------    
    # VCO2 vs. VO2
    fig.add_trace(go.Scatter(x=df['VO2'], 
                             y=df['VCO2'], 
                             mode='markers', 
                             name='VCO2 vs. VO2', 
                             marker=dict(color='gray', size=DATA_SIZE, opacity=0.9, line=dict(width=1, color='white'))), 
                  row=1, col=1)
    
    # Global Line
    X = df['VO2'].values.reshape(-1, 1)
    y = df['VCO2'].values
    model_g = LinearRegression().fit(X, y)
    m_g, b_g = model_g.coef_[0], model_g.intercept_
    x_range = np.array([df['VO2'].min(), df['VO2'].max()])
    fig.add_trace(go.Scatter(x=x_range, y=m_g*x_range + b_g, name='Global Regression', line=dict(color='black', dash='dash', width=1.5)), row=1, col=1)
    
    # Segments
    m_low, b_low = np.polyfit(df['VO2'].iloc[:vt1_idx], df['VCO2'].iloc[:vt1_idx], 1)
    m_high, b_high = np.polyfit(df['VO2'].iloc[vt1_idx:], df['VCO2'].iloc[vt1_idx:], 1)
    
    x_seg1 = df['VO2'].iloc[:vt1_idx+1]
    fig.add_trace(go.Scatter(x=x_seg1, y=m_low*x_seg1 + b_low, name='Lower Regression', line=dict(color='blue', width=1.5)), row=1, col=1)
    
    x_seg2 = df['VO2'].iloc[vt1_idx:]
    fig.add_trace(go.Scatter(x=x_seg2, y=m_high*x_seg2 + b_high, name='Upper Regression', line=dict(color='red', width=1.5)), row=1, col=1)
    
    # --- ROW 1, COL 2: Ventilatory Equivalents --------------------------------------------------------------------------
    # VE/VO2
    fig.add_trace(go.Scatter(x=df['VO2'], 
                             y=df['VE/VO2'], 
                             mode='lines+markers', 
                             name='VE/VO2', 
                             marker=dict(color='blue', size=DATA_SIZE, opacity=0.9, line=dict(width=1.5, color='white')), 
                             line=dict(width=0.5, color='blue')), 
                  row=1, col=2)
    
    # VE/VCO2
    fig.add_trace(go.Scatter(x=df['VO2'], 
                             y=df['VE/VCO2'], 
                             mode='lines+markers', 
                             name='VE/VCO2', 
                             marker=dict(color='red', size=DATA_SIZE, opacity=0.9, line=dict(width=1.5, color='white')), 
                             line=dict(width=0.5, color='red')), 
                  row=1, col=2)
    
    # Set range
    combined_ve_data = pd.concat([df['VE/VO2'], df['VE/VCO2']])
    vent_eq_min = 20
    while vent_eq_min >= combined_ve_data.min():
        vent_eq_min -= 5
    vent_eq_max = 35    
    while vent_eq_max <= combined_ve_data.max():
        vent_eq_max += 5    
    fig.update_yaxes(range=[vent_eq_min, vent_eq_max], row=1, col=2)
    
    # --- ROW 2, COL 1: EXCESS CO2 -----------------------------------------------------------------------------------------
    # Excess CO2
    fig.add_trace(go.Scatter(x=df['VO2'], 
                             y=df['excess_co2'], 
                             mode='lines+markers', 
                             name='Excess CO2', 
                             marker=dict(color='gray', size=DATA_SIZE),
                             line=dict(width=0.5, color='gray')),
                 row=2, col=1)
    
    # --- ROW 2, COL 2: End-Tidals -----------------------------------------------------------------------------------------
    
    if "PetO2" in df.columns:
        # --- ROW 2, COL 2: PetO2 & PetCO2 ---
        # PetO2
        fig.add_trace(go.Scatter(x=df['VO2'], 
                                 y=df['PetO2'], 
                                 mode='markers', 
                                 name='PetO2', 
                                 marker=dict(color='blue', size=DATA_SIZE)), 
                     row=2, col=2, secondary_y=False)
        
        # PetCO2
        fig.add_trace(go.Scatter(x=df['VO2'], 
                                 y=df['PetCO2'], 
                                 mode='markers', 
                                 name='PetCO2', 
                                 marker=dict(color='red', size=DATA_SIZE)), 
                     row=2, col=2, secondary_y=True)
    
        # Set range
        peto2_min = 70
        peto2_max = 120
        petco2_min = 0
        petco2_max = 50
        while peto2_min >= df['PetO2'].min():
            peto2_min -= 10
        while peto2_max <= df['PetO2'].max():
            peto2_max += 10    
        while petco2_max <= df['PetCO2'].max():
            petco2_max += 10
        fig.update_yaxes(range=[peto2_min, peto2_max], secondary_y=False, row=2, col=2)
        fig.update_yaxes(range=[petco2_min, petco2_max], secondary_y=True, row=2, col=2)
    
    # --- ROW 3: Ensemble Plot ------------------------------------------------------------------------------------------------
    for m in st.session_state.selected_options:
        if m in df.columns:
            fig.add_trace(go.Scatter(x=df['VO2'], y=df[m], 
                                     name=m, line=dict(dash='dot', width=1), opacity=0.7), row=3, col=1)

    fig.add_trace(go.Scatter(x=df['VO2'], y=df['Ensemble_Error'], 
                             name="ENSEMBLE_ERROR", line=dict(color='black', width=4)), row=3, col=1)

    # Threshold indicator lines
    fig.add_vline(x=df['VO2'].iloc[st.session_state.manual_vt1_idx], line_color="LimeGreen", line_dash="dash", row="all", col="all")
    #fig.add_vline(x=df['VO2'].iloc[st.session_state.manual_vt2_idx], line_color="DarkOrange", line_dash="dash", row="all", col="all")

    # Plot size and layout
    fig.update_layout(height=2000, showlegend=False, template="plotly_white")
    st.plotly_chart(fig, width='stretch')
    