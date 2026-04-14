import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from MetabolicTest import MetabolicTest

def load_threshold_tab():
    if 'met_test' in st.session_state:
        populate_threshold_tab(st.session_state['met_test'])
    else:
        st.warning("Please upload a file.")

def apply_ensemble_logic(met_test):    
    active_smooth = st.session_state.get('smooth_type', 'None')
    smooth_val = st.session_state.get('roll_val' if active_smooth == "Rolling" else 'gauss_val', 0)
    smooth_scope = st.session_state.get('smooth_scope')
    selected_methods = st.session_state.get('selected_options', [])
    T = st.session_state.get('T')
    
    # Compute Bayesian Ensemble
    if selected_methods:
        _, post, cdf = met_test.compute_bayesian_ensemble(selected_methods, active_smooth, smooth_val, smooth_scope, T, weights=None)
        
        st.session_state['posterior'] = post
        st.session_state['cdf'] = cdf
        st.session_state['vt1_MAP_idx'] = int(np.argmax(post))
        st.session_state['vt1_Mean'] = np.nansum(post * met_test.exercise_df_edited['VO2'].values)

def populate_threshold_tab(met_test):
    st.sidebar.header("Threshold Settings")
    
    # Methods to Ensemble
    opts = ["FatMaxMask", "RER>1.0Mask", "V-Slope", "VCO2vs.VO2", "VE/VO2vs.VO2", "ExcessCO2vs.VO2", "RER=0.85", "PetO2vs.VO2"]
    default_opts = ["FatMaxMask", "RER>1.0Mask", "VCO2vs.VO2", "ExcessCO2vs.VO2", "RER=0.85"]
    
    st.sidebar.write("### Methods to Ensemble")
    selected_methods = []
    for opt in opts:
        # This creates a vertical list that is always "tall" and visible
        if st.sidebar.checkbox(opt, value=(opt in default_opts), key=f"check_{opt}"):
            selected_methods.append(opt)

    # Then pass selected_methods to your logic
    st.session_state['selected_options'] = selected_methods
    
    # Smoothing Type
    st.sidebar.radio("Smoothing Type", ["None", "Rolling", "Gaussian"], key="smooth_type", index=2, on_change=apply_ensemble_logic, args=(met_test,))
    
    # Smoothing Value
    if st.session_state.smooth_type == 'Rolling':
        st.sidebar.select_slider("Window Size", [3, 5, 7, 9, 11], 5, key="roll_val")
    elif st.session_state.smooth_type == 'Gaussian':
        st.sidebar.slider("Sigma (Width)", 0.25, 5.0, 2.0, 0.25, key="gauss_val")
        
    # Smoothing Scope
    if st.session_state.smooth_type != 'None':
        st.sidebar.radio("Smoothing Scope", ["Both (Individual + Average)", "Final Average Only"], key="smooth_scope", on_change=apply_ensemble_logic, args=(met_test,))
    
    # Temperature (T)
    st.sidebar.slider("Temperature (T)", 0.005, 0.2, 0.035, 0.005, key="T", format="%.3f", help="Controls how 'peaky' the probability distribution is.")
    
    apply_ensemble_logic(met_test)
    df = met_test.exercise_df_edited
    vt1_MAP_idx = st.session_state.get('vt1_MAP_idx', 0)
    vt1_Mean = st.session_state.get('vt1_Mean', 0)
    selected_methods = st.session_state.get('selected_options', [])

    # --- PLOTTING ---
    DATA_SIZE = 8

    fig = make_subplots(
        rows=4, cols=2, 
        vertical_spacing=0.06,
        row_heights=[1.0, 1.0, 0.7, 0.7],
        specs=[
            [{}, {}], 
            [{}, {"secondary_y": True}], 
            [{"colspan": 2}, None], 
            [{"colspan": 2}, None]
        ],
        subplot_titles=(
            "V-Slope (VCO2 vs VO2)", "Ventilatory Equivalents", 
            "Excess Metrics", "End-Tidal Gases (PetO2/CO2)", 
            "Individual Normalized Error Curves", 
            "Ensemble Consensus: VT1 Location Probability"
        )
    )
    
    
    # V-Slope (Row 1, Col 1)
    fig.add_trace(go.Scatter(x=df['VO2'], y=df['VCO2'], mode='markers', name='Data'), row=1, col=1)
    if vt1_MAP_idx > 2 and (len(df) - vt1_MAP_idx) > 2:
        m1, b1 = np.polyfit(df['VO2'].iloc[:vt1_MAP_idx], df['VCO2'].iloc[:vt1_MAP_idx], 1)
        m2, b2 = np.polyfit(df['VO2'].iloc[vt1_MAP_idx:], df['VCO2'].iloc[vt1_MAP_idx:], 1)
        fig.add_trace(go.Scatter(x=df['VO2'], y=m1*df['VO2']+b1, name='Low Segment'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['VO2'], y=m2*df['VO2']+b2, name='High Segment'), row=1, col=1)

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
    
    # ROW 3: Individual Error Curves (Top Panel from your Matplotlib code)
    for col in selected_methods:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['VO2'], y=df[col],
                mode='lines',
                name=col,
                line=dict(width=1),
                opacity=0.4
            ), row=3, col=1)

    # ROW 4: Bayesian Posterior (Bottom Panel from your Matplotlib code)
    if 'posterior' in st.session_state:
        post = st.session_state['posterior']
        cdf = st.session_state['cdf']
        
        # Main Posterior Curve
        fig.add_trace(go.Scatter(
            x=df['VO2'], y=post, 
            name="Posterior", 
            fill='tozeroy', 
            line=dict(color='steelblue', width=2)
        ), row=4, col=1)
        
        # 90% Credible Interval Shading
        idx_low = np.searchsorted(cdf, 0.05)
        idx_high = min(np.searchsorted(cdf, 0.95), len(df)-1)
        ci_low, ci_high = df['VO2'].iloc[idx_low], df['VO2'].iloc[idx_high]
        
        fig.add_vrect(
            x0=ci_low, x1=ci_high, 
            fillcolor="dodgerblue", opacity=0.2, 
            layer="below", line_width=0, 
            row=4, col=1
        )

        # Global threshold lines
        all_subplots = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1)]
        for r, c in all_subplots:
            # MAP Line (Red)
            fig.add_vline(x=df['VO2'].iloc[vt1_MAP_idx], line_dash="dash", line_color="red", row=r, col=c, annotation_text="MAP")
            # Mean Line (Yellow/Orange)
            fig.add_vline(x=vt1_Mean, line_dash="dot", line_color="orange", row=r, col=c, annotation_text="Mean")

    # Formatting
    fig.update_layout(height=1600, showlegend=False, template="plotly_white")
    fig.update_xaxes(title_text="VO2 (L/min)", row=4, col=1)
    fig.update_yaxes(title_text="Norm. Error", row=3, col=1)
    fig.update_yaxes(title_text="Probability", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    c1, c2 = st.columns(2)
    c1.metric("VT1 MAP (VO2)", f"{df['VO2'].iloc[vt1_MAP_idx]:.3f} L/min")
    c2.metric("VT1 Mean (VO2)", f"{vt1_Mean:.3f} L/min")
    