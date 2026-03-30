import streamlit as st

def increment_idx(max_val, key_name):
    if st.session_state[key_name] < max_val:
        st.session_state[key_name] += 1
    elif st.session_state[key_name] > max_val:
        # Emergency reset if data shrunk unexpectedly
        st.session_state[key_name] = max_val

def decrement_idx(key_name):
    if st.session_state[key_name] > 0:
        st.session_state[key_name] -= 1
        