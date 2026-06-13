import streamlit as st
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fuzzy Game Evaluator", page_icon="🎮", layout="centered")

# --- FUZZY LOGIC FROM NOTEBOOK ---
def trimf(x, abc):
    a, b, c = abc
    if a == b:
        return np.where(x < b, 0.0, np.where(x <= c, (c - x) / (c - b), 0.0))
    if b == c:
        return np.where(x < a, 0.0, np.where(x <= b, (x - a) / (b - a), 0.0))
    first_half = (x - a) / (b - a)
    second_half = (c - x) / (c - b)
    return np.maximum(0, np.minimum(first_half, second_half))

out_x         = np.linspace(0, 10, 500)
out_poor      = trimf(out_x, [0, 0,  4 ])
out_fair      = trimf(out_x, [3, 5,  7 ])
out_excellent = trimf(out_x, [6, 10, 10])

def fuzzify(global_in, avg_rating_in, count_in, platform_in, recent_in):
    return {
        'global': {
            'Poor'    : float(trimf(global_in, [0,   0,   2.5])),
            'Average' : float(trimf(global_in, [1.5, 2.5, 3.5])),
            'High'    : float(trimf(global_in, [3,   5,   5  ])),
        },
        'avg_rating': {
            'Low'    : float(trimf(avg_rating_in, [1, 1, 3])),
            'Medium' : float(trimf(avg_rating_in, [2, 3, 4])),
            'High'   : float(trimf(avg_rating_in, [3, 5, 5])),
        },
        'count': {
            'Few'      : float(trimf(count_in, [0,  0,   40 ])),
            'Moderate' : float(trimf(count_in, [20, 50,  80 ])),
            'Many'     : float(trimf(count_in, [60, 100, 100])),
        },
        'platform': {
            'Narrow' : float(trimf(platform_in, [1, 1,  6 ])),
            'Medium' : float(trimf(platform_in, [3, 7,  12])),
            'Wide'   : float(trimf(platform_in, [8, 22, 22])),
        },
        'recent': {
            'Old'    : float(trimf(recent_in, [0, 0,  4 ])),
            'Recent' : float(trimf(recent_in, [3, 5,  7 ])),
            'New'    : float(trimf(recent_in, [6, 10, 10])),
        },
    }

def fuzzy_and(*args): 
    return min(args)
def fuzzy_or(*args): 
    return max(args)

def rule(fuzzyValues) : 
    poor_rules = [
        fuzzy_or(fuzzyValues['global']['Poor'], fuzzyValues['avg_rating']['Low']),
        fuzzy_and(fuzzyValues['global']['Poor'], fuzzyValues['count']['Few']),
        fuzzy_and(fuzzyValues['avg_rating']['Low'], fuzzyValues['platform']['Narrow']),
        fuzzy_and(fuzzyValues['recent']['Old'], fuzzyValues['count']['Few']),
        fuzzy_and(fuzzyValues['global']['Poor'], fuzzyValues['avg_rating']['Low'], fuzzyValues['recent']['Old']),
        fuzzy_and(fuzzyValues['global']['Poor'], fuzzyValues['platform']['Narrow']),
        fuzzy_and(fuzzyValues['avg_rating']['Low'], fuzzyValues['count']['Few'])
    ]
    decent_rules = [
        fuzzy_and(fuzzyValues['global']['Average'], fuzzyValues['avg_rating']['Medium']),
        fuzzy_and(fuzzyValues['global']['Average'], fuzzyValues['count']['Moderate']),
        fuzzy_and(fuzzyValues['avg_rating']['Medium'], fuzzyValues['count']['Moderate']),
        fuzzy_and(fuzzyValues['global']['High'], fuzzyValues['count']['Few']),
        fuzzy_and(fuzzyValues['avg_rating']['Medium'], fuzzyValues['platform']['Medium']),
        fuzzy_and(fuzzyValues['recent']['Recent'], fuzzyValues['avg_rating']['Medium']),
        fuzzy_and(fuzzyValues['global']['Average'], fuzzyValues['platform']['Medium'])
    ]
    excellent_rules = [
        fuzzy_and(fuzzyValues['global']['High'], fuzzyValues['avg_rating']['High'], fuzzyValues['count']['Many']),
        fuzzy_and(fuzzyValues['global']['High'], fuzzyValues['avg_rating']['High'], fuzzyValues['platform']['Wide']),
        fuzzy_and(fuzzyValues['global']['High'], fuzzyValues['recent']['New'], fuzzyValues['platform']['Wide']),
        fuzzy_and(fuzzyValues['avg_rating']['High'], fuzzyValues['count']['Many'], fuzzyValues['recent']['New']),
        fuzzy_and(fuzzyValues['global']['High'], fuzzyValues['avg_rating']['High'], fuzzyValues['count']['Many'], fuzzyValues['platform']['Wide']),
        fuzzy_and(fuzzyValues['global']['High'], fuzzyValues['avg_rating']['High']),
        fuzzy_and(fuzzyValues['global']['High'], fuzzyValues['platform']['Wide'])
    ]
    return {
        'Poor': max(poor_rules),
        'Decent' : max(decent_rules), 
        'Excellent' : max(excellent_rules)
    }

def mamdani_inference(fv):
    rule_strengths = rule(fv)
    rule_poor      = np.minimum(rule_strengths['Poor'],      out_poor)
    rule_decent    = np.minimum(rule_strengths['Decent'],    out_fair)
    rule_excellent = np.minimum(rule_strengths['Excellent'], out_excellent)
    return np.maximum(rule_poor, np.maximum(rule_decent, rule_excellent))

def sugeno_inference(fv):
    rule_strengths = rule(fv)
    return [
        {'weight': rule_strengths['Poor'],      'output': 0.0},
        {'weight': rule_strengths['Decent'],    'output': 5.0},
        {'weight': rule_strengths['Excellent'], 'output': 10.0},
    ]

def defuzzify_mamdani(x_output, aggregated_mf):
    numerator   = np.sum(x_output * aggregated_mf)
    denominator = np.sum(aggregated_mf)
    return numerator / denominator if denominator != 0 else 5.0

def defuzzify_sugeno(rules):
    numerator   = sum(r['weight'] * r['output'] for r in rules)
    denominator = sum(r['weight'] for r in rules)
    return numerator / denominator if denominator != 0 else 5.0

# --- STREAMLIT UI ---
st.title("🎮 Video Game Performance Evaluator")
st.markdown("This application evaluates games using the DKA Project's **Fuzzy Logic Framework**.")

st.sidebar.header("🕹️ Input Game Parameters")
global_score = st.sidebar.slider("Global Rating (0-5)", 0.0, 5.0, 2.5)
avg_rating = st.sidebar.slider("User Rating (1-5)", 1.0, 5.0, 3.0)
platforms = st.sidebar.slider("Number of Platforms", 1, 22, 5)
ratings_cnt = st.sidebar.slider("Rating Count", 0, 100, 50)
recency = st.sidebar.slider("Recency (0=Old, 10=New)", 0.0, 10.0, 5.0)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Global", global_score)
col2.metric("User", avg_rating)
col3.metric("Platforms", platforms)
col4.metric("Count", ratings_cnt)
col5.metric("Recency", recency)

st.write("---")

if st.button("Calculate Performance", type="primary"):
    with st.spinner("Processing through Fuzzy Inference System..."):
        fv = fuzzify(global_score, avg_rating, ratings_cnt, platforms, recency)

        m_pred = defuzzify_mamdani(out_x, mamdani_inference(fv))
        s_pred = defuzzify_sugeno(sugeno_inference(fv))

        st.success("Evaluation Complete!")
        c1, c2 = st.columns(2)
        c1.write("### Mamdani System Prediction:")
        c1.markdown(f"<h2 style='color: #4CAF50;'>{round(m_pred, 2)} / 10</h2>", unsafe_allow_html=True)
        c2.write("### Sugeno System Prediction:")
        c2.markdown(f"<h2 style='color: #2196F3;'>{round(s_pred, 2)} / 10</h2>", unsafe_allow_html=True)
