import streamlit as st
import pandas as pd
import colorsys
import joblib

# -----------------------------
# 1. Load Models
# -----------------------------
@st.cache_resource
def load_assets():
    try:
        return {
            'lr': joblib.load('lrmodel.joblib'),
            'rf': joblib.load('rfmodel.joblib'),
            'xgb': joblib.load('xgmodel.joblib'),
            'dtree': joblib.load('dtreemodel.joblib'),
            'scaler': joblib.load('scaler.joblib'),
            'lab': joblib.load('labencode.joblib')
        }
    except Exception as e:
        st.error(f"Models missing: {e}")
        return None

assets = load_assets()

# -----------------------------
# 2. State & Sync Logic
# -----------------------------
# Initialize session state if empty
if 'r' not in st.session_state:
    st.session_state.r, st.session_state.g, st.session_state.b = 128, 128, 128
    st.session_state.h, st.session_state.s, st.session_state.l = 0, 0, 50

def on_rgb_change():
    """Triggered only when RGB sliders are moved"""
    h, l, s = colorsys.rgb_to_hls(st.session_state.r/255, st.session_state.g/255, st.session_state.b/255)
    st.session_state.h = int(h * 360)
    st.session_state.s = int(s * 100)
    st.session_state.l = int(l * 100)

def on_hsl_change():
    """Triggered only when HSL sliders are moved"""
    r, g, b = colorsys.hls_to_rgb(st.session_state.h/360, st.session_state.l/100, st.session_state.s/100)
    st.session_state.r = int(r * 255)
    st.session_state.g = int(g * 255)
    st.session_state.b = int(b * 255)

# -----------------------------
# 3. UI Layout
# -----------------------------
st.title("🎨 Independent Color Mood Predictor")

# Mode Selection
mode = st.radio("Active Control Set:", ["RGB", "HSL"], horizontal=True)

# Layout Columns
col1, col2, col3 = st.columns(3)

if mode == "RGB":
    with col1: st.slider("Red", 0, 255, key="r", on_change=on_rgb_change)
    with col2: st.slider("Green", 0, 255, key="g", on_change=on_rgb_change)
    with col3: st.slider("Blue", 0, 255, key="b", on_change=on_rgb_change)
    st.info(f"Current HSL: {st.session_state.h}°, {st.session_state.s}%, {st.session_state.l}%")
else:
    with col1: st.slider("Hue", 0, 360, key="h", on_change=on_hsl_change)
    with col2: st.slider("Saturation", 0, 100, key="s", on_change=on_hsl_change)
    with col3: st.slider("Lightness", 0, 100, key="l", on_change=on_hsl_change)
    st.info(f"Current RGB: {st.session_state.r}, {st.session_state.g}, {st.session_state.b}")

# -----------------------------
# 4. Color Preview Box
# -----------------------------
hex_color = '#%02x%02x%02x' % (st.session_state.r, st.session_state.g, st.session_state.b)
st.markdown(
    f"""
    <div style="background-color: {hex_color}; height: 150px; border-radius: 15px; 
    border: 4px solid white; box-shadow: 0px 4px 10px rgba(0,0,0,0.3); margin-bottom: 20px;">
    </div>
    """, 
    unsafe_allow_html=True
)

# -----------------------------
# 5. Prediction
# -----------------------------
if assets:
    model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest", "XGBoost", "Decision Tree"])
    
    # Model config
    config = {
        "Logistic Regression": (assets['lr'], True),
        "Random Forest": (assets['rf'], False),
        "XGBoost": (assets['xgb'], False),
        "Decision Tree": (assets['dtree'], False)
    }
    model, use_scaler = config[model_name]

    if st.button("🔮 Predict Mood", use_container_width=True):
        # Data Prep
        features = ['R','G','B','Hue','Saturation','Lightness']
        data = [[st.session_state.r, st.session_state.g, st.session_state.b, 
                 st.session_state.h, st.session_state.s, st.session_state.l]]
        df = pd.DataFrame(data, columns=features)
        
        # Scaling & Prediction
        X = assets['scaler'].transform(df) if use_scaler else df.values
        res = model.predict(X)
        mood = assets['lab'].inverse_transform(res)[0]
        
        st.markdown(f"### The Color feels: **{mood}**")