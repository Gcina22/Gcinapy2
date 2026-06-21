
import streamlit as st
import base64

st.set_page_config(
    page_title="Smart Farming Platform",
    page_icon="🌽",
    layout="wide"
)

# Function to load image as base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ==========================
# BACKGROUND IMAGE
# ==========================
bg_image = get_base64("maize_background.jpg")

page_bg = f"""
<style>

[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

.main {{
    background: rgba(0,0,0,0);
}}

.navbar {{
    background: rgba(0, 70, 32, 0.85);
    padding: 12px 30px;
    border-radius: 10px;
    margin-bottom: 30px;
}}

.nav-links {{
    display: flex;
    justify-content: center;
    gap: 35px;
    color: white;
    font-size: 16px;
    font-weight: 500;
}}

.hero {{
    text-align: center;
    color: white;
    padding-top: 100px;
    padding-bottom: 120px;
}}

.hero h1 {{
    font-size: 55px;
    font-weight: bold;
}}

.hero p {{
    font-size: 22px;
    max-width: 850px;
    margin: auto;
}}

.read-btn {{
    background-color: #d4af37;
    color: black;
    padding: 12px 30px;
    border-radius: 30px;
    font-size: 18px;
    font-weight: bold;
    display: inline-block;
    margin-top: 20px;
}}

.feature-card {{
    background: rgba(255,255,255,0.88);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    height: 180px;
}}

.feature-title {{
    font-size: 20px;
    font-weight: bold;
    color: #14532d;
}}

.feature-text {{
    color: #444;
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ==========================
# NAVBAR
# ==========================
st.markdown("""
<div class="navbar">
    <div class="nav-links">
        <span>Home</span>
        <span>About Us</span>
        <span>Facilities</span>
        <span>Media</span>
        <span>Training Services</span>
        <span>Tenders</span>
        <span>Careers</span>
        <span>Contact Us</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================
# HERO SECTION
# ==========================
st.markdown("""
<div class="hero">
    <h1>Smart Farming with Orbital Data</h1>
    <p>
        Monitor maize health, water status, crop growth and soil conditions
        using satellite imagery, drones and advanced analytics.
        Deliver actionable insights for farmers, agronomists and businesses.
    </p>
    <div class="read-btn">Read More</div>
</div>
""", unsafe_allow_html=True)

# ==========================
# FEATURES
# ==========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🌍 Satellite Data</div>
        <br>
        <div class="feature-text">
            Real-time monitoring of crop conditions using Earth observation data.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🚁 Drone Analytics</div>
        <br>
        <div class="feature-text">
            UAV-based crop assessment for precision agriculture.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">💧 Irrigation Control</div>
        <br>
        <div class="feature-text">
            Optimize water use and detect crop water stress.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">📈 Yield Prediction</div>
        <br>
        <div class="feature-text">
            Forecast crop performance using AI and remote sensing.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center>
<h3 style='color:white;'>
🌽 Precision Agriculture for Sustainable Maize Production
</h3>
</center>
""", unsafe_allow_html=True)
