import streamlit as st

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="Smart Farming",
    page_icon="🌽",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>

/* Background */
.stApp {
    background-image: url('https://images.unsplash.com/photo-1500937386664-56d1dfef3854');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Remove default Streamlit spacing */
.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Navigation Bar */
.navbar {
    background: rgba(0, 80, 40, 0.85);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.navbar ul {
    list-style-type: none;
    display: flex;
    justify-content: center;
    gap: 35px;
    margin: 0;
    padding: 0;
}

.navbar li {
    color: white;
    font-weight: bold;
    cursor: pointer;
}

/* Hero Section */
.hero {
    text-align: center;
    padding-top: 80px;
    padding-bottom: 120px;
    color: white;
}

.hero h1 {
    font-size: 60px;
    font-weight: 700;
}

.hero p {
    font-size: 22px;
    max-width: 900px;
    margin: auto;
    background: rgba(0,0,0,0.35);
    padding: 15px;
    border-radius: 10px;
}

/* Feature Cards */
.card {
    background: rgba(255,255,255,0.9);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    min-height: 180px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}

.card h3 {
    color: #0b6e3b;
}

.card p {
    color: #333333;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# NAVIGATION BAR
# ==========================
st.markdown("""
<div class="navbar">
    <ul>
        <li>Home</li>
        <li>About Us</li>
        <li>Facilities</li>
        <li>Media</li>
        <li>Training</li>
        <li>Tenders</li>
        <li>Careers</li>
        <li>Contact Us</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ==========================
# HERO SECTION
# ==========================
st.markdown("""
<div class="hero">
    <h1>Smart Farming with Orbital Data</h1>

    <p>
    Monitor maize health, crop growth, soil conditions and irrigation
    requirements using satellite imagery, drones and advanced analytics.
    Deliver actionable insights for farmers, agronomists and agricultural
    businesses.
    </p>
</div>
""", unsafe_allow_html=True)

# Centered Button
_, center_col, _ = st.columns([2,1,2])

with center_col:
    st.button("Read More", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================
# FEATURE SECTION
# ==========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
        <h3>🌍 Satellite Data</h3>
        <p>Monitor crop performance and vegetation health using Earth observation data.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>🚁 Drone Analytics</h3>
        <p>Collect high-resolution imagery for precision agriculture applications.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>💧 Irrigation Control</h3>
        <p>Detect crop water stress and optimize irrigation scheduling.</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <h3>📈 Yield Prediction</h3>
        <p>Use AI and remote sensing data to forecast crop yield and productivity.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
st.markdown("""
<div style='text-align:center;
            color:white;
            background:rgba(0,0,0,0.6);
            padding:15px;
            border-radius:10px;'>

<h3>🌽 Precision Agriculture Dashboard</h3>

<p>
Remote Sensing | GIS | UAV Analytics | Crop Monitoring | Smart Irrigation
</p>

</div>
""", unsafe_allow_html=True)
