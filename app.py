import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Farming Platform",
    page_icon="🌽",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Background image */
.stApp {
    background-image: linear-gradient(
        rgba(0,0,0,0.45),
        rgba(0,0,0,0.45)
    ),
    url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Remove default spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Navigation Bar */
.navbar {
    background-color: rgba(0, 100, 0, 0.85);
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 30px;
}

.navbar ul {
    display: flex;
    justify-content: center;
    list-style: none;
    gap: 30px;
    margin: 0;
    padding: 0;
}

.navbar li {
    color: white;
    font-weight: bold;
    font-size: 16px;
}

/* Hero Section */
.hero {
    text-align: center;
    padding-top: 120px;
    padding-bottom: 120px;
}

.hero-title {
    color: white;
    font-size: 60px;
    font-weight: bold;
}

.hero-text {
    color: white;
    font-size: 22px;
    max-width: 900px;
    margin: auto;
    line-height: 1.8;
}

/* Feature Cards */
.card {
    background: rgba(255,255,255,0.90);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    min-height: 220px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
}

.card-title {
    color: #006400;
    font-size: 22px;
    font-weight: bold;
}

.card-text {
    color: #333333;
    font-size: 16px;
}

/* Footer */
.footer {
    background: rgba(0,0,0,0.7);
    color: white;
    text-align: center;
    padding: 20px;
    border-radius: 10px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVIGATION BAR
# --------------------------------------------------
st.markdown("""
<div class="navbar">
    <ul>
        <li>Home</li>
        <li>About Us</li>
        <li>Facilities</li>
        <li>Media</li>
        <li>Training Services</li>
        <li>Tenders</li>
        <li>Careers</li>
        <li>Contact Us</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">
        Smart Farming with Orbital Data
    </div>

    <br>

    <div class="hero-text">
        Harness the power of satellite imagery, UAV technology,
        GIS and advanced analytics to monitor crop health,
        optimize irrigation, assess soil conditions and improve
        agricultural productivity through data-driven
        decision-making.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# BUTTON
# --------------------------------------------------
col1, col2, col3 = st.columns([2,1,2])

with col2:
    st.button("Read More", use_container_width=True)

st.write("")

# --------------------------------------------------
# FEATURE CARDS
# --------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="card">
        <div class="card-title">🌍 Satellite Data</div>
        <br>
        <div class="card-text">
            Monitor crop growth and vegetation health using
            Earth Observation technologies.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🚁 UAV Analytics</div>
        <br>
        <div class="card-text">
            Capture high-resolution imagery for precision
            agriculture and crop monitoring.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
        <div class="card-title">💧 Irrigation Management</div>
        <br>
        <div class="card-text">
            Detect water stress and optimize irrigation
            scheduling across fields.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="card">
        <div class="card-title">📈 Yield Prediction</div>
        <br>
        <div class="card-text">
            Use AI and geospatial data to forecast crop yields
            and improve farm productivity.
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    <h3>🌽 Precision Agriculture Dashboard</h3>
    <p>
        Remote Sensing | GIS | UAV Analytics | Crop Monitoring |
        Smart Irrigation | Artificial Intelligence
    </p>
</div>
""", unsafe_allow_html=True)
