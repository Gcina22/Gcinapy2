import streamlit as st

# 1. MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Innocom Smart Farming",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. CUSTOM CSS FOR BRANDING, HERO OVERLAY, & NAVIGATION
st.markdown("""
<style>
    /* Top Brand Bar Styling */
    .top-brand-bar {
        background-color: #0f2419;
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        border-bottom: 1px solid #1e3d2b;
    }
    .brand-title {
        font-weight: bold;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Top Features Row */
    .features-row {
        background-color: #122d1f;
        padding: 12px 20px;
        display: flex;
        justify-content: space-around;
        color: #a3b899;
        font-size: 0.85rem;
        border-bottom: 1px solid #1e3d2b;
    }
    .feature-item {
        text-align: center;
        cursor: pointer;
    }
    .feature-item:hover {
        color: #ffffff;
    }

    /* Main Hero Text Overlay styling */
    .hero-container {
        text-align: center;
        padding: 40px 20px 10px 20px;
        margin-top: 10px;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #112e1f;
        line-height: 1.2;
        margin-bottom: 15px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #4a5d4e;
        max-width: 800px;
        margin: 0 auto 25px auto;
    }
</style>
""", unsafe_allow_html=True)

# 3. TOP BRANDING BAR
st.markdown("""
<div class="top-brand-bar">
    <div class="brand-title">🌐 Innocom Smart Farming</div>
    <div>
        <span style="margin-right:15px; cursor:pointer; font-size:0.9rem;">Sign in</span>
        <button style="background-color:#4CAF50; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:bold;">Get Started</button>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. CORE UTILITY/SERVICES NAVIGATION BAR
st.markdown("""
<div class="features-row">
    <div class="feature-item">📡<br>Satellite Data<br>Management</div>
    <div class="feature-item">☁️<br>Climate Risk<br>Assessment</div>
    <div class="feature-item">📈<br>Crop Yield<br>Analytics</div>
    <div class="feature-item">💧<br>Irrigation<br>Control Panel</div>
    <div class="feature-item">🚚<br>Field Operations<br>Logistics</div>
    <div class="feature-item">📊<br>Market Intelligence<br>Dashboard</div>
</div>
""", unsafe_allow_html=True)

st.write("") # Spacer

# 5. MAIN HORIZONTAL NAVIGATION MENU
nav_cols = st.columns([1, 1.2, 1.2, 1, 2.5, 1, 1.5, 1.2, 1.2])

with nav_cols[0]: st.selectbox("Home", ["Overview"], label_visibility="collapsed")
with nav_cols[1]: st.selectbox("About Us", ["Our Team", "History"], label_visibility="collapsed")
with nav_cols[2]: st.selectbox("Facilities", ["Labs", "Fields"], label_visibility="collapsed")
with nav_cols[3]: st.selectbox("Media", ["News", "Gallery"], label_visibility="collapsed")
with nav_cols[4]: st.selectbox("Training & Advisory", ["Services", "Workshops"], label_visibility="collapsed")
with nav_cols[5]: st.selectbox("Tenders", ["Active", "Archive"], label_visibility="collapsed")
with nav_cols[6]: st.selectbox("Careers at ARC", ["Openings"], label_visibility="collapsed")
with nav_cols[7]: st.selectbox("Contact Us", ["Offices"], label_visibility="collapsed")
with nav_cols[8]: st.selectbox("Quick Links", ["Resources"], label_visibility="collapsed")

st.divider()

# 6. HERO SECTION CONTENT
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Climate change is transforming<br>maize farming with orbital data</h1>
    <p class="hero-subtitle">
        Monitor maize health, water status, and soil conditions from space. 
        Innocom Smart Farming delivers actionable insights from orbit to help 
        farmers, agronomists, and businesses adapt and thrive.
    </p>
</div>
""", unsafe_allow_html=True)

# Centered "Read More" Button
col_btn_l, col_btn_c, col_btn_r = st.columns([5, 2, 5])
with col_btn_c:
    st.button("Read More", use_container_width=True, type="primary")

st.write("") # Spacer

# 7. MAIN HERO VISUAL
st.image(
    "https://images.unsplash.com/photo-1592417817098-8f3d6eb19675?q=80&w=1200", 
    caption="Smart Orbital and Field Monitoring Solutions over Agricultural Landscapes",
    use_container_width=True
)

# 8. FOOTER SECTION
st.divider()
footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.caption("**Innocom Geospatial (Pty) Ltd**")
    st.caption("Democratizing agricultural geospatial technology.")

with footer_col2:
    st.markdown(
        "<div style='text-align: right; color: gray; font-size: 0.8rem;'>"
        "© 2026 Innocommunications i.e. All rights reserved."
        "</div>", 
        unsafe_allow_html=True
    )
