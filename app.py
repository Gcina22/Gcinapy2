# app.py
import streamlit as st
import pandas as pd

# pip install folium streamlit-folium rasterio pillow numpy
HAS_OVERLAY = True
try:
    import numpy as np
    import folium
    from streamlit_folium import st_folium
    import rasterio
    from rasterio.warp import transform_bounds
except Exception:
    HAS_OVERLAY = False


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Water Hyacinth Seasonal Monitoring – Laing Dam (2025)",
    layout="wide"
)


# --------------------------------------------------
# Load data (used for explorer pages; map view uses rasters)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("water_hyacinth_2025.csv", sep=";")
    df.columns = df.columns.str.strip()
    for c in ["Area_km2", "Latitude", "Longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def validate_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.write("Columns found:", list(df.columns))
        st.stop()


# --------------------------------------------------
# Raster helpers: bounds + class-color image
# --------------------------------------------------
def raster_bounds_4326(tif_path: str):
    """Return bounds in EPSG:4326 as [[south, west], [north, east]]"""
    with rasterio.open(tif_path) as src:
        b = src.bounds
        src_crs = src.crs
        if src_crs and str(src_crs).upper() != "EPSG:4326":
            west, south, east, north = transform_bounds(
                src_crs, "EPSG:4326", b.left, b.bottom, b.right, b.top
            )
        else:
            west, south, east, north = b.left, b.bottom, b.right, b.top
    return [[south, west], [north, east]]


def classify_raster_to_rgba(tif_path: str):
    """
    Convert a single-band raster to an RGBA image with 2 classes:
      - background (0 or nodata) -> blue
      - hyacinth (>0) -> green
    Returns (rgba_uint8, bounds_4326)
    """
    bounds = raster_bounds_4326(tif_path)

    with rasterio.open(tif_path) as src:
        band = src.read(1)

        nodata = src.nodata
        band = band.astype("float32")

        # Build masks
        if nodata is not None:
            is_nodata = (band == nodata) | np.isnan(band)
        else:
            is_nodata = np.isnan(band)

        # Background: 0 or nodata
        is_background = is_nodata | (band <= 0)

        # Hyacinth: > 0
        is_hyacinth = (~is_nodata) & (band > 0)

        h, w = band.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Blue background (water/other)
        rgba[is_background] = [0, 120, 255, 180]  # R,G,B,A

        # Green hyacinth
        rgba[is_hyacinth] = [0, 200, 0, 220]      # R,G,B,A

        return rgba, bounds


def add_legend(m: folium.Map, title="Legend"):
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: #000000;
        color: #ffffff;
        padding: 12px 14px;
        border: 2px solid rgba(255,255,255,0.4);
        border-radius: 8px;
        font-size: 14px;
        ">
        <div style="font-weight: 600; margin-bottom: 10px; color: #ffffff;">
            {title}
        </div>

        <div style="display:flex; align-items:center; margin-bottom:8px;">
            <div style="
                width:18px;
                height:18px;
                background:#0078ff;
                margin-right:8px;
                border:1px solid #ffffff;
            "></div>
            <div>Water </div>
        </div>

        <div style="display:flex; align-items:center;">
            <div style="
                width:18px;
                height:18px;
                background:#00c800;
                margin-right:8px;
                border:1px solid #ffffff;
            "></div>
            <div>Water hyacinth</div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))



def build_raster_map(tif_path: str, map_title: str, opacity: float):
    """Create a Folium map that fits to raster bounds and shows classified overlay + legend."""
    rgba, bounds = classify_raster_to_rgba(tif_path)

    # Center from bounds
    (south, west), (north, east) = bounds
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

    # Add overlay
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=opacity,
        name=map_title,
        interactive=False,
        zindex=1
    ).add_to(m)

    # Fit to raster bounds (this ensures it zooms in correctly)
    m.fit_bounds(bounds)

    folium.LayerControl(collapsed=True).add_to(m)
    add_legend(m, title=map_title)
    return m


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Project Overview", "Seasonal Data Explorer", "Map View", "Contact"]
)

# --------------------------------------------------
# Load data (for non-map pages)
# --------------------------------------------------
data = load_data()
validate_columns(data, ["Season", "Month", "Area_km2", "Latitude", "Longitude"])


# --------------------------------------------------
# Project Overview
# --------------------------------------------------
if menu == "Project Overview":
    st.title("Water Hyacinth Seasonal Monitoring – Laing Dam (2025)")

    st.write("""
    This Streamlit application explores the **seasonal distribution of water hyacinth**
    in **Laing Dam** during **Winter and Summer 2025**.

    Developed for the **CHPC Summer Coding School**, the app demonstrates
    data handling, filtering, visualisation, and raster-based mapping in Streamlit.
    """)

    st.subheader("Study Area")
    st.write("Laing Dam, Eastern Cape, South Africa")

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)


# --------------------------------------------------
# Seasonal Data Explorer
# --------------------------------------------------
elif menu == "Seasonal Data Explorer":
    st.title("Seasonal Data Explorer")

    seasons = sorted(data["Season"].dropna().unique().tolist())
    season = st.selectbox("Select Season", seasons)

    min_area = float(data["Area_km2"].min())
    max_area = float(data["Area_km2"].max())

    area_range = st.slider(
        "Filter by Area (km²)",
        min_value=min_area,
        max_value=max_area,
        value=(min_area, max_area)
    )

    filtered = data[
        (data["Season"] == season) &
        (data["Area_km2"].between(area_range[0], area_range[1]))
    ].copy()

    st.subheader(f"Hyacinth Coverage – {season}")
    st.dataframe(filtered, use_container_width=True)

    st.subheader("Area by Month")
    st.bar_chart(filtered.groupby("Month")["Area_km2"].sum())


# --------------------------------------------------
# Map View (Winter vs Summer side-by-side)
# --------------------------------------------------
elif menu == "Map View":
    st.title("Winter vs Summer Raster Comparison (Laing Dam, 2025)")

    if not HAS_OVERLAY:
        st.error(
            "GeoTIFF overlay requires additional libraries.\n\n"
            "Install with:\n"
            "pip install folium streamlit-folium rasterio pillow numpy"
        )
        st.stop()

    st.write("These maps are raster-only (no point markers). Colors represent two classes: **blue** (water/other) and **green** (water hyacinth).")

    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.85, 0.05)

    # Make sure these TIFFs are in the same folder as app.py
    winter_tif = "Winter_2025_Hyacinth_Map.tif"
    summer_tif = "Summer_2025_Hyacinth_Map.tif"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Winter 2025")
        winter_map = build_raster_map(winter_tif, "Winter 2025 – Hyacinth Classes", opacity)
        st_folium(winter_map, width=650, height=550)

    with col2:
        st.subheader("Summer 2025")
        summer_map = build_raster_map(summer_tif, "Summer 2025 – Hyacinth Classes", opacity)
        st_folium(summer_map, width=650, height=550)


# --------------------------------------------------
# Contact
# --------------------------------------------------
elif menu == "Contact":
    st.header("Contact Information")
    st.write("**Name:** Sinesipho Gom")
    st.write("**Programme:** CHPC Summer Coding School")
    st.write("**Institution:** University of Fort Hare")
