"""
CalorieNet Streamlit UI
"""
import os

import streamlit as st
import requests
from PIL import Image


# Page config
st.set_page_config(
    page_title="dish2calories - Food Nutrition Analyzer",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS (colors use transparency so it works in both light and dark themes)
st.markdown("""
<style>
    .block-container { max-width: 880px; padding-top: 2rem; }
    #MainMenu, footer { visibility: hidden; }

    .hero {
        text-align: center;
        padding: 2.2rem 1rem 1.6rem;
        margin-bottom: 1.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(16,185,129,.18), rgba(59,130,246,.16) 55%, rgba(168,85,247,.16));
        border: 1px solid rgba(128,128,128,.22);
    }
    .hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0; padding: 0;
               background: linear-gradient(90deg,#10b981,#3b82f6 60%,#a855f7);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p { margin: .5rem 0 0; opacity: .8; font-size: 1.05rem; }
    .hero .pills { margin-top: 1rem; }
    .pill { display: inline-block; padding: .2rem .75rem; margin: .15rem; border-radius: 999px;
            font-size: .78rem; border: 1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.10); }

    .section-title { font-size: 1.15rem; font-weight: 700; margin: 1.6rem 0 .6rem; }

    .kcal-card {
        text-align: center; padding: 1.4rem 1rem; border-radius: 18px;
        background: linear-gradient(135deg, #10b981, #3b82f6); color: #fff;
        box-shadow: 0 8px 24px rgba(59,130,246,.25);
    }
    .kcal-card .value { font-size: 3.4rem; font-weight: 800; line-height: 1; }
    .kcal-card .label { opacity: .9; margin-top: .3rem; letter-spacing: .08em; text-transform: uppercase; font-size: .8rem; }

    .macro-card { text-align: center; padding: 1rem .5rem; border-radius: 16px;
                  background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.20);
                  border-top: 4px solid var(--accent); }
    .macro-card .value { font-size: 1.7rem; font-weight: 750; }
    .macro-card .label { font-size: .8rem; opacity: .7; margin-top: .15rem; }

    .split-bar { display: flex; height: 14px; border-radius: 999px; overflow: hidden; margin: .6rem 0 .3rem;
                 background: rgba(128,128,128,.18); }
    .split-legend { display: flex; justify-content: space-between; font-size: .8rem; opacity: .8; flex-wrap: wrap; gap: .4rem; }
    .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: .3rem; }

    .ingredient { display: flex; align-items: center; gap: .7rem; padding: .45rem .8rem; margin: .3rem 0;
                  border-radius: 12px; background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18); }
    .ingredient .name { flex: 0 0 38%; font-weight: 600; text-transform: capitalize; }
    .ingredient .bar { flex: 1; height: 8px; border-radius: 999px; background: rgba(128,128,128,.2); overflow: hidden; }
    .ingredient .bar > span { display: block; height: 100%; border-radius: 999px;
                              background: linear-gradient(90deg,#10b981,#3b82f6); }
    .ingredient .pct { flex: 0 0 3.2rem; text-align: right; font-size: .85rem; opacity: .8; }
    .ingredient.dim { opacity: .55; }

    .note { padding: .8rem 1rem; border-radius: 14px; font-size: .88rem; margin: 1rem 0;
            background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.4); }
    .footer { text-align: center; opacity: .6; font-size: .82rem; margin-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# Configuration
# If API_URL is set, the UI talks to a separate FastAPI server (Docker / compose).
# Otherwise the model runs inside this process (Streamlit Community Cloud, local `streamlit run app.py`).
API_URL = os.getenv("API_URL", "").rstrip("/")
USE_API = bool(API_URL)
UPLOAD_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"


@st.cache_resource(show_spinner="Loading model (first run only)...")
def get_local_predictor():
    """Load the model once per server process (imports torch lazily so the page renders fast)."""
    import service
    return service.load_predictor()


def check_api_health():
    """Check that the backend (remote API or local model) is ready"""
    if not USE_API:
        try:
            get_local_predictor()
            return True
        except Exception:
            return False
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200 and response.json().get("model_loaded", False)
    except Exception:
        return False


def predict_local(image_file):
    """Run the model in-process"""
    try:
        import numpy as np
        import service
        img = np.array(Image.open(image_file).convert("RGB"))
        results = get_local_predictor().predict_img(img)
        return service.format_results(results), None
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def predict_nutrition(image_file):
    """Get nutrition predictions (remote API or local model)"""
    if not USE_API:
        return predict_local(image_file)
    try:
        files = {"file": image_file}
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return None, f"API Error: {error_detail}"
            
    except requests.exceptions.ConnectionError:
        return None, f"❌ Cannot connect to API at {API_URL}"
    except requests.exceptions.Timeout:
        return None, "⏱️ Request timed out. Please try again."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def display_metrics(data):
    """Calories hero card, macro cards and a calorie-split bar"""
    st.markdown(f"""
    <div class="kcal-card">
        <div class="value">{data['total_calories']:.0f}</div>
        <div class="label">kcal · estimated total</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    cards = [
        ("Protein", f"{data['total_protein']:.1f} g", "#3b82f6"),
        ("Fat", f"{data['total_fat']:.1f} g", "#f59e0b"),
        ("Carbs", f"{data['total_carbs']:.1f} g", "#10b981"),
        ("Mass", f"{data['total_mass']:.0f} g", "#a855f7"),
    ]
    for col, (label, value, color) in zip(st.columns(4), cards):
        col.markdown(f"""
        <div class="macro-card" style="--accent:{color}">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Calories by macro (4/9/4 kcal per gram)
    kc = {"Protein": 4 * data["total_protein"], "Fat": 9 * data["total_fat"], "Carbs": 4 * data["total_carbs"]}
    total = sum(kc.values()) or 1.0
    colors = {"Protein": "#3b82f6", "Fat": "#f59e0b", "Carbs": "#10b981"}
    bar = "".join(f'<div style="width:{v / total * 100:.1f}%;background:{colors[k]}"></div>' for k, v in kc.items())
    legend = "".join(
        f'<span><span class="dot" style="background:{colors[k]}"></span>{k} {v / total:.0%}</span>' for k, v in kc.items()
    )
    st.markdown(f'<div class="split-bar">{bar}</div><div class="split-legend">{legend}</div>', unsafe_allow_html=True)


def display_detected_foods(data):
    """Detected ingredients with confidence bars, plus the model's other top guesses"""
    st.markdown('<div class="section-title">🥗 Detected ingredients</div>', unsafe_allow_html=True)

    if not data["labels_txt"]:
        st.warning("No ingredient reached the confidence threshold, so the nutrition numbers are not meaningful. "
                   "Try a clearer, top-down photo of a single plate.")
    rows = ""
    for name, conf in zip(data["labels_txt"], data["labels_conf"]):
        rows += (f'<div class="ingredient"><span class="name">{name}</span>'
                 f'<span class="bar"><span style="width:{conf * 100:.0f}%"></span></span>'
                 f'<span class="pct">{conf:.0%}</span></div>')
    st.markdown(rows, unsafe_allow_html=True)

    others = [(n, p) for n, p in data.get("top_candidates", []) if n not in data["labels_txt"]]
    if others:
        with st.expander("Other candidates the model considered"):
            rows = "".join(
                f'<div class="ingredient dim"><span class="name">{n}</span>'
                f'<span class="bar"><span style="width:{p * 100:.0f}%"></span></span>'
                f'<span class="pct">{p:.0%}</span></div>' for n, p in others
            )
            st.markdown(rows, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="hero">
        <h1>🍽️ dish2calories</h1>
        <p>Snap a plate. Get calories, macros and ingredients in seconds.</p>
        <div class="pills">
            <span class="pill">EfficientNet-B0</span><span class="pill">Nutrition5k</span>
            <span class="pill">75 ingredients</span><span class="pill">ONNX Runtime</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check backend status
    if not check_api_health():
        if USE_API:
            st.error("🚨 **The API is not running!**")
            st.info("Please start the API server first:")
            st.code("python inference.py", language="bash")
        else:
            st.error("🚨 **Could not load the model.** Check that `weights/` and `configs/` exist.")
        st.stop()

    st.markdown(
        '<div class="note">💡 <b>Works best on:</b> a single plate photographed from above, in good light, '
        'with everyday cafeteria-style foods. The model only knows 75 ingredients '
        '(no pasta, bread, noodles, soups or desserts), so unfamiliar dishes get matched to the closest known ingredient '
        'and the numbers can be far off.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a photo of your dish",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="JPG, PNG, BMP or WEBP, up to 10 MB",
    )

    if uploaded_file is None:
        st.session_state.pop("analysis", None)
        st.markdown('<div class="footer">Estimates only, not medical or dietary advice.</div>', unsafe_allow_html=True)
        return

    file_key = (uploaded_file.name, uploaded_file.size)
    # Drop results that belong to a previously uploaded image
    if st.session_state.get("analysis", {}).get("key") != file_key:
        st.session_state.pop("analysis", None)

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("🔍 Analyze nutrition", type="primary", use_container_width=True):
        with st.spinner("Analyzing your dish..."):
            uploaded_file.seek(0)
            result, error = predict_nutrition(uploaded_file)
        if error:
            st.error(error)
        else:
            st.session_state["analysis"] = {"key": file_key, "result": result}

    analysis = st.session_state.get("analysis")
    if analysis:
        result = analysis["result"]
        st.markdown('<div class="section-title">📊 Nutrition summary</div>', unsafe_allow_html=True)
        display_metrics(result)
        display_detected_foods(result)
        if result["mean_conf"] < 0.6 and result["labels_txt"]:
            st.markdown('<div class="note">⚠️ Low confidence. This dish may be outside what the model was trained on.</div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="footer">Estimates only, not medical or dietary advice. '
                'Built with PyTorch → ONNX · Streamlit</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
