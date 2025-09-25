"""
CalorieNet Streamlit UI
"""
import streamlit as st
import requests
from PIL import Image


# Page config
st.set_page_config(
    page_title="CalorieNet - Food Nutrition Analyzer",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .food-tag {
        display: inline-block;
        background: #f0f2f6;
        color: #262730;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
        border: 2px solid #e0e0e0;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200 and response.json().get("model_loaded", False)
    except:
        return False

def predict_nutrition(image_file):
    """Send image to API and get nutrition predictions"""
    try:
        files = {"file": image_file}
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return None, f"API Error: {error_detail}"
            
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to API. Make sure the CalorieNet API is running on localhost:8000"
    except requests.exceptions.Timeout:
        return None, "⏱️ Request timed out. Please try again."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def create_nutrition_charts(data):
    """Create beautiful nutrition visualization charts"""
    # This function is no longer used but kept for compatibility
    pass

def display_metrics(data):
    """Display key metrics in a visually appealing way"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['total_calories']:.0f}</div>
            <div class="metric-label">Calories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['total_protein']:.1f}g</div>
            <div class="metric-label">Protein</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['total_fat']:.1f}g</div>
            <div class="metric-label">Fat</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['total_carbs']:.1f}g</div>
            <div class="metric-label">Carbs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['total_mass']:.1f}g</div>
            <div class="metric-label">Total Mass</div>
        </div>
        """, unsafe_allow_html=True)

def display_detected_foods(data):
    """Display detected foods with mean confidence"""
    st.markdown("### 🍽️ Detected Foods")
    
    # Mean confidence
    st.markdown(f"**Mean Confidence: {data['mean_conf']:.1%}**")
    
    # Food tags
    foods_html = ""
    for food in data['labels_txt']:
        foods_html += f'<span class="food-tag">{food}</span>'
    
    st.markdown(foods_html, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🍎 CalorieNet</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Food Nutrition Analyzer</p>', unsafe_allow_html=True)
    
    # Check API status
    if not check_api_health():
        st.error("🚨 **CalorieNet API is not running!**")
        st.info("Please start the API server first:")
        st.code("python infrence.py", language="bash")
        st.stop()
    
    # File uploader
    st.markdown("### 📸 Upload Food Image")
    uploaded_file = st.file_uploader(
        "Choose a food image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Upload a clear image of food for nutrition analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image centered
        st.markdown("#### 📸 Your Image")
        image = Image.open(uploaded_file)
        
        # Center the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Center the analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_clicked = st.button("🔍 Analyze Nutrition", type="primary", use_container_width=True)
        
        if analyze_clicked:
            with st.spinner("🧠 AI is analyzing your food..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Get prediction
                result, error = predict_nutrition(uploaded_file)
                
                if error:
                    st.error(error)
                else:
                    st.success("✅ Analysis Complete!")
                    
                    # Store result in session state for persistent display
                    st.session_state['analysis_result'] = result
        
        # Display results if available
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            
            st.markdown("---")
            
            # Key metrics
            st.markdown("### 📊 Nutrition Summary")
            display_metrics(result)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detected foods
            display_detected_foods(result)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "Made with ❤️ using CalorieNet AI | Upload food images for instant nutrition analysis"
        "</p>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
