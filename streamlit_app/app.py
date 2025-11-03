import streamlit as st
import pandas as pd
import joblib
import sys
import os
import requests
from bs4 import BeautifulSoup
import time
import re
from textstat import flesch_reading_ease

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

st.set_page_config(
    page_title="SEO Content Analyzer", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
    }
    .success-banner {
        background-color: #1E40AF;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Universal file path loader
def load_file_smart(relative_path):
    """Try multiple path options to find files"""
    paths_to_try = [
        relative_path,                    # Streamlit Cloud
        f'../{relative_path}',           # Local development
        f'../../{relative_path}',        # Alternative local
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            return path
    
    # If no path works, return the original and let it fail with clear error
    return relative_path

# Load model and data with universal paths
@st.cache_resource
def load_model():
    model_path = load_file_smart('models/quality_model.pkl')
    return joblib.load(model_path)

@st.cache_data
def load_data():
    features_path = load_file_smart('data/features.csv')
    extracted_path = load_file_smart('data/extracted_content.csv')
    
    features = pd.read_csv(features_path)
    extracted = pd.read_csv(extracted_path)
    return features, extracted

def scrape_and_parse_url(url):
    """Enhanced scraping with better headers and error handling"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        time.sleep(1)  # Respectful delay
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Check if content is reasonable
        if len(response.content) < 1000:
            return "", "", 0
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else ""
        
        # Enhanced content extraction
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', 
            '.post-content', '.article-content', '.entry-content',
            'section', '.main-content'
        ]
        
        body_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                text = ' '.join([elem.get_text().strip() for elem in elements if elem.get_text().strip()])
                if len(text) > 200:  # More meaningful threshold
                    body_text = text
                    break
        
        # Fallback: get all paragraph text
        if not body_text:
            paragraphs = soup.find_all('p')
            body_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # Final fallback
        if not body_text:
            body_text = soup.get_text()
        
        # Clean text
        body_text = ' '.join(body_text.split())
        word_count = len(body_text.split())
        
        return title, body_text, word_count
        
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return "", "", 0

def calculate_features(body_text, word_count):
    """Calculate features from text"""
    sentence_count = len(re.findall(r'[.!?]+', body_text)) if body_text else 0
    readability = flesch_reading_ease(body_text) if body_text and len(body_text.split()) > 10 else 0
    is_thin = word_count < 500
    
    return sentence_count, readability, is_thin

def predict_quality(model, word_count, sentence_count, readability):
    """Predict quality using trained model"""
    features = pd.DataFrame([{
        'word_count': word_count,
        'sentence_count': sentence_count,
        'flesch_reading_ease': readability
    }])
    
    return model.predict(features)[0]

def improved_similarity(target_url, target_text, existing_data, top_n=5):
    """Improved similarity that excludes self-matches"""
    if not target_text:
        return []
    
    target_word_count = len(target_text.split())
    
    similar_pages = []
    for _, row in existing_data.iterrows():
        # CRITICAL: Skip the same URL to avoid self-match
        if row['url'] == target_url:
            continue
            
        if pd.isna(row['body_text']):
            continue
            
        existing_text = row['body_text']
        existing_wc = len(existing_text.split())
        
        # Multiple similarity factors
        wc_similarity = 1 - abs(target_word_count - existing_wc) / max(target_word_count, existing_wc, 1)
        
        # Keyword overlap
        target_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', target_text.lower()))
        existing_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', existing_text.lower()))
        
        if target_words and existing_words:
            keyword_overlap = len(target_words.intersection(existing_words)) / len(target_words.union(existing_words))
        else:
            keyword_overlap = 0
        
        # Combined similarity score
        combined_similarity = (wc_similarity * 0.6) + (keyword_overlap * 0.4)
        
        if combined_similarity > 0.3:
            similar_pages.append({
                'url': row['url'],
                'similarity': combined_similarity,
                'word_count': existing_wc,
                'quality': row.get('quality_label', 'Unknown')
            })
    
    similar_pages.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_pages[:top_n]

# Load data
try:
    model = load_model()
    existing_data, extracted_data = load_data()
    # Merge with features for similarity
    enhanced_data = extracted_data.merge(existing_data, on='url', how='left')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Header
st.markdown('<h1 class="main-header">🔍 SEO Content Quality & Duplicate Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze any URL for SEO content quality and discover similar pages</p>', unsafe_allow_html=True)

# URL Input Section - Improved Layout
st.markdown("### 📝 Enter URL to Analyze")

# Using columns with better proportions
input_col, button_col = st.columns([4, 1])

with input_col:
    url = st.text_input(
        "Website URL", 
        value="https://www.bbc.com/news/technology",
        placeholder="https://example.com",
        label_visibility="collapsed"
    )

with button_col:
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

if analyze_btn and url:
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    with st.spinner("🔍 Analyzing content... This may take 10-15 seconds."):
        title, body_text, word_count = scrape_and_parse_url(url)
        
        if word_count == 0:
            st.error("""
            ❌ **Unable to extract content from this URL.** 
            
            Common reasons:
            - Website blocks automated requests
            - Invalid or inaccessible URL
            - No meaningful text content
            - Connection timeout
            
            Try a different URL or check if the website is accessible.
            """)
        else:
            # Calculate features
            sentence_count, readability, is_thin = calculate_features(body_text, word_count)
            quality_label = predict_quality(model, word_count, sentence_count, readability)
            
            # Success banner
            st.markdown(f"""
            <div class="success-banner">
                <h3>✅ Analysis Complete!</h3>
                <p>Successfully analyzed <strong>{url}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics in cards
            st.markdown("### 📊 Content Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Word Count", f"{word_count:,}")
                st.metric("Thin Content", "✅ No" if not is_thin else "❌ Yes")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Readability Score", f"{readability:.1f}")
                # Readability interpretation
                if readability > 60:
                    level = "😊 Easy"
                elif readability > 30:
                    level = "😐 Moderate" 
                else:
                    level = "😞 Complex"
                st.metric("Readability Level", level)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Sentence Count", sentence_count)
                # Quality with color coding
                quality_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[quality_label]
                st.metric("Quality Rating", f"{quality_emoji} {quality_label}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_sentence = word_count / max(sentence_count, 1)
                st.metric("Avg. Sentence Length", f"{avg_sentence:.1f} words")
                st.metric("Content Depth", "Comprehensive" if word_count > 2000 else "Standard")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Improved Similar Content
            st.markdown("### 🔗 Similar Content in Database")
            
            similar_pages = improved_similarity(url, body_text, enhanced_data, top_n=5)
            
            if similar_pages:
                st.write(f"Found {len(similar_pages)} potentially related pages:")
                
                for i, page in enumerate(similar_pages, 1):
                    similarity_percent = page['similarity'] * 100
                    with st.expander(f"{i}. {similarity_percent:.0f}% similar - {page['url']}"):
                        st.write(f"**Word Count:** {page['word_count']:,}")
                        st.write(f"**Quality:** {page['quality']}")
                        st.write(f"**Similarity Score:** {similarity_percent:.1f}%")
            else:
                st.info("No similar content found in the database. This appears to be unique content!")
            
            # Content Preview
            with st.expander("📄 View Extracted Text Preview", expanded=False):
                if body_text:
                    preview_length = min(1000, len(body_text))
                    preview = body_text[:preview_length] + ("..." if len(body_text) > preview_length else "")
                    st.text_area("Extracted Content", preview, height=200, key="preview")
                else:
                    st.warning("No text content extracted")

# Sidebar
with st.sidebar:
    st.markdown("### 📈 Database Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Pages", len(existing_data))
    with col2:
        st.metric("Avg Words", f"{existing_data['word_count'].mean():.0f}")
    
    st.metric("Most Common Quality", existing_data['quality_label'].mode().iloc[0] if not existing_data.empty else "N/A")
    
    # Quality distribution
    st.markdown("#### Quality Distribution")
    quality_dist = existing_data['quality_label'].value_counts()
    for quality, count in quality_dist.items():
        st.write(f"{quality}: {count} pages")
    
    st.markdown("---")
    st.markdown("### ℹ️ About This Tool")
    st.info("""
    This SEO analyzer uses machine learning to evaluate:
    - **Content Quality** (Low/Medium/High)
    - **Readability** (Flesch Reading Ease)
    - **Content Similarity** 
    - **Thin Content Detection**
    
    Perfect for content audits and competitive analysis!
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280;'>"
    "Built with ❤️ using Streamlit | SEO Content Analyzer v1.0"
    "</div>", 
    unsafe_allow_html=True
)