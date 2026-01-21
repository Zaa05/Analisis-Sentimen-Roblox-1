# app.py - Aplikasi Analisis Sentimen Roblox dengan Model BNB_TFIDF
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources (first time only)
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-eng')
except:
    pass

# Set page configuration
st.set_page_config(
    page_title="Analisis Sentimen Roblox",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E90FF;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive {
        color: #28a745;
        font-weight: bold;
        background-color: rgba(40, 167, 69, 0.1);
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    .neutral {
        color: #ffc107;
        font-weight: bold;
        background-color: rgba(255, 193, 7, 0.1);
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
        background-color: rgba(220, 53, 69, 0.1);
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1E90FF, #4169E1);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(30, 144, 255, 0.3);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
    }
    .stTextArea textarea:focus {
        border-color: #1E90FF;
        box-shadow: 0 0 0 0.2rem rgba(30, 144, 255, 0.25);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background-color: #f8f9fa;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E90FF !important;
        color: white !important;
    }
    
    /* Custom containers */
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #dc3545, #ffc107, #28a745);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'label_mapping' not in st.session_state:
    st.session_state.label_mapping = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'sentiment_counts' not in st.session_state:
    st.session_state.sentiment_counts = {'positif': 0, 'netral': 0, 'negatif': 0}

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text for sentiment analysis"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove emojis and special characters (keep only letters and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Load model function
@st.cache_resource(show_spinner="Memuat model dan vectorizer...")
def load_model():
    """Load the trained model and vectorizer"""
    try:
        # Load model
        model = joblib.load('bnb_tfidf.pkl')
        
        # Load vectorizer
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        # Load label mapping
        with open('label_mapping.json', 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        
        # Reverse mapping for display
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        
        return model, vectorizer, reverse_mapping
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Predict sentiment function
def predict_sentiment(text):
    """Predict sentiment for given text"""
    if st.session_state.model is None or st.session_state.vectorizer is None:
        return None, None
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Transform text using vectorizer
    text_vectorized = st.session_state.vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = st.session_state.model.predict(text_vectorized)[0]
    prediction_proba = st.session_state.model.predict_proba(text_vectorized)[0]
    
    # Map prediction to label
    sentiment = st.session_state.label_mapping.get(prediction, 'netral')
    confidence = max(prediction_proba)
    
    return sentiment, confidence, prediction_proba

# Sidebar navigation
with st.sidebar:
    # Logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/5968/5968520.png", width=80)
    
    st.markdown("<h2 style='text-align: center;'>🎮 Sentimen Analyzer</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load model section
    st.markdown("### 🔧 Model Configuration")
    
    if st.button("🚀 Load Model BNB_TFIDF", use_container_width=True):
        with st.spinner("Loading model and vectorizer..."):
            model, vectorizer, label_mapping = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.session_state.label_mapping = label_mapping
                st.success("✅ Model berhasil dimuat!")
            else:
                st.error("❌ Gagal memuat model")
    
    st.markdown("---")
    
    # Model info
    if st.session_state.model is not None:
        st.markdown("### 📊 Model Info")
        st.info(f"""
        **Model:** Bernoulli Naive Bayes  
        **Features:** TF-IDF Vectorization  
        **Classes:** Positif, Netral, Negatif  
        **Status:** ✅ Loaded
        """)
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📈 Statistics")
    st.metric("Total Predictions", st.session_state.total_predictions)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👍 Positif", st.session_state.sentiment_counts['positif'])
    with col2:
        st.metric("😐 Netral", st.session_state.sentiment_counts['netral'])
    with col3:
        st.metric("👎 Negatif", st.session_state.sentiment_counts['negatif'])
    
    st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_predictions = 0
        st.session_state.sentiment_counts = {'positif': 0, 'netral': 0, 'negatif': 0}
        st.success("History cleared!")
        st.rerun()

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🤖 Predict", "📊 Analytics", "ℹ️ About"])

# Tab 1: Home
with tab1:
    st.markdown('<h1 class="main-header">🎮 Roblox Sentiment Analyzer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
        Analisis sentimen otomatis untuk review dan komentar Roblox menggunakan model <strong>Bernoulli Naive Bayes dengan TF-IDF</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>⚡ Real-time Analysis</h3>
            <p>Prediksi sentimen secara instan dengan akurasi tinggi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Visualization</h3>
            <p>Visualisasi hasil dengan grafik interaktif</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>💾 History Tracking</h3>
            <p>Simpan dan analisis riwayat prediksi</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🚀 Quick Start Guide</h3>', unsafe_allow_html=True)
    
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    
    with guide_col1:
        st.markdown("""
        ### 1. Load Model
        Klik tombol **"Load Model"** di sidebar untuk memuat model yang telah ditraining
        """)
    
    with guide_col2:
        st.markdown("""
        ### 2. Input Text
        Masukkan teks review Roblox di tab **"Predict"**
        """)
    
    with guide_col3:
        st.markdown("""
        ### 3. Get Results
        Klik **"Analyze Sentiment"** untuk melihat hasil analisis
        """)
    
    # Model architecture
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🏗️ Model Architecture</h3>', unsafe_allow_html=True)
    
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*HrF0vLQ8W739e2-WXa3XKQ.png", 
             caption="Bernoulli Naive Bayes dengan TF-IDF Vectorization")
    
    st.markdown("""
    **Pipeline Process:**
    1. **Text Cleaning**: Remove URLs, HTML, special characters
    2. **Tokenization**: Split text into words
    3. **Stopword Removal**: Remove common words
    4. **TF-IDF Transformation**: Convert text to numerical features
    5. **BNB Classification**: Bernoulli Naive Bayes prediction
    6. **Result Mapping**: Map to sentiment labels
    """)

# Tab 2: Predict
with tab2:
    st.markdown('<h1 class="main-header">🤖 Sentiment Prediction</h1>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please load the model first from the sidebar!")
        st.info("Click the '🚀 Load Model BNB_TFIDF' button in the sidebar to load the trained model.")
    else:
        # Input section
        st.markdown('<h3 class="sub-header">📝 Input Text for Analysis</h3>', unsafe_allow_html=True)
        
        # Example texts
        example_texts = {
            "Positive Example": "This game is absolutely amazing! The graphics are stunning and the gameplay is very smooth. I love the new updates and the community is very friendly.",
            "Neutral Example": "The game is okay, nothing special. Sometimes it lags but it's playable. Could use some improvements in the user interface.",
            "Negative Example": "Very disappointing experience. The game crashes constantly and customer support is terrible. Waste of time and money."
        }
        
        # Example selector
        example_choice = st.selectbox(
            "Choose an example text:",
            ["Custom Input", "Positive Example", "Neutral Example", "Negative Example"]
        )
        
        if example_choice != "Custom Input":
            default_text = example_texts[example_choice]
        else:
            default_text = ""
        
        # Text input
        input_text = st.text_area(
            "Enter your Roblox review or comment:",
            value=default_text,
            height=150,
            placeholder="Type your text here...",
            help="Enter any text related to Roblox for sentiment analysis"
        )
        
        # Analyze button
        col1, col2, col3 = st.columns([3, 2, 3])
        with col2:
            analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)
        
        if analyze_btn and input_text:
            with st.spinner("Analyzing sentiment..."):
                # Get prediction
                sentiment, confidence, proba = predict_sentiment(input_text)
                
                if sentiment:
                    # Update statistics
                    st.session_state.total_predictions += 1
                    st.session_state.sentiment_counts[sentiment] += 1
                    
                    # Add to history
                    st.session_state.history.append({
                        'text': input_text[:100] + "..." if len(input_text) > 100 else input_text,
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'timestamp': pd.Timestamp.now()
                    })
                    
                    # Display results
                    st.markdown("---")
                    st.markdown('<h3 class="sub-header">📊 Analysis Results</h3>', unsafe_allow_html=True)
                    
                    # Result container
                    sentiment_color_class = {
                        'positif': 'positive',
                        'netral': 'neutral',
                        'negatif': 'negative'
                    }
                    
                    st.markdown(f"""
                    <div class='result-container'>
                        <h2 style='margin-bottom: 1rem;'>Predicted Sentiment</h2>
                        <h1 class='{sentiment_color_class[sentiment]}' style='font-size: 3rem; margin: 1rem 0;'>
                            {sentiment.upper()}
                        </h1>
                        <h3>Confidence: {confidence:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bars
                    st.markdown("#### 📈 Confidence Distribution")
                    
                    labels = ['Negatif', 'Netral', 'Positif']
                    colors = ['#dc3545', '#ffc107', '#28a745']
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=labels,
                            y=proba,
                            marker_color=colors,
                            text=[f'{p:.2%}' for p in proba],
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        yaxis=dict(range=[0, 1], tickformat=".0%"),
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Text analysis
                    st.markdown("#### 🔍 Text Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Text Length", len(input_text))
                    
                    with col2:
                        st.metric("Word Count", len(input_text.split()))
                    
                    with col3:
                        cleaned_text = preprocess_text(input_text)
                        st.metric("Cleaned Words", len(cleaned_text.split()))
                    
                    # Show cleaned text
                    with st.expander("📝 View Cleaned Text"):
                        st.code(cleaned_text)
                    
                    # Keywords extraction
                    st.markdown("#### 🔑 Keywords Detected")
                    words = cleaned_text.split()
                    word_freq = Counter(words)
                    top_words = word_freq.most_common(10)
                    
                    if top_words:
                        keywords_html = ""
                        for word, freq in top_words:
                            keywords_html += f"<span style='background-color: #e0e0e0; padding: 5px 10px; margin: 5px; border-radius: 15px; display: inline-block;'>{word} ({freq})</span>"
                        st.markdown(keywords_html, unsafe_allow_html=True)
        
        elif analyze_btn and not input_text:
            st.warning("⚠️ Please enter some text to analyze!")

# Tab 3: Analytics
with tab3:
    st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if st.session_state.total_predictions == 0:
        st.info("📈 No predictions yet. Analyze some text to see analytics!")
    else:
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", st.session_state.total_predictions)
        
        with col2:
            st.metric("Positive Rate", 
                     f"{(st.session_state.sentiment_counts['positif']/st.session_state.total_predictions*100):.1f}%")
        
        with col3:
            st.metric("Neutral Rate", 
                     f"{(st.session_state.sentiment_counts['netral']/st.session_state.total_predictions*100):.1f}%")
        
        with col4:
            st.metric("Negative Rate", 
                     f"{(st.session_state.sentiment_counts['negatif']/st.session_state.total_predictions*100):.1f}%")
        
        # Sentiment distribution chart
        st.markdown("---")
        st.markdown('<h3 class="sub-header">📈 Sentiment Distribution</h3>', unsafe_allow_html=True)
        
        sentiments = list(st.session_state.sentiment_counts.keys())
        counts = list(st.session_state.sentiment_counts.values())
        colors = ['#dc3545', '#ffc107', '#28a745']
        
        fig = go.Figure(data=[
            go.Pie(
                labels=sentiments,
                values=counts,
                hole=0.3,
                marker=dict(colors=colors),
                textinfo='percent+label',
                hoverinfo='value+percent'
            )
        ])
        
        fig.update_layout(
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # History table
        st.markdown("---")
        st.markdown('<h3 class="sub-header">📋 Prediction History</h3>', unsafe_allow_html=True)
        
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            
            # Add color coding for sentiment column
            def color_sentiment(val):
                if val == 'positif':
                    return 'color: #28a745'
                elif val == 'netral':
                    return 'color: #ffc107'
                else:
                    return 'color: #dc3545'
            
            # Display styled dataframe
            st.dataframe(
                history_df.style.map(color_sentiment, subset=['sentiment']),
                use_container_width=True,
                column_config={
                    'text': 'Text',
                    'sentiment': 'Sentiment',
                    'confidence': 'Confidence',
                    'timestamp': 'Timestamp'
                }
            )
            
            # Export option
            if st.button("📥 Export History as CSV"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="sentiment_history.csv",
                    mime="text/csv"
                )
        else:
            st.info("No prediction history available.")

# Tab 4: About
with tab4:
    st.markdown('<h1 class="main-header">ℹ️ About This Application</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎮 Roblox Sentiment Analyzer
        
        Aplikasi web ini dikembangkan untuk menganalisis sentimen dari review dan komentar
        pengguna Roblox menggunakan model **Bernoulli Naive Bayes dengan TF-IDF**.
        
        #### 🎯 Fitur Utama:
        
        1. **Real-time Sentiment Analysis** - Analisis sentimen secara instan
        2. **Multi-class Classification** - 3 kelas: Positif, Netral, Negatif
        3. **Interactive Visualizations** - Grafik dan chart interaktif
        4. **Prediction History** - Riwayat analisis tersimpan
        5. **Text Preprocessing** - Pembersihan teks otomatis
        
        #### 🔧 Teknologi yang Digunakan:
        
        - **Streamlit** - Web application framework
        - **Scikit-learn** - Machine learning library
        - **NLTK** - Natural Language Processing
        - **Plotly** - Interactive visualizations
        - **Joblib** - Model serialization
        
        #### 📊 Model Details:
        
        - **Algorithm**: Bernoulli Naive Bayes
        - **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Features**: Unigram + Bigram
        - **Max Features**: 5000
        - **Accuracy**: ~85% (on test set)
        
        #### 📁 File Structure:
        ```
        models_3class/
        ├── bnb_tfidf.pkl           # Trained model
        ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
        └── label_mapping.json      # Label mapping
        ```
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2107/2107736.png", width=200)
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;">
            <h4>🚀 Quick Stats</h4>
            <p><strong>Model Status:</strong> Bernoulli Naive Bayes</p>
            <p><strong>Vectorizer:</strong> TF-IDF</p>
            <p><strong>Classes:</strong> 3</p>
            <p><strong>Predictions:</strong> {}</p>
        </div>
        """.format(st.session_state.total_predictions), unsafe_allow_html=True)
    
    # Model workflow diagram
    st.markdown("---")
    st.markdown('<h3 class="sub-header">⚙️ Model Workflow</h3>', unsafe_allow_html=True)
    
    workflow_cols = st.columns(5)
    steps = [
        ("📥 Input Text", "User enters text"),
        ("🧹 Cleaning", "Remove URLs, HTML, etc."),
        ("🔡 Preprocessing", "Tokenization, stopwords"),
        ("🔢 Vectorization", "TF-IDF transformation"),
        ("🤖 Prediction", "BNB classification")
    ]
    
    for i, (step, desc) in enumerate(steps):
        with workflow_cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 10px; color: white; height: 120px; display: flex; flex-direction: column; justify-content: center;">
                <h3 style="margin: 0;">{step}</h3>
                <p style="font-size: 0.8rem; margin: 5px 0 0 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if i < 4:
            with workflow_cols[i]:
                st.markdown("<div style='text-align: center; font-size: 1.5rem;'>➡️</div>", unsafe_allow_html=True)
    

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🎮 Roblox Sentiment Analyzer**")

with footer_col2:
    st.markdown(f"**🕐 Last Updated:** {pd.Timestamp.now().strftime('%d %B %Y')}")

with footer_col3:
    st.markdown("**⚡ Powered by:** Streamlit + BernoulliNB")
