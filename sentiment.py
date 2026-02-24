import base64
import os
import pickle
import requests
from datetime import datetime
import warnings

import cohere
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer

from textblob import TextBlob
from wordcloud import WordCloud
from bs4 import BeautifulSoup

# For backward compatibility
from packaging import version

from utils import get_embeddings, seed_everything, streamlit_header_and_footer_setup

warnings.filterwarnings("ignore")
seed_everything(3777)

# =============================
# Streamlit Setup
# =============================
st.set_page_config(
    layout="wide",
    page_title="Advanced Sentiment Analysis",
    page_icon="🎭",
    initial_sidebar_state="expanded"
)

streamlit_header_and_footer_setup()

st.markdown("# 🎭 Advanced Sentiment Analysis")
st.markdown("### Understand emotions in text with multi-label classification")

MODEL_PATH = "./data/models/emotion_chain.pkl"

default_classes_mapping = {
    0: 'Anger',
    1: 'Anticipation',
    2: 'Disgust',
    3: 'Fear',
    4: 'Joy',
    5: 'Sadness',
    6: 'Surprise',
    7: 'Trust'
}

emotion_colors = {
    'Anger': '#FF4B4B',
    'Anticipation': '#FFA500',
    'Disgust': '#8B4513',
    'Fear': '#800080',
    'Joy': '#FFD700',
    'Sadness': '#4169E1',
    'Surprise': '#FF69B4',
    'Trust': '#32CD32'
}

# =============================
# EMOTION IMAGES MAPPING
# =============================
def load_emotion_images():
    emotions2image_mapping = {
        'Anger': './data/emotions/anger.gif',
        'Anticipation': './data/emotions/anticipation.gif',
        'Disgust': './data/emotions/disgust.gif',
        'Fear': './data/emotions/fear.gif',
        'Joy': './data/emotions/joy.gif',
        'Sadness': './data/emotions/sadness.gif',
        'Surprise': './data/emotions/surprise.gif',
        'Trust': './data/emotions/trust.gif',
    }
    
    # Load images if they exist
    for key, value in emotions2image_mapping.items():
        if os.path.exists(value):
            with open(value, "rb") as f:
                emotions2image_mapping[key] = f.read()
        else:
            emotions2image_mapping[key] = None
    
    return emotions2image_mapping

# =============================
# CREATE SAMPLE DATA (if needed)
# =============================
def create_sample_data():
    """Create a sample training data file for testing"""
    sample_data = {
        "0": {
            "text": "I'm so happy and excited about this wonderful day!",
            "labels_text": ["Joy", "Anticipation"],
            "embeddings": np.random.randn(768).tolist()
        },
        "1": {
            "text": "This makes me angry and frustrated with the situation.",
            "labels_text": ["Anger", "Disgust"],
            "embeddings": np.random.randn(768).tolist()
        },
        "2": {
            "text": "I'm feeling sad and lonely after hearing the news.",
            "labels_text": ["Sadness", "Fear"],
            "embeddings": np.random.randn(768).tolist()
        },
        "3": {
            "text": "Wow! That's amazing and surprising news!",
            "labels_text": ["Surprise", "Joy"],
            "embeddings": np.random.randn(768).tolist()
        },
        "4": {
            "text": "I trust you completely with this important task.",
            "labels_text": ["Trust"],
            "embeddings": np.random.randn(768).tolist()
        },
        "5": {
            "text": "I fear what might happen in the future.",
            "labels_text": ["Fear"],
            "embeddings": np.random.randn(768).tolist()
        },
        "6": {
            "text": "The disgusting smell made me feel sick.",
            "labels_text": ["Disgust"],
            "embeddings": np.random.randn(768).tolist()
        },
        "7": {
            "text": "I anticipate great things from this project.",
            "labels_text": ["Anticipation"],
            "embeddings": np.random.randn(768).tolist()
        },
        "8": {
            "text": "Oh, brilliant. My car broke down in the pouring rain right before my final exam.",
            "labels_text": ["Anger", "Sadness"],
            "embeddings": np.random.randn(768).tolist()
        },
        "9": {
            "text": "I can't believe I won the lottery! This is the best day ever!",
            "labels_text": ["Joy", "Surprise"],
            "embeddings": np.random.randn(768).tolist()
        }
    }
    
    # Add more variations
    for i in range(10, 50):
        emotions = list(default_classes_mapping.values())
        import random
        num_emotions = random.randint(1, 3)
        selected_emotions = random.sample(emotions, num_emotions)
        
        sample_data[str(i)] = {
            "text": f"Sample text {i} with multiple emotions",
            "labels_text": selected_emotions,
            "embeddings": np.random.randn(768).tolist()
        }
    
    os.makedirs("./data", exist_ok=True)
    df = pd.DataFrame.from_dict(sample_data, orient='index')
    df.to_json("./data/xed_with_embeddings.json", orient='index')
    return True

# =============================
# TRAIN MODEL (SKLEARN ≥1.7 SAFE)
# =============================
def train_and_save():
    try:
        data_path = "./data/xed_with_embeddings.json"
        if not os.path.exists(data_path):
            st.warning("Training data not found. Creating sample data...")
            create_sample_data()
            st.info("✅ Sample data created!")

        with st.spinner("Loading training data..."):
            df = pd.read_json(data_path, orient="index")
            st.sidebar.info(f"Loaded {len(df)} training samples")

        mlb = MultiLabelBinarizer()
        X = np.array(df.embeddings.tolist())
        y = mlb.fit_transform(df.labels_text)

        classes_mapping = {i: c for i, c in enumerate(mlb.classes_)}
        st.sidebar.info(f"Classes: {list(classes_mapping.values())}")

        os.makedirs("./data/models", exist_ok=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        base_lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=0
        )

        # Use the correct parameter name based on sklearn version
        sklearn_version = sklearn.__version__
        if version.parse(sklearn_version) >= version.parse("1.7.0"):
            # Newer versions use 'estimator'
            chain = ClassifierChain(
                estimator=base_lr,
                order="random",
                random_state=0
            )
        else:
            # Older versions use 'base_estimator'
            chain = ClassifierChain(
                base_estimator=base_lr,
                order="random",
                random_state=0
            )

        with st.spinner("Training model..."):
            chain.fit(X_train, y_train)

        accuracy = chain.score(X_test, y_test)
        st.sidebar.success(f"✅ Model accuracy: {accuracy:.3f}")

        model_data = {
            "model": chain,
            "mlb": mlb,
            "classes_mapping": classes_mapping,
            "sklearn_version": sklearn_version,
            "accuracy": accuracy
        }

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_data, f)

        return True

    except Exception as e:
        st.error(f"❌ Training error: {str(e)}")
        return False


# =============================
# SAFE MODEL LOADING
# =============================
@st.cache_resource
def load_model():
    chain_model = None
    mlb = None
    classes_mapping = default_classes_mapping.copy()

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model_data = pickle.load(f)

            # Check if it's the new dictionary format
            if isinstance(model_data, dict):
                chain_model = model_data["model"]
                mlb = model_data.get("mlb")
                classes_mapping = model_data.get(
                    "classes_mapping", default_classes_mapping
                )
                
                # Handle sklearn version compatibility
                if hasattr(chain_model, 'base_estimator') and not hasattr(chain_model, 'estimator'):
                    chain_model.estimator = chain_model.base_estimator
                
                # Test prediction
                test_input = np.random.rand(1, 768)
                _ = chain_model.predict_proba(test_input)
                
            else:
                # Old format - try to convert
                chain_model = model_data
                if hasattr(chain_model, 'base_estimator') and not hasattr(chain_model, 'estimator'):
                    chain_model.estimator = chain_model.base_estimator
                
                # Test prediction
                test_input = np.random.rand(1, 768)
                _ = chain_model.predict_proba(test_input)
                
                st.sidebar.warning("⚠️ Using legacy model format. Consider retraining.")

        except Exception as e:
            st.warning(f"⚠️ Incompatible model detected: {str(e)}. Please retrain.")
            if os.path.exists(MODEL_PATH):
                backup_path = MODEL_PATH + ".backup"
                os.rename(MODEL_PATH, backup_path)
                st.info(f"Old model backed up to {backup_path}")
            chain_model = None

    return chain_model, mlb, classes_mapping


# =============================
# TEXT STATISTICS
# =============================
def analyze_text_statistics(text):
    """Analyze text statistics"""
    try:
        blob = TextBlob(text)
        
        stats = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "sentence_count": len(blob.sentences),
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
        
        # Sentiment category
        if stats["polarity"] > 0.1:
            stats["sentiment"] = "Positive 😊"
        elif stats["polarity"] < -0.1:
            stats["sentiment"] = "Negative 😔"
        else:
            stats["sentiment"] = "Neutral 😐"
        
        return stats
    except:
        # Fallback if TextBlob fails
        return {
            "word_count": len(text.split()),
            "char_count": len(text),
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            "polarity": 0,
            "subjectivity": 0,
            "sentiment": "Unknown"
        }


# =============================
# URL TEXT EXTRACTION
# =============================
def extract_text_from_url(url):
    """Extract text content from a URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # Limit to 5000 characters
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return None


# =============================
# EMBEDDINGS
# =============================
def get_embeddings_with_fallback(text, co_client):
    try:
        if co_client:
            response = co_client.embed(
                texts=[text],
                model="multilingual-22-12"
            )
            return np.array(response.embeddings)
        else:
            return None
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None


# =============================
# ANALYZE TEXT
# =============================
def analyze_text(text, co_client, chain_model):
    embeddings = get_embeddings_with_fallback(text, co_client)
    if embeddings is None or chain_model is None:
        return None

    try:
        probs = chain_model.predict_proba(embeddings)[0]
        indices = np.argsort(probs)[::-1]
        return probs, indices
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# =============================
# CREATE CHARTS
# =============================
def create_emotion_chart(emotions, probabilities, chart_type="Bar Chart"):
    df = pd.DataFrame({
        'Emotion': emotions,
        'Probability': probabilities
    })
    
    if chart_type == "Bar Chart":
        fig = px.bar(df, x='Emotion', y='Probability', color='Emotion',
                     color_discrete_map=emotion_colors, 
                     title="Emotion Probabilities",
                     range_y=[0, 1])
        fig.update_layout(showlegend=False)
        
    elif chart_type == "Pie Chart":
        fig = px.pie(df, values='Probability', names='Emotion', 
                     color='Emotion', color_discrete_map=emotion_colors,
                     title="Emotion Distribution")
        
    else:  # Radar Chart
        categories = emotions.tolist() if isinstance(emotions, np.ndarray) else emotions
        values = probabilities.tolist()
        values += values[:1]  # Complete the loop
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            line_color='blue'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Emotion Radar Chart"
        )
    
    return fig


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.header("⚙️ Configuration")
    COHERE_API_KEY = st.text_input("Cohere API Key", type="password", 
                                   help="Enter your Cohere API key for embeddings")
    co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None
    
    st.markdown("---")
    st.header("📊 Visualization Settings")
    chart_type = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart", "Radar Chart"])
    
    st.markdown("---")
    st.header("🔄 Model Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Create Data", use_container_width=True):
            with st.spinner("Creating sample data..."):
                if create_sample_data():
                    st.success("✅ Sample data created!")
    
    with col2:
        if st.button("🚀 Train Model", use_container_width=True):
            if train_and_save():
                st.success("✅ Model trained successfully!")
                st.rerun()
    
    # Show model status
    st.markdown("---")
    st.header("📊 Model Status")
    chain_model, mlb, classes_mapping = load_model()
    if chain_model is not None:
        st.success("✅ Model loaded")
        if hasattr(chain_model, 'estimators_'):
            n_classes = len(chain_model.estimators_)
            st.info(f"Number of emotions: {n_classes}")
            st.info(f"Emotions: {', '.join(list(classes_mapping.values())[:n_classes])}")
    else:
        st.warning("⚠️ No model loaded")

# Load emotion images
emotions2image_mapping = load_emotion_images()

# =============================
# MAIN UI TABS
# =============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Text Input", 
    "🔗 URL", 
    "📋 Batch Processing",
    "ℹ️ About"
])

# Tab 1: Text Input
with tab1:
    st.subheader("Enter Text for Emotion Analysis")
    
    # Example text
    example_text = """Oh, brilliant. My car broke down in the pouring rain right before my final exam. This is exactly how I wanted to spend my morning."""
    
    input_text = st.text_area("Your Text Here", value=example_text, height=150, key="text_input")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.number_input("Show Top N Emotions", min_value=1, max_value=8, value=4, key="text_topk")
    with col2:
        analyze_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    if analyze_btn and input_text:
        if chain_model is None:
            st.error("❌ No model loaded. Please train a model first (click 'Train Model' in sidebar).")
        elif co is None:
            st.error("❌ Please enter your Cohere API key in the sidebar.")
        else:
            with st.spinner("Analyzing text..."):
                result = analyze_text(input_text, co, chain_model)
                
                if result:
                    probs, indices = result
                    
                    # Get actual number of emotions from the model
                    n_emotions = len(probs)
                    
                    # Adjust top_k to not exceed available emotions
                    actual_top_k = min(top_k, n_emotions)
                    
                    if actual_top_k < top_k:
                        st.info(f"ℹ️ Model only predicts {n_emotions} emotions. Showing top {actual_top_k}.")
                    
                    # Display emotion images and metrics
                    st.subheader("🎭 Top Emotions")
                    
                    # Create columns based on actual_top_k (max 4 per row)
                    cols_per_row = min(actual_top_k, 4)
                    cols = st.columns(cols_per_row)
                    
                    for i in range(actual_top_k):
                        col_idx = i % cols_per_row
                        idx = indices[i]
                        emotion = classes_mapping[idx]
                        prob = probs[idx]
                        
                        with cols[col_idx]:
                            # Display emotion image if available
                            if emotions2image_mapping.get(emotion):
                                try:
                                    image_data = emotions2image_mapping[emotion]
                                    if image_data:
                                        image_gif = base64.b64encode(image_data).decode("utf-8")
                                        st.markdown(
                                            f'<img src="data:image/gif;base64,{image_gif}" style="width:100%;border-radius: 15px;">',
                                            unsafe_allow_html=True,
                                        )
                                except Exception:
                                    pass  # Skip image if there's an error
                            
                            # Display metric
                            st.metric(emotion, f"{prob*100:.1f}%")
                    
                    # Chart - use actual available emotions
                    if actual_top_k > 0:
                        st.subheader("📊 Emotion Distribution")
                        emotions_list = [classes_mapping[indices[i]] for i in range(actual_top_k)]
                        probs_list = [probs[indices[i]] for i in range(actual_top_k)]
                        fig = create_emotion_chart(emotions_list, probs_list, chart_type)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Text Statistics
                    st.subheader("📈 Text Statistics")
                    stats = analyze_text_statistics(input_text)
                    
                    # Display stats in columns
                    stat_cols = st.columns(5)
                    stat_cols[0].metric("Words", stats['word_count'])
                    stat_cols[1].metric("Characters", stats['char_count'])
                    stat_cols[2].metric("Sentences", stats['sentence_count'])
                    stat_cols[3].metric("Polarity", f"{stats['polarity']:.2f}")
                    stat_cols[4].metric("Subjectivity", f"{stats['subjectivity']:.2f}")
                    
                    # Sentiment
                    st.info(f"📌 Overall Sentiment: {stats['sentiment']}")
                    
                    # Wordcloud (if enough text)
                    if len(input_text.split()) > 3:
                        st.subheader("☁️ Word Cloud")
                        try:
                            wc = WordCloud(width=800, height=400, background_color='white').generate(input_text)
                            fig2, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig2)
                        except Exception as wc_error:
                            st.warning("Could not generate word cloud.")

# Tab 2: URL Input
with tab2:
    st.subheader("Extract Text from URL")
    url = st.text_input("Enter URL", placeholder="https://example.com/article")
    
    if st.button("🔗 Extract and Analyze", type="primary") and url:
        if chain_model is None:
            st.error("❌ No model loaded. Please train a model first.")
        elif co is None:
            st.error("❌ Please enter your Cohere API key in the sidebar.")
        else:
            with st.spinner("Extracting text from URL..."):
                extracted_text = extract_text_from_url(url)
                if extracted_text:
                    st.success("✅ Text extracted successfully!")
                    with st.expander("View extracted text"):
                        st.text(extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""))
                    
                    with st.spinner("Analyzing text..."):
                        result = analyze_text(extracted_text, co, chain_model)
                        
                        if result:
                            probs, indices = result
                            n_emotions = len(probs)
                            display_count = min(4, n_emotions)
                            
                            st.subheader("🎭 Top Emotions")
                            cols = st.columns(display_count)
                            for i in range(display_count):
                                idx = indices[i]
                                emotion = classes_mapping[idx]
                                prob = probs[idx]
                                
                                with cols[i]:
                                    if emotions2image_mapping.get(emotion):
                                        try:
                                            image_data = emotions2image_mapping[emotion]
                                            if image_data:
                                                image_gif = base64.b64encode(image_data).decode("utf-8")
                                                st.markdown(
                                                    f'<img src="data:image/gif;base64,{image_gif}" style="width:100%;border-radius: 15px;">',
                                                    unsafe_allow_html=True,
                                                )
                                        except Exception:
                                            pass
                                    st.metric(emotion, f"{prob*100:.1f}%")

# Tab 3: Batch Processing
with tab3:
    st.subheader("Batch Text Processing")
    st.info("Enter multiple texts (one per line) for batch analysis")
    
    batch_texts = st.text_area("Enter texts (one per line)", height=200,
                                value="I'm feeling great today!\nThis is disappointing.\nI can't believe it!\nOh no, my car broke down!")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        batch_top_k = st.number_input("Top Emotions per text", min_value=1, max_value=8, value=3, key="batch_topk")
    with col2:
        process_btn = st.button("🔄 Process Batch", type="primary", use_container_width=True)
    
    if process_btn and batch_texts:
        if chain_model is None:
            st.error("❌ No model loaded. Please train a model first.")
        elif co is None:
            st.error("❌ Please enter your Cohere API key in the sidebar.")
        else:
            texts = [t.strip() for t in batch_texts.split('\n') if t.strip()]
            
            if texts:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, text in enumerate(texts):
                    status_text.text(f"Processing text {i+1}/{len(texts)}...")
                    result = analyze_text(text, co, chain_model)
                    if result:
                        results.append((text, result[0], result[1]))
                    progress_bar.progress((i + 1) / len(texts))
                
                status_text.empty()
                progress_bar.empty()
                
                if results:
                    st.success(f"✅ Successfully processed {len(results)} texts")
                    
                    # Display individual results
                    for j, (text, probs, indices) in enumerate(results):
                        n_emotions = len(probs)
                        display_count = min(batch_top_k, n_emotions)
                        
                        with st.expander(f"📝 Text {j+1}: {text[:50]}..."):
                            cols = st.columns(display_count)
                            for k in range(display_count):
                                idx = indices[k]
                                emotion = classes_mapping[idx]
                                prob = probs[idx]
                                cols[k].metric(emotion, f"{prob*100:.1f}%")
                    
                    # Comparison chart
                    st.subheader("📊 Batch Comparison")
                    comparison_data = []
                    for j, (text, probs, indices) in enumerate(results):
                        n_emotions = len(probs)
                        display_count = min(3, n_emotions)  # Top 3 for comparison
                        for k in range(display_count):
                            comparison_data.append({
                                'Sample': f"Sample {j+1}",
                                'Emotion': classes_mapping[indices[k]],
                                'Probability': probs[indices[k]]
                            })
                    
                    if comparison_data:
                        df_comp = pd.DataFrame(comparison_data)
                        fig_comp = px.bar(df_comp, x='Sample', y='Probability', color='Emotion',
                                         color_discrete_map=emotion_colors,
                                         title="Batch Comparison (Top 3 Emotions)",
                                         barmode='group')
                        st.plotly_chart(fig_comp, use_container_width=True)

# Tab 4: About
with tab4:
    st.subheader("ℹ️ About This App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Features:
        - **Multi-label emotion classification** - Detects multiple emotions simultaneously
        - **Multiple input methods** - Direct text, URLs, Batch processing
        - **Text statistics** - Word count, sentiment analysis, readability metrics
        - **Interactive visualizations** - Bar, Pie, and Radar charts
        - **Word cloud generation** - Visual text representation
        """)
    
    with col2:
        st.markdown("""
        ### 🎭 Supported Emotions:
        - 😠 **Anger** - #FF4B4B
        - 🎯 **Anticipation** - #FFA500
        - 🤢 **Disgust** - #8B4513
        - 😨 **Fear** - #800080
        - 😊 **Joy** - #FFD700
        - 😢 **Sadness** - #4169E1
        - 😲 **Surprise** - #FF69B4
        - 🤝 **Trust** - #32CD32
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 How to Use:
    1. **Enter your Cohere API key** in the sidebar
    2. **Create sample data** (if you don't have training data)
    3. **Train the model** by clicking "Train Model"
    4. **Choose an input method** from the tabs above
    5. **Enter your text** and click Analyze
    
    ### 📁 Data Requirements:
    - Training data should be at: `./data/xed_with_embeddings.json`
    - Emotion GIFs (optional) at: `./data/emotions/`
    
    ### 🔧 Troubleshooting:
    - If model won't load, click "Train Model" to retrain
    - Make sure your Cohere API key is valid
    - Check that training data exists in the correct location
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 10px;'>"
    "Made with 🐾 using Streamlit and Cohere"
    "</div>", 
    unsafe_allow_html=True
)