import streamlit as st
import pandas as pd
import altair as alt
import base64
import nltk
import cleantext
import re
from sklearn.svm import SVC
from joblib import load
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_option_menu import option_menu
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from PIL import Image

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load image for page icon
img = Image.open('sephora.png')
st.set_page_config(page_title='Sentiment Analysis', page_icon=img)

# Load the pre-trained Random Forest model and count vectorizer
model = load('svc_model.pkl')
tfidf_vectorizer = load('tfidf_vectorizer.pkl')

# Function to clean and lemmatize text
def clean_and_lemmatize_text(text):
    # Remove emojis
    text = re.sub(r'[^\w\s,]', '', text)

    # Remove digits and non-word/space characters
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove consecutive repeating characters
    text = re.sub(r'(.)\1+', r'\1\1', text)

    # Clean the text using cleantext
    cleaned_text = cleantext.clean(
        text,
        clean_all=False,
        extra_spaces=True,
        stopwords=True,
        lowercase=True,
        numbers=True,
        punct=True
    )
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(cleaned_text)  # Tokenize the cleaned text
    stop_words = set(stopwords.words('english'))

    # Lemmatize each word and join them back into a sentence
    lemmatized_tokens = []
    for token in tokens:
        if token.endswith('ing'):
            # Example: running -> run
            lemma = lemmatizer.lemmatize(token, pos='v')
        elif token.endswith('ed'):
            # Example: walked -> walk
            lemma = lemmatizer.lemmatize(token, pos='v')
        else:
            lemma = lemmatizer.lemmatize(token)
        if lemma not in stop_words:
            lemmatized_tokens.append(lemma)
    return ' '.join(lemmatized_tokens)

# Function to analyze sentiment of individual tokens in text
def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.05:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= -0.05:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)
    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result

# Function to convert sentiment analysis result to DataFrame
def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

# Main function
def main():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Product Review Analyzer", "Product Recommendation"],
        icons=["house", "book", "heart"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "white"},
            "icon": {"color": "orange", "font-size": "12px"},
            "nav-link": {"font-size": "12px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "chocolate"},
        },
    )

    if selected == "Home":
        # Set background image
        page_bg_img = """
        <style>
        [data-testid="stAppViewContainer"] {
          background-image: url("https://images.preview.ph/preview/images/2021/12/22/preview-beauty-awards-skincare-nm.jpg ");
          background-size: cover;
        }

        [data-testid = "stHeader"] {
          background-color: rgba(0, 0, 0, 0);
        }
        </style>
        """

        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: blacks;'>Sentiment Analysis Beauty and Skincare Product Review</h1>", unsafe_allow_html=True)
        st.write("""
        This project aims to conduct sentiment analysis on beauty and skincare product reviews sourced from Sephora.com.
        The objective is to classify reviews as positive, negative or neutral.

        By categorizing the reviews, the analysis aims to provide insights into customer satisfaction and preferences, helping brands understand consumer opinions and can improve their products.
        This sentiment analysis can also guide potential buyers by highlighting the general sentiment towards various products, thus influencing purchasing decisions.

        Let's delve into the review analysis!!!
        """)

    elif selected == "Product Review Analyzer":
        page_bg_img = """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #EED8AE;
        }

        [data-testid = "stHeader"] {
          background-color: rgba(0, 0, 0, 0);
        }
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

        with st.form(key='nlpForm'):
            product_name = st.text_input("**Enter Product Name:**")
            product_review = st.text_area("**Enter Product Review:**")

            original_text = st.text_area("**Original Text:**")
            if original_text:
                # Clean and lemmatize the text
                lemmatized_text = clean_and_lemmatize_text(original_text)
                st.write("**Cleaned Text:**", lemmatized_text)

            submit_button = st.form_submit_button(label='**Analyze**')

        if submit_button:
            word_count = len(product_review.split())
            if word_count > 300:
                st.warning("**Product review exceeds 300 words. Please shorten your review.** :warning:")
            elif not product_review:
                st.warning("**Please enter product review to analyze!**:warning:")
            else:
                st.info("**Results**")

                # Ensure proper encoding before analysis
                product_review = product_review.encode('utf-8').decode('utf-8', 'ignore')

                # Use VADER to compute sentiment scores
                scores = sid.polarity_scores(product_review)
                compound_score = scores['compound']

                # Display the sentiment label with larger font size
                if compound_score > 0.05:
                    st.markdown("<h3 style='font-size: 20px; color: green;'>Sentiment: Positive 😊</h3>", unsafe_allow_html=True)
                elif compound_score < -0.05:
                    st.markdown("<h3 style='font-size: 20px; color: red;'>Sentiment: Negative 😡</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='font-size: 20px; color: yellow;'>Sentiment: Neutral 😐</h3>", unsafe_allow_html=True)

                # Create a DataFrame from the sentiment analysis
                sentiment = pd.DataFrame([{
                    'metric': 'compound',
                    'value': compound_score
                }])

                # Create and display a bar chart
                c = alt.Chart(sentiment).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric'
                )

                # Create columns to display the DataFrame and the chart side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(sentiment)
                with col2:
                    st.altair_chart(c, use_container_width=True)

        with st.expander('**Analyze CSV**'):
            upl = st.file_uploader('Upload file')

            def score(x):
                if isinstance(x, str):
                    scores = sid.polarity_scores(x)
                    return scores['compound']
                else:
                    return 0

            def analyze(x):
                if x > 0.05:
                    return 'Positive'
                elif x < -0.05:
                    return 'Negative'
                else:
                    return 'Neutral'

            if upl:
                df = pd.read_csv(upl)
                df['Score'] = df['review_text'].apply(score)
                df['analysis'] = df['Score'].apply(analyze)
                st.write(df.head())

                @st.cache_data
                def convert_df(df):
                    return df.to_csv().encode('utf-8')

                csv = convert_df(df)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='sentiment.csv',
                    mime='text/csv',
                )

    elif selected == "Product Recommendation":
        page_bg_img = """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #EED8AE;
        }

        [data-testid = "stHeader"] {
          background-color: rgba(0, 0, 0, 0);
        }
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

        # Automatically read and display data from the uploaded CSV file in Colab environment
        csv_path = 'top_10_brands_sentiment_summary.csv'
        df = pd.read_csv(csv_path)

        st.write("**Top10 recommended product:**")
        st.write(df.head(11))

if __name__ == '__main__':
    main()
