import numpy as np 
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob
import plotly.graph_objects as go
import streamlit as st

us_comments = pd.read_csv("datasets/UScomments.csv",on_bad_lines='skip',low_memory=False)
us_videos = pd.read_csv("datasets/USvideos.csv", on_bad_lines='skip',low_memory=False)

# Dropping Nan values
us_comments.dropna(inplace = True)

pol=[]  # list which will contain the polarity of the comments
for i in us_comments['comment_text']:
    try:
        analysis = TextBlob(i)
        pol.append(analysis.sentiment.polarity)
    except:
        pol.append(0)

us_comments['sentiment_polarity']=pol

THRESHOLD = 0.3
# Set rows with sentiment_polarity equal to 0 to 0
us_comments.loc[us_comments['sentiment_polarity'] == 0, 'sentiment_polarity'] = 0
us_comments.loc[us_comments['sentiment_polarity'] > 0, 'sentiment_polarity'] = 1
us_comments.loc[us_comments['sentiment_polarity'] < 0, 'sentiment_polarity'] = -1


# Considering Bad Words as negative comment
yuck_data = us_comments[us_comments['comment_text'].str.contains("fuck", na=False)]
# print(yuck_data[['comment_text', 'sentiment_polarity']][:20])


# Word Cloud - Positive 
df_positive = us_comments[us_comments.sentiment_polarity==1]
positives = (' '.join(df_positive['comment_text']))
st.subheader('Word Cloud of Positive Comments')
wordcloud = WordCloud(width=1000, height=500).generate(positives)
plt.figure(figsize=(15, 5))
plt.imshow(wordcloud)
plt.axis('off')
st.pyplot(plt)

# Word Cloud - Negative
df_negative = us_comments[us_comments.sentiment_polarity==-1]
negatives = (' '.join(df_negative['comment_text']))
st.subheader('Word Cloud of Negative Comments')
wordcloud = WordCloud(width=1000, height=500).generate(negatives)
plt.figure(figsize=(15, 5))
plt.imshow(wordcloud)
plt.axis('off')
st.pyplot(plt)

# Word Cloud - Neutral 
df_neutral = us_comments[us_comments.sentiment_polarity==0]
neutrals = (' '.join(df_neutral['comment_text'].astype(str)))
st.subheader('Word Clouds of Neutal Comments')
wordcloud_neutral = WordCloud(width=1000, height=500).generate(neutrals)
plt.figure(figsize=(15, 5))
plt.imshow(wordcloud_neutral)
plt.axis('off')
st.pyplot(plt)

# Sentiment Distributions (Positive/Negative/Neutral)
st.subheader('Distribution of Comment Sentiments')
us_comments['sentiment'] = us_comments['sentiment_polarity'].replace({1: 'Positive', 0: 'Neutral', -1: 'Negative'})
sentiment_counts = us_comments['sentiment'].value_counts()

fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index,
                             values=sentiment_counts.values,
                             textinfo='percent+label',
                             marker=dict(colors=['green', 'gray', 'red']))])
fig.update_layout(width=500, height=500)
st.plotly_chart(fig)

# Sentiment Subjectivity
subj = []  
for comment in us_comments['comment_text']:
    try:
        analysis = TextBlob(comment)
        subj.append(analysis.sentiment.subjectivity)
    except:
        subj.append(0)  # Fallback to a neutral subjectivity score when TextBlob fails
us_comments['sentiment_subjectivity'] = subj

THRESHOLD = 0.5  
us_comments.loc[us_comments['sentiment_subjectivity'] >= THRESHOLD, 'sentiment_subjectivity'] = 1
us_comments.loc[us_comments['sentiment_subjectivity'] < THRESHOLD, 'sentiment_subjectivity'] = 0
sentiment_counts = us_comments['sentiment_subjectivity'].value_counts()

st.subheader('Subjective vs. Objective Comments')
fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
             text=sentiment_counts.values,
             labels={'x': 'Comment Type', 'y': 'Number'},
             title='Types of Comments (Subjective vs. Objective)')

fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(showlegend=False, height=500, width=700, title_x=0.5)
st.plotly_chart(fig)

# Relationship between Views and Comments
st.subheader('Views vs Total Amount of Comments')
fig = px.scatter(us_videos, x='views', y='comment_total',
                 title='Views vs Total Amount of Comments',
                 labels={'views': 'Total Views', 'comment_total': 'Total Comments'})
st.plotly_chart(fig)

# Group comments by video ID and count polarities
comments_grouped = us_comments.groupby(['video_id', 'sentiment_polarity'])['sentiment_polarity'].count().unstack(fill_value=0)
comments_grouped = comments_grouped.rename(columns={-1: 'negative', 0: 'neutral', 1: 'positive'})
video_polarity = pd.merge(us_videos[['video_id', 'views', 'title']], comments_grouped, on='video_id', how='left').fillna(0)


# Sentiment Distrbution based on Views
# View vs Positive Comments
st.subheader('Views vs Positive and Negative Comments')
fig = px.scatter(video_polarity, x='views', y='positive',
                 title='Views vs Positive and Negative Comments',
                 labels={'views': 'Views', 'positive': 'Positive Comments'},
                 color='negative',  # Use the 'negative' column to define the color
                 color_continuous_scale=['red', 'darkblue'],  # Set the color scale
                 color_continuous_midpoint=0)  # Define midpoint for the color scale

fig.update_layout(xaxis_title='Views', yaxis_title='Positive Comments Amount',
                  coloraxis_colorbar=dict(title='Negative Comments'),
                  showlegend=True)
st.plotly_chart(fig)

# Distribution of negative comments
st.subheader('Distribution of Negative Comments')
fig = go.Figure(data=[go.Histogram(x=video_polarity['negative'], nbinsx=20, marker_color='green',
                                   name='Negative Comments',
                                   hovertemplate='Negative Comments: %{y}<extra></extra>')])
fig.update_layout(title='Distribution of Negative Comments',
                  xaxis_title='Negative Comments',
                  yaxis_title='Frequency')
st.plotly_chart(fig)

# Distribution of positive comments
st.subheader('Distribution of Positive Comments')
fig = go.Figure(data=[go.Histogram(x=video_polarity['positive'], nbinsx=20, marker_color='blue',
                                   name='Positive Comments',
                                   hovertemplate='Positive Comments: %{y}<extra></extra>')])
fig.update_layout(title='Distribution of Positive Comments',
                  xaxis_title='Positive Comments',
                  yaxis_title='Frequency')
st.plotly_chart(fig)






