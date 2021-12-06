import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import streamlit as st
from textblob import TextBlob

st.title("5-Minute Crafts:Do Video Titles Relate to Views?")
st.markdown("**Remi Inoue**: [GitHub Repository](https://github.com/remii11/Math10_FinalProject)")
st.subheader("Dataset:")
st.markdown("[5-Minute Crafts: Video Clickbait Titles?](https://www.kaggle.com/shivamb/5minute-crafts-video-views-dataset)")
#intro
st.write("5-Minute Crafts is a popular channel on YouTube, with 74.6 million subscribers. The channel is often criticized for its misleading aspects such as the thumbnails, that can be considered as 'clickbait'. However, these factors most likely contribute to the large amount of views and video engagement that the channel generates. Although the video thumbnail is a significant factor towards audience engagement, a viewer is likely to be convinced by the title as well. In this project, I will be analyzing the correlation between a variety of aspects of video titles relative to its popularity")
df = pd.read_csv("5-Minute Crafts.csv")
st.header("Data Cleaning and Confounding Variables")
st.write("Although we are studying effectiveness of a video title relative to its popularity, the dataset contains other elements that can also affect the popularity as well. Therefore, we will examine the influence of other elements, and minimize this as much as possible in order to keep other categories constant, as not to wrongly influence the results based on title. ")
#linear regression: active_since_days
st.subheader("Confounding Variable: Recency")
st.write("In this dataset, the growth rate of the video is not given. Therefore, it can be said that the popularity of a video is unreliable when newer videos have lower views. To confirm this, below is a chart showing the correlation between a video's number of active days and view count at select points. ")
day_range = st.slider("Select a range",1,100,(1,100))
def make_chart(df,day_range):
    reg = LinearRegression()
    df1 = df[(df["active_since_days"]>=day_range[0])&(df["active_since_days"]<=day_range[1])].copy()
    X = np.array(df1["active_since_days"]).reshape(-1,1)
    y = np.array(df1["total_views"]).reshape(-1,1)
    reg.fit(X,y)
    df1["predict"]= reg.predict(X)
    corl = df1[['active_since_days','total_views']].corr().iloc[0,1]
    y_true = alt.Chart(df1).mark_circle(color= 'black').encode(
        x= 'active_since_days',
        y= 'total_views'
    )
    y_predict = alt.Chart(df1).mark_line(color= 'green').encode(
        x= 'active_since_days',
        y= 'predict',
    )
    params = alt.Chart(df1).transform_regression(
    'active_since_days', 'total_views', params=True
    ).mark_text(align='left').encode(
    x=alt.value(20),
    y=alt.value(20),
    text=alt.value(f"r: {corl:.3f}")
)
    return y_true+y_predict+params
try:
    linreg_chart = make_chart(df,day_range)
    st.altair_chart(linreg_chart)
except:
    st.write("There are no values in this range")
st.write("From testing different values in this chart, it is evident that there is a growth in views relative to the number of days active. In particular, the correlation coefficient is strongest in between 1 and 6 days. From a starting point of 1 active day, the correlation declines significantly at 20 days. Therefore, I will be modifying the dataset to exclude videos that have been active for less than 20 days.")

# standard deviation and mean
st.subheader("Confounding Variable:Duration ")
st.write("Another confounding variable is duration. Whether or not a user clicks on a video is in part reliant on this aspect, as users may be hesitant to watch longer or shorter videos.The chart below shows the duration relative to total views, with no correlation for the most part. However, it is evident that most of the videos within one standard deviation fall into the same 'duration category', and anything outside of that can be considered as within a different category. This is because longer videos are most likely 'compliations' of previous videos, and shorter videos are 'shorts'. Although there is no correlation evident from the chart, I will be cleaning the dataset to remove all videos not within one standard deviation of the mean, in order to keep the videos' 'duration category' constant. ")
mean = df["duration_seconds"].mean()
stdev = df["duration_seconds"].std()
stdev1 = mean +stdev
stdev1b = mean -stdev
range_ = ['red','green']
domain_ = ['1 standard deviation','mean']
scatter_plot = alt.Chart(df).mark_circle().encode(
    alt.X('duration_seconds'),
    alt.Y('total_views'),
)

mean = pd.DataFrame({
    'duration_seconds': [930,930],
    'total_views':  [0, 300000000],
})

mean_line = alt.Chart(mean).mark_line(color = 'green').encode(
    x= 'duration_seconds',
    y= 'total_views',
)

stdev1_line = pd.DataFrame({
    'duration_seconds': [210, 210],
    'total_views':  [0, 300000000],
    'Key': "1 standard deviation"
})

stdev1_plot = alt.Chart(stdev1_line).mark_line().encode(
    x= 'duration_seconds',
    y= 'total_views',
    color = alt.Color('Key', scale = alt.Scale(domain = domain_,range = range_))
)

stdev1_lineb = pd.DataFrame({
    'duration_seconds': [1650, 1650],
    'total_views':  [0, 300000000],
    
})

stdev1_plotb = alt.Chart(stdev1_lineb).mark_line(color = 'red').encode(
    x= 'duration_seconds',
    y= 'total_views',
)

st.altair_chart(scatter_plot + stdev1_plot  + stdev1_plotb + mean_line)

# final dataset
sub_df=df.copy()
df= df[(df['duration_seconds']<stdev1)&(df['duration_seconds']>stdev1b)&
       (df['active_since_days']>=20)]
df = df.drop(["duration_seconds","active_since_days"],axis = 1)


#keywords
st.header("Keywords")
st.write("When deciding on whether or not to click on a video, certain keywords may influence this. This section will analyze the existence of a correlation between a video's popularity and its use of 'keywords'.")
keyword_df = df.copy()
#returns title converted to a list of words
def getwords(title):
    title  = list(title)
    titlelist = title
    wordlist = []
    wordcount = 1
    for k in range(len(title)):
        if title[k]== " ":
            wordcount = wordcount +1
    for j in range(wordcount):
        word=[]
        for i in range(len(titlelist)):
            if titlelist[i]==" ":
                titlelist.remove(" ")
                titlelist = titlelist[i:len(titlelist)]
                break
            word.append(titlelist[i])
        wordlist.append(''.join(word))
    return wordlist
keyword_df["word_list"]=keyword_df["title"].apply(getwords)
#removes special characters from title
def remove_special_chars(title):
    for i in range(len(title)):
        word = list(title[i])
        word = [x for x in word if x.isalpha()==True]
        word = ''.join(word)
        title[i]=word
    return [x for x in title if x != '']
keyword_df["word_list"]=keyword_df["word_list"].apply(remove_special_chars)

#makes a list of all words used, and removes 'nonkeywords'
A= list(keyword_df["word_list"])
A1 = A[0]+A[1]
for i in range(2,len(A)):
    A1 = A1 +A[i]
nonkeywords = ["AND","TO","YOUR","YOU","FOR","WILL","THAT"]
for i in range(len(A1)):
    for j in range(len(nonkeywords)):
        if nonkeywords[j]==A1[i]:
            A1[i]= " "
A1 = [x for x in A1 if x != " "]
A2 = pd.DataFrame(A1)
#value_counts() of top 10 keywords used
frequency = list(A2.value_counts().iloc[0:10])
keywords = ["HACKS","LIFE","IDEAS","DIY","TRICKS","MAKE","EASY","COOL","KNOW","TIPS"]
topkeywords = pd.DataFrame({
    'Keyword': ["HACKS","LIFE","IDEAS","DIY","TRICKS","MAKE","EASY","COOL","KNOW","TIPS"],
    'Frequency': frequency
})
topkeywords.index = [1,2,3,4,5,6,7,8,9,10]
st.subheader("Top 10 Keywords Used")
st.write("Based on the top 10 most popular keywords utilized by the channel, it is evident that the word 'HACKS' is significantly the most common keyword used. ")
st.table(topkeywords.style.apply(lambda x: ['color:white;background-color:green' 
                                   if x.name in [1] else '' for i in x], 
                        axis=1))
#returns number of keywords
def numkeywords(wordlist):
    keywordcount = 0
    for i in range(len(wordlist)):
        for j in range(len(keywords)):
            if keywords[j]==wordlist[i]:
                keywordcount = keywordcount +1
    return keywordcount   
def mainkeyword(title):
    for i in range(len(keywords)):
        for j in range(len(title)):
            if title[j]==keywords[i]:
                mainkeyword_ = keywords[i] 
                return mainkeyword_
keyword_df["num_keywords"]= keyword_df["word_list"].apply(numkeywords)
keyword_df["main_keyword"]= keyword_df["word_list"].apply(mainkeyword)
def none(x):
    if x is None:
        return "NONE"
    else:
        return x
def categorize(x):
    if x == "HACKS":
        return "HACKS"
    if x in keywords[1:]:
        return "OTHER KEYWORD"
    else:
        return "NONE"
keyword_df["main_keyword"]=keyword_df["main_keyword"].apply(none)
keyword_df["keyword_category"]=keyword_df["main_keyword"].apply(categorize)

st.subheader("Does Using Keywords Lead to More Views?")
brush = alt.selection_interval()
chart = alt.Chart(keyword_df).mark_circle().encode(
    y='total_views:Q',
    color=alt.condition(brush, 'keyword_category:N', alt.value('lightgray'))
).properties(
    width=250,
    height=250
).add_selection(
    brush
)

#K-Means
st.write("Below is an interactive chart comparing aspects of the keywords,such as the number of keywords and total characters in the title. Each video was placed into a category based on the most popular keyword that it contained. Since the word 'HACKS' was used significantly more than any keyword, it was given its own category. Using numerous keywords or writing longer titles does not seem to affect the view counts as much. However, it is very evident from the chart that the most popular videos correspond to using the word: 'HACKS'.")
st.subheader("Keyword Usage Relative to Total Views")
st.altair_chart(chart.encode(x='num_keywords:Q') | 
                chart.encode(x='num_chars:Q'))
num_cols_keyword = [c for c in keyword_df.columns if is_numeric_dtype(keyword_df[c])]
kmeans = KMeans(3)
kmeans.fit(keyword_df[num_cols_keyword])
keyword_df["cluster"]=kmeans.predict(keyword_df[num_cols_keyword])

st.header("Analysis with K-Means Clustering")
st.write("Although we can conclude that the use of the word 'HACKS' in the title correlates to higher view counts,there does not seem to be any other factors that consistently affect the total views. Even when using K-Means Clustering, it is evident that the videos have only been categorized based on total views, as there are no consistent classifications beyond this element.")
cols1 = ['num_chars','num_words','num_punctuation','num_words_uppercase']
cols2 = ['num_words_lowercase','num_stopwords','avg_word_len','contain_digits']
cols3 = ['startswith_digits','title_sentiment','num_keywords']
chart1=alt.Chart(keyword_df).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y("total_views", type='quantitative'),
    color = "cluster:N"
).properties(
    width=100,
    height=100
).repeat(
    column=cols1
) 
chart2=alt.Chart(keyword_df).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y("total_views", type='quantitative'),
    color = "cluster:N"
).properties(
    width=100,
    height=100
).repeat(
    column=cols2
) 
chart3=alt.Chart(keyword_df).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y("total_views", type='quantitative'),
    color = "cluster:N"
).properties(
    width=100,
    height=100
).repeat(
    column=cols3
) 
st.subheader("Total Views with K-Means Clustering")
st.altair_chart(chart1)
st.altair_chart(chart2)
st.altair_chart(chart3)

#title sentiment
st.header("Title Sentiment Using TextBlob")
st.write("The dataset also includes an element called 'title sentiment', which computes a 'sentiment score' for the title in the interval between -1 and 1. A score of less than one means the title is 'negative', more than one means it is 'positive', and a score of zero means it is 'neutral'. Although it is concluded that there is not much correlation between video titles and popularity, the title is still a very crucial element to the video. ")
def title_sentiment(title):
    title = TextBlob(title)
    return title.sentiment.polarity
st.subheader("How is Title Sentiment Analyzed?")
st.write("Try creating your own title, and compare your score with similar titles from 5-Minute Crafts:")
user_title = st.text_input("Enter Your Title Here:")
user_title_sentiment= title_sentiment(user_title)
if user_title=='' or user_title== ' ':
    st.write("Please enter a title")
else:
    st.write("Your title sentiment is", user_title_sentiment)
    st.write("Here are some video titles that had similar scores to your title:")
    user_df = df[["title","total_views","title_sentiment"]][round((df["title_sentiment"]),1)==round(user_title_sentiment,1)].head().copy()
    user_df.columns=["Title","Total Views","Title Sentiment"]
    st.table(user_df)

st.header("References")
st.markdown("- scatterplots with mean and standard deviation lines: https://stackoverflow.com/questions/62854174/altair-draw-a-line-in-plot-where-x-y")
st.markdown("- correlation coefficient in linear regression: https://stackoverflow.com/questions/61277181/adding-r-value-correlation-to-scatter-chart-in-altair ")
st.markdown("- excluding columns from a pandas Data Frame: https://stackoverflow.com/questions/14940743/selecting-excluding-sets-of-columns-in-pandas")
st.markdown("- convert None values to str: https://stackoverflow.com/questions/3930188/how-to-convert-nonetype-to-int-or-string")
st.markdown("- pandas styler: https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Styler-Functions")
st.markdown("- Altair interaction: https://altair-viz.github.io/user_guide/interactions.html")
st.markdown("- title sentiment with TextBlob: https://textblob.readthedocs.io/en/dev/quickstart.html")

