import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

dataFrame = pd.read_csv(r'netflix_titles.csv')

#Dataframe for Type of content: Movies, TV Shows
type_data = dataFrame["type"]
type_data_count = type_data.value_counts()

#Dataframe for Rating of content
rating_data = dataFrame["rating"]
rating_data = rating_data.fillna("No Info on Rating")
rating_data_count = rating_data.value_counts()

#Dataframe for Countries where the content is available
countries_data = dataFrame["country"].fillna("Unpecified")
countries_data_array = countries_data.str.split(', ',expand=True).stack().to_numpy()
countries_dataframe = pd.DataFrame(countries_data_array,columns=['Countries'])
countries_counts = countries_dataframe.value_counts()
countries_dataframe = countries_dataframe.Countries.unique()
countries_dataframe = pd.DataFrame(countries_dataframe,columns=['Country'])

#Datframe to analyze the sentiment through the description provided
sentiment_data = dataFrame[['release_year','description']]
for index,row in sentiment_data.iterrows():
    desc = row['description']
    analyzer = TextBlob(desc)
    polarity = analyzer.sentiment.polarity
    if polarity==0:
        msg = 'Neutral'
    elif polarity==1:
        msg = 'Positive'
    else:
        msg = 'Negative'
    sentiment_data.loc[[index,2],'Sentiment'] = msg
sentiment_data = sentiment_data.groupby(['release_year','Sentiment']).size().reset_index(name='Total Content')

#Group the data with low count
def group_lower_ranking_values(column):
    rating_counts = dataFrame.groupby(column).agg('count')
    pct_value = rating_counts[lambda x: x.columns[0]].quantile(.75)
    values_below_pct_value = rating_counts[lambda x: x.columns[0]].loc[lambda s: s < pct_value].index.values
    def fix_values(row):
        if row[column] in values_below_pct_value:
            row[column] = 'Other'
        return row 
    rating_grouped = dataFrame.apply(fix_values, axis=1).groupby(column).agg('count')
    return rating_grouped

#plotting Donut Chart to show the difference
#between number of TV shows and movies
plt.subplot(221)
plt.pie(type_data_count,labels=type_data.unique(),autopct="%.1f%%")
plt.title('Movies vs TV Shows')
circle = plt.Circle((0,0),0.7,color='white')
p = plt.gcf()
p.gca().add_artist(circle)

#plotting Pie Chart to show the count percentage of ratings
plt.subplot(222)
rating_grouped = group_lower_ranking_values('rating')
rating_labels = rating_grouped.show_id.sort_values().index
rating_counts = rating_grouped.show_id.sort_values()
plt.pie(rating_counts,colors=('r','y','b','g','m','c'),labels=rating_labels,autopct="%.1f%%")
plt.title('Distribution of Ratings')

#plotting bar graph to show the top 10 countries
#where content was added 
plt.subplot(223)
plt.bar(countries_dataframe.Country.head(10),countries_counts[:10])
plt.xlabel("Countries")
plt.ylabel("Movies/TV Shows")
plt.setp(plt.gca().get_xticklabels(),rotation=45,horizontalalignment='right')
plt.title("Major Countries")

#plotting stacked bar graph to show number of
#positve, neutral and negative contents released after 2010
plt.subplot(224)
sentiment_data=sentiment_data[sentiment_data['release_year']>=2010]
data_to_append = {'release_year':[2011,2012,2016,2017],'Sentiment':['Positive','Positive','Positive','Positive'],'Total Content':[0,0,0,0]}
data_to_append_frame = pd.DataFrame(data_to_append)
sentiment_data = sentiment_data.append(data_to_append_frame)

neu = sentiment_data[sentiment_data['Sentiment']=='Neutral']
neg = sentiment_data[sentiment_data['Sentiment']=='Negative']
pos = sentiment_data[sentiment_data['Sentiment']=='Positive']

plt.bar(sentiment_data['release_year'].unique(),pos['Total Content'].to_numpy(),color="orange",label='postive')
plt.bar(sentiment_data['release_year'].unique(),neu['Total Content'].to_numpy(),bottom=pos['Total Content'].to_numpy(),color='g',label='neutral')
plt.bar(sentiment_data['release_year'].unique(),neg['Total Content'].to_numpy(),bottom=neu['Total Content'].to_numpy()+pos['Total Content'].to_numpy(),color='r',label='negative')
plt.legend()
plt.xlabel("Years")
plt.ylabel("Sentiment")
plt.title("Type of Content over the Years")
plt.show()
