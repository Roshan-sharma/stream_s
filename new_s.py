import numpy as np
import pandas as pd
import csv
import streamlit as st
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import altair as alt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt




@st.cache
def load_data(nrows):
	df1=pd.read_csv("review.csv",nrows=nrows)
	return df1



@st.cache
def load_text():
	df1=load_data(1000)
	df_text=pd.DataFrame(df1["Text"])
	return df_text



@st.cache
def load_summary():
	df1=load_data(1000)
	df_summary=pd.DataFrame(df1["Summary"])
	return df_summary





@st.cache
def load_summary_text():		
	df2=load_summary()
	#df2.to_csv(r'c:\data\pandas.txt', header=None, index=None, sep=' ', mode='a')
	np.savetxt(r'np.txt', df2.values, fmt='%s')
	#text_sentiment=pd.read_csv("np.txt",sep=" ",header=None)
	# file1=open("np.txt","r")
	# return file1


@st.cache
def load_sensor(nrows):
	df2=pd.read_csv("sensor.csv",nrows=nrows)
	return df2






@st.cache
def sentiment_analysis(sentiment_analysis_text):
	load_summary_text()
	result=SentimentIntensityAnalyzer().polarity_scores(sentiment_analysis_text)
	return result





dataframe = pd.DataFrame({
  'first column': ["review","sensor"]
})
st.sidebar.title("Choose your dataset")
option = st.sidebar.selectbox('',dataframe['first column'])
#df1=pd.read_csv(option+".csv")
#df=pd.DataFrame(df1)
Lemmatizer=WordNetLemmatizer()
porterstemmer=PorterStemmer()
st.title("Streamlit presentation")



if option=="review":
	dataframe = pd.DataFrame({
  'first column': ["Text","Summary"]})
	st.sidebar.subheader("Choose text for analysis and summary for sentiment analyze")
	review_option = st.sidebar.selectbox('',dataframe['first column'])
	if review_option=="Text":
		df_text=load_text()
		if st.sidebar.button("Load"):
			st.header("Datasets")
			st.write(df_text)
		if st.sidebar.button("Token"):
			count=0
			#df_text=pd.DataFrame(df["Text"])
			for row in df_text["Text"]:
				if count==1:
					break
				else:
					st.header("Data")
					st.write(row)
					st.subheader("Sentence token")
					st.success(sent_tokenize(row))
					st.subheader("Word token")
					st.success(word_tokenize(row))
					count=count+1
		if st.sidebar.button("Lemma and stem"):
			st.header("Lemma and stem")
			count=0
			#df_text=pd.DataFrame(df["Text"])
			for row in df_text["Text"]:
				sentence_words=word_tokenize(row)
				if count==9:
					break
				result=' '.join(Lemmatizer.lemmatize(i,"v") for i in sentence_words)
				result_porterstemmer=' '.join(porterstemmer.stem(i) for i in sentence_words)
				st.subheader("Data")
				st.write(row)
				st.subheader("Lemma")
				st.success(result)
				st.subheader("Stem")
				st.success(result_porterstemmer)
				count=count+1



	elif review_option=="Summary":
		df_summary=load_summary()
		if st.sidebar.button("Load"):
			st.write(df_summary)
		if st.sidebar.button("Token"):
			count=0
			#df_text=pd.DataFrame(df["Text"])
			for row in df_summary["Summary"]:
				if count==10:
					break
				else:
					st.header("Data")
					st.write(row)
					st.subheader("Sentence token")
					st.success(sent_tokenize(row))
					st.subheader("Word token")
					st.success(word_tokenize(row))
					count=count+1
		if st.sidebar.button("Lemma and stem"):
			st.header("Lemma and stem")
			count=0
			for row in df_summary["Summary"]:
				sentence_words=word_tokenize(row)
				if count==9:
					break
				result=' '.join(Lemmatizer.lemmatize(i,"v") for i in sentence_words)
				result_porterstemmer=' '.join(porterstemmer.stem(i) for i in sentence_words)
				st.subheader("Data")
				st.write(row)
				st.subheader("Lemma")
				st.success(result)
				st.subheader("Stem")
				st.success(result_porterstemmer)
				count=count+1
		if st.sidebar.button("Sentiment analysis"):
			st.header("Sentiment Analysis")
			count=0
			for t in df_summary["Summary"]:
				if count==10:
					break
				else:
					st.write(t)
					result=sentiment_analysis(t)
					st.success(result)
					count=count+1






if option=="sensor":
	sensor_data=load_sensor(100000)
	sensor_data=pd.DataFrame(sensor_data)
	if st.sidebar.checkbox("Sensor dataset"):
		st.subheader("Sensor datasets")
		st.write(sensor_data)
	dataframe = pd.DataFrame({
  'first column': ["sensor_00","sensor_01","sensor_02"]})
	st.sidebar.subheader("Choose your sensor for ploting")
	sensor_option = st.sidebar.selectbox('',dataframe['first column'])
	df3=pd.DataFrame({"date":sensor_data["timestamp"],sensor_option:sensor_data[sensor_option]})
	if st.sidebar.button("Plot"):
		df3=df3.rename(columns={'date':'index'}).set_index('index')
		st.subheader("Line chart")
		st.line_chart(sensor_data[sensor_option])
		st.subheader("Bar chart")
		st.bar_chart(sensor_data[sensor_option])
		df3.hist()
		st.subheader("Histogram")
		st.pyplot()
		st.subheader("Area chart")
		st.area_chart(sensor_data[sensor_option])
		#st.line_chart(sensor_data[sensor_option])
		#st.subheader("Area chart")
		#st.area_chart(sensor_data[sensor_option])