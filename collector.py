# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:14:53 2017

"""

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_svet as s

import sqlite3

#consumer key, consumer secret, access token, access secret.
ckey = '' 
csecret = ''
atoken = ''
asecret = ''

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value,confidence = s.sentiment(tweet)
        print(tweet,sentiment_value,confidence)
        output=open("twitter-out.csv","a")
        output.write(tweet)
        
        conn = sqlite3.connect('sentiments.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE tweet
             (tweet, sentiment score)''')
        c.execute("INSERT INTO tweet VALUES (tweet,sentiment_value)")
        conn.commit()
        conn.close()
        output.write('\n')
        output.close()
        
        if confidence*100>=80:
            output=open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
            
        return(True)

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Palantir"])