# -*- coding: utf-8 -*-
"""
Created on Wed Nov 1 22:38:47 2017

"""

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy.streaming import Stream
import time
import json
import difflib
import re
import sentiment_svet as sent
import difflib
from difflib import SequenceMatcher as sm
import datetime

# Input the personal Twitter credentials/ deliberately hidden
ckey = '' 
csecret = ''
atoken = ''
asecret = ''

import http.client
from http.client import IncompleteRead
import urllib3
import requests
import sys

class listener(StreamListener):
    def on_data(self,data):
        try:
            tweet=json.loads(data)
            tweet=tweet['text']
            retweet_count=json.loads(data)
            retweet_count=retweet_count['retweet_count']
            timing= json.loads(data)
            timing=timing['created_at']
            
            stringtowrite=(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            stringtowrite+=','
            stringtowrite+=stock
            stringtowrite+=','
            stringtowrite+=tweet
            sentiment_value, confidence_percentage= sent.sentiment(tweet)
            count=0
            if confidence_percentage ==1:
                if sentiment_value=='positive':
                    count+=1
                    stringtowrite+=','+str(count)
                elif sentiment_value=='negative':
                    count-=0.4
                    stringtowrite+=','+str(count)
            #print(stringtowrite)
            print(timing, '   ', confidence_percentage, '   ', retweet_count, '   ', sentiment_value)
            
            stringtowrite=[]
            
            saveThis = str(time.time()) + ':::' + tweet #+ ':::' + sent.sentiment(tweet)
            saveFile=open('twitDB.csv','a')
            saveFile.write(saveThis)
            saveFile.write('\n')
            saveFile.close()
            
            conn = sqlite3.connect('sentiments.db')
            c = conn.cursor()
            c.execute('''CREATE TABLE tweet
             (tweet, sentiment score)''')
            c.execute("INSERT INTO tweet VALUES (tweet,sentiment_value)")
            conn.commit()
            conn.close()
            output.write('\n')
            output.close()
            
            
            return True
        
        except BaseException as e:
            print('failed ondata,',str(e))
            time.sleep(1)
            
def on_error(self,status):
    print(status)
    
def on_timeout(self):
    time.sleep(120)
    
auth=OAuthHandler(ckey,csecret)
auth.set_access_token(atoken,asecret)
twitterStream=Stream(auth,listener(),secure=True,)

def download():
    print("Please Enter The Name of A Stock : ",end="")
    global stock
    stock=str(input())
    
    while True:
        try:
            temp=['42589787','243318995','1307870299','18856867','47621568','65466158','15204596','20547642']
            res=['5988062','4898091','3108351','34713362','3034842802','19546277','32359921','624413']
            follow=res+temp
            twitterStream.filter(languages=["en"],track=[stock],follow=follow)
            time.sleep(5)
        except IncompleteRead:
            continue
        except requests.packages.urllib3.exceptions.ProtocolError:
            continue
        except KeyboardInterrupt:
            twitterStream.disconnect()
            break

download()
