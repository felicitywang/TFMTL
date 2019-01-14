#!/usr/bin/env bash
set -e

mkdir -p data/raw/RTC
mkdir -p data/json/RTC

echo "Downloading the RTC data..."
wget -nc http://ugrad.cs.jhu.edu/~fxwang/rtc.tar.gz

echo "Untarring the RTC data..."
tar -zxvf rtc.tar.gz
mv -f rtc.tar.gz data/raw/RTC/


echo "Converting the RTC data to json..."
python3 ../../datasets/sentiment/RTC/convert_RTC_to_JSON.py ./

cp ../../datasets/sentiment/RTC/label.json ./label_RTC.json -f

mv -f rtc/ data/raw/RTC/
mv -f data.json.gz data/json/RTC/
mv -f index.json.gz data/json/RTC/

