#!/bin/bash

cd ..

mkdir data

cd data

mkdir raw && mkdir processed

cd raw

mkdir amazon_beauty && mkdir foursquare-tky

curl -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip

unzip ml-1m.zip 

curl -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip

unzip ml-100k.zip 

rm ml-1m.zip && rm ml-100k.zip

cd amazon_beauty

curl -k -o All_Beauty.json.gz https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/All_Beauty.json.gz

gzip -d All_Beauty.json.gz

cd ..

curl -o dataset_tsmc2014.zip http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip

unzip dataset_tsmc2014.zip

mv dataset_tsmc2014 foursquare-nyc

cp foursquare-nyc/dataset_TSMC2014_TKY.txt foursquare-tky

cp foursquare-nyc/dataset_TSMC2014_readme.txt foursquare-tky

rm dataset_tsmc2014.zip && rm foursquare-nyc/dataset_TSMC2014_TKY.txt