python get_raw_users.py

echo 'Good job with raw'

python merge_raw_users.py 

echo 'Good job with merging'

python clean_raw.py 

echo 'Good job with cleaning'

python delete_small_users.py

echo 'Good job subsampling'

python get_json.py 

echo 'Good job creating json'

python preprocess.py

echo 'Good job preprocessing'
