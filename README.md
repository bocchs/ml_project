# Classifying Wildfire Causes

To replicate experiments:

1. Download code from repository into working directory
2. Download wildfire dataset (https://www.kaggle.com/rtatman/188-million-us-wildfires) into working directory (dataset file should just be "FPA_FOD_20170508.sqlite")
3. Create ./weather_data directory
3. Download weather dataset (https://www.kaggle.com/selfishgene/historical-hourly-weather-data) into ./weather_data (dataset is several csv files)
4. Run script

<br />
The combined wildfire-weather dataset is generated on the script's first run and may take a few minutes. Future runs will read from the generated csv.

<br />
<br />
Run the script and save test results to text files (will create "tests" folder):

`python wildfire.py`

<br />
Run the script and output test results to console (without saving to text files):

`python wildfire.py --display`


<br />
The test results we obtain are provided in the tests folder in the repository.
