# Code for seminar: `fintopmet`

## Replication

1. Set your `kaggle` API key as described [here](https://www.kaggle.com/docs/api) and download kaggle's `CLI`
2. Download the data by running `make data`
3. Process (and explore) the data by running the notebook [a1-data.ipynb](notebooks/a1-data.ipynb)
4. Estimate the recursive and non-recursive models by running `make recursive` and `make non-recursive`; alternatively, `make models` for the two combined
5. Run `make forecast` to make the forecasts and get the csv-files ready to be submitted
