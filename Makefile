.PHONY: all

all:


data/m5:
	mkdir -p $@
	kaggle competitions download -c m5-forecasting-accuracy
	kaggle competitions download -c m5-forecasting-uncertainty
	unzip m5-forecasting-accuracy.zip -d $@/accuracy
	unzip m5-forecasting-uncertainty.zip -d $@/uncertainty
	mv $@/accuracy/sample_submission.csv $@/uncertainty/sample_submission_accuracy.csv
	mv $@/uncertainty/sample_submission.csv $@/uncertainty/sample_submission_uncertainty.csv
	rm -rf $@/accuracy/
	rm m5-forecasting-accuracy.zip
	rm m5-forecasting-uncertainty.zip
	mv $@/uncertainty/* $@
	rmdir $@/uncertainty

recursive:
	python scripts/train_models.py recursive --agg store
	python scripts/train_models.py recursive --agg store_dept
	python scripts/train_models.py recursive --agg store_cat

non-recursive:
	python scripts/train_models.py non-recursive --agg store
	python scripts/train_models.py non-recursive --agg store_dept
	python scripts/train_models.py non-recursive --agg store_cat

models: recursive non-recursive


forecasts:
	python -m fintopmet.models.forecast

submission:
	kaggle competitions submit -c m5-forecasting-accuracy -f data/m5/submission_final.csv -m "Replication of top solution of the M5 competition; refactored by jsr-p"

# Other 

benchmark:
	 python -m fintopmet.aux.benchmark
	 python -m fintopmet.aux.benchmark --pre-transform

tutorial:
	python -m fintopmet.models.gradientboosting

figures_desc:
	python scripts/proc_data_plot.py
	Rscript scripts/seasonal.R
