.PHONY: help install data sample train evaluate predict app notebook clean smoke

help:
	@echo "make install    install dependencies"
	@echo "make data       download the Intel dataset from Kaggle"
	@echo "make sample     generate a tiny synthetic dataset (no Kaggle token needed)"
	@echo "make smoke      2-epoch run on the synthetic data to check everything works"
	@echo "make train      train with configs/default.yaml"
	@echo "make evaluate   evaluate the most recent checkpoint on the test set"
	@echo "make app        launch the Gradio demo"
	@echo "make notebook   open the walkthrough notebook"
	@echo "make clean      remove __pycache__ and .ipynb_checkpoints"

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

sample:
	python scripts/make_sample_data.py --dest data/sample

smoke: sample
	python scripts/train.py --run-name smoke --set data.root=data/sample train.epochs=2 \
		data.num_workers=0 data.batch_size=8 model.unfreeze_at_epoch=2
	python scripts/evaluate.py --checkpoint outputs/smoke/checkpoints/best.pt

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py --save-errors

app:
	python app.py

notebook:
	jupyter notebook notebooks/walkthrough.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
