PYTHON ?= python3
DATASET ?= data.csv
TRAIN_DATA ?= data_training.csv
VALID_DATA ?= data_validation.csv
MODEL ?= saved_model.npy
CURVES ?= learning_curves.png

.PHONY: all split train predict clean help

all: train

split:
	$(PYTHON) Split.py --dataset $(DATASET)

train: split
	$(PYTHON) Train.py \
		--train $(TRAIN_DATA) \
		--valid $(VALID_DATA) \
		--save $(MODEL)

predict: $(MODEL) $(VALID_DATA)
	$(PYTHON) Predict.py --model $(MODEL) --dataset $(VALID_DATA)

$(MODEL): Train.py $(TRAIN_DATA) $(VALID_DATA)
	$(PYTHON) Train.py \
		--train $(TRAIN_DATA) \
		--valid $(VALID_DATA) \
		--save $(MODEL)

$(TRAIN_DATA) $(VALID_DATA): $(DATASET) Split.py
	$(PYTHON) Split.py --dataset $(DATASET)

clean:
	rm -f $(TRAIN_DATA) $(VALID_DATA) $(MODEL) $(CURVES)

help:
	@printf "Available targets:\n"
	@printf "  make split    Create training and validation CSV files\n"
	@printf "  make train    Split the dataset and train the model\n"
	@printf "  make predict  Evaluate the saved model on validation data\n"
	@printf "  make clean    Remove generated datasets and model outputs\n"
	@printf "\nOverride variables, for example:\n"
	@printf "  make train DATASET=other.csv MODEL=model.npy\n"
