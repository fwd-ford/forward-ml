.PHONY: serve score train train-temporal test

## Run FastAPI server locally
serve:
	uvicorn src.api:app --reload --port 8000

## Run batch scoring (calculate churn scores → Supabase)
score:
	python -m src.scoring

## Run full training pipeline (segmentation + classification, stratified random split)
train:
	python -m src.segmentation && python -m src.classification

## Train classifier with temporal split (validation realistic for production)
train-temporal:
	python -m src.classification --temporal-split

## Run tests
test:
	pytest tests/ -v
