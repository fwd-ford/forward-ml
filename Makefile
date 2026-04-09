.PHONY: serve score train test

## Run FastAPI server locally
serve:
	uvicorn src.api:app --reload --port 8000

## Run batch scoring (calculate churn scores → Supabase)
score:
	python -m src.scoring

## Run full training pipeline (segmentation + classification)
train:
	python -m src.segmentation && python -m src.classification

## Run tests
test:
	pytest tests/ -v
