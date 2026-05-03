.PHONY: train predict api test clean install help

# Default target
help:
	@echo "📖 Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make train    - Train the model"
	@echo "  make predict  - Run demo prediction"
	@echo "  make api      - Start FastAPI server"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Remove generated files"

install:
	pip install -r requirements.txt

train:
	python -m src.train

predict:
	python -m src.predict

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

test:
	python -m pytest tests/ -v

clean:
	rm -rf data/processed/ models/*.pt models/*.png __pycache__/ src/__pycache__/ tests/__pycache__/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
