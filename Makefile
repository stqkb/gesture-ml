.PHONY: train predict api test clean install collect camera help

help:
@echo "make install  - Install dependencies"
@echo "make train    - Train the model"
@echo "make predict  - Run demo prediction"
@echo "make api      - Start FastAPI server"
@echo "make collect  - Collect data from webcam"
@echo "make camera   - Real-time recognition"
@echo "make test     - Run tests"
@echo "make clean    - Remove generated files"

install:
pip install -r requirements.txt

train:
python -m src.train

predict:
python -m src.predict

api:
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

collect:
python -m src.collect_data

camera:
python -m src.camera_predict

test:
python -m pytest tests/ -v

clean:
rm -rf data/processed/ models/*.pt models/*.onnx models/*.xgb.pkl models/*.png __pycache__/ src/__pycache__/ tests/__pycache__/