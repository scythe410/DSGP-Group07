FROM python:3.10-slim

# System dependencies required by OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces requires user with uid 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install Python dependencies (done before copying code for layer caching)
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir

# Copy application code
# Note: model weights (*.pt, *.pkl, *.safetensors) are gitignored and downloaded
# at startup from HuggingFace Hub via _bootstrap_models() in api.py
COPY --chown=user app/ ./app/
COPY --chown=user price-model/ ./price-model/
COPY --chown=user data/initial-cleaning/cleaned_no_outliers.csv ./data/initial-cleaning/cleaned_no_outliers.csv

# Space secrets (GEMINI_API_KEY, HF_TOKEN) are injected as environment variables
# by HuggingFace Spaces — no .env file needed in production

EXPOSE 7860

CMD ["sh", "-c", "python app/backend/bootstrap.py && python -m uvicorn app.backend.api:app --host 0.0.0.0 --port 7860"]
