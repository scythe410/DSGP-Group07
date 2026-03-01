FROM python:3.9-slim

# Create the user and set permissions for HuggingFace Space
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /code

# Install system dependencies for OpenCV (required for YOLO cv2 operations)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Switch to the non-root user (HuggingFace constraint)
USER user
WORKDIR $HOME/app

# Copy all project files into the container
COPY --chown=user . $HOME/app

EXPOSE 7860

# Run the FastAPI server natively
CMD ["uvicorn", "app.backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
