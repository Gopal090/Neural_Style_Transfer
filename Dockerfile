FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create non-root user (required by Hugging Face Spaces security policy)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code, templates, and pre-trained weights
COPY --chown=user . .

# Hugging Face Spaces requires the container to listen on port 7860
EXPOSE 7860

# Launch with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "app:app"]
