# 1. Use a standard slim image for Python 3.10
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies (CRITICAL for OpenCV/PIL)
# libgl1-mesa-glx and libglib2.0-0 are required for cv2 to function in a container
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY the requirements first
# This optimizes the build process by caching the pip install layer
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code
# This includes main.py, class_indices.json, and the trained_model folder
COPY . /app

# 7. Streamlit Environment Variables
# These replace the need for manual config.toml or credentials.toml files
ENV STREAMLIT_SERVER_PORT=80
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 8. Expose port 80 (standard HTTP)
EXPOSE 80

# 9. Launch command
# Using main.py as the entry point
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=80", "--server.address=0.0.0.0"]