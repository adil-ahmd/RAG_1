FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
EXPOSE 8000

# default to FastAPI server (API endpoints)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# To run Streamlit instead, you can override at docker run / compose level:
# CMD ["streamlit", "run", "interface/app.py", "--server.port=8501", "--server.address=0.0.0.0"]