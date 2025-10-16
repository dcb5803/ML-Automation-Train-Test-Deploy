FROM python:3.10-slim
WORKDIR /app
COPY app.py ./
RUN pip install pandas scikit-learn flask joblib gunicorn
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
