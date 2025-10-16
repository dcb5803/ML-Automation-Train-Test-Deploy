FROM python:3.10-slim
WORKDIR /app
COPY app.py ./
RUN pip install pandas scikit-learn flask joblib
EXPOSE 7860
CMD ["python", "app.py"]
