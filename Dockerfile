FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY hearth ./hearth
COPY simulator ./simulator
ENV HEARTH_HOST=0.0.0.0 HEARTH_PORT=8765 HEARTH_DB=/data/hearth.db
VOLUME /data
EXPOSE 8765
CMD ["python", "-m", "hearth.server"]
