FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user ./spock /app/spock
RUN pip install --no-cache-dir --upgrade -e /app/spock

COPY --chown=user . /app

RUN chown -R user:user /app && chmod -R 755 /app

USER user
EXPOSE 8501
CMD ["streamlit", "run", "/app/app.py", "--server.port", "8501", "--server.enableXsrfProtection", "false"]