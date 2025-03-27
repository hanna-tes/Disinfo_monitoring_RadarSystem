WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

* **Explanation:**
    * `FROM python:3.10-slim`: Uses a lightweight Python 3.10 base image.
    * `WORKDIR /app`: Sets the working directory inside the container.
    * `COPY requirements.txt .`: Copies the `requirements.txt` file into the container.
    * `RUN pip install --no-cache-dir -r requirements.txt`: Installs the Python dependencies.
    * `COPY . .`: Copies all files from your repository into the container.
    * `EXPOSE 8501`: Exposes port 8501, which is the default Streamlit port.
    * `CMD ["streamlit", "run", "app.py"]`: Sets the command to run your Streamlit application.
