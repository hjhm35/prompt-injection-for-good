# Gebruik een lichte Python-image
FROM python:3.11-slim

# Stel werkdirectory in
WORKDIR /app

# Kopieer projectbestanden
COPY . /app

# Installeer dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose poort
EXPOSE 5000

# Start de webinterface
CMD ["python", "run_web_ui.py"]
