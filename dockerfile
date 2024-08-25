# Use the official Python image from the Docker Hub
FROM python:3.10.6

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port on which the app runs
EXPOSE 5000

# Define the command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
