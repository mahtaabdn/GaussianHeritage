# Use an Ubuntu 22.04 base image
FROM ubuntu:22.04

# Update package lists and install dependencies
RUN apt update && \
    apt install -y \
    python3-pip \
    docker.io  # Installing the Docker CLI inside the Flask container allows it to run Docker commands.

# Install Flask and Pillow
RUN pip3 install Flask Pillow

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Create a directory to store uploaded images on the host system
VOLUME /data

# Change directory to /app/flask
WORKDIR /app/flask

# Command to run the Flask application
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
