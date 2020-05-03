FROM python:latest

WORKDIR /deploy/

COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./RFC_model.pkl /deploy/

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
# CMD ["python", "./app.py"]