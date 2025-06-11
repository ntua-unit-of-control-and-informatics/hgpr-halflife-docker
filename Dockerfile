FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./src /code/src

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/
COPY ./hgpr_model.pkl /code/

EXPOSE 8000

CMD ["python", "-m", "main", "--port", "8000"]
