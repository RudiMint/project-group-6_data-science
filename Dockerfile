FROM python:3.11

ENV APP_HOME /project-group-6_data-science

WORKDIR $APP_HOME

COPY . .

RUN pip cache purge
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install numpy
RUN pip install uvicorn
RUN pip install fastapi
RUN pip install Pillow
RUN pip install Keras
RUN pip install tensorflow

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]