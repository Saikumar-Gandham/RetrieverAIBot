FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r Requirements.txt

COPY . .


RUN chmod +x app.sh

EXPOSE 5000

ENTRYPOINT ["./app.sh"]
