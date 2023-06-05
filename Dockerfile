FROM node:18

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

ENV PORT=8000
ENV Link_Bucket=https://storage.googleapis.com/bucket-for-ml-model/model.json

EXPOSE 8000

CMD [ "npm", "start" ]