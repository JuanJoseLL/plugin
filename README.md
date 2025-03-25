### RUN IT

#### Plugin/chatbot-dev
runs the app
```
node server.js
```

#### Pluegin/rag-backend
Runs the backend
```
uvicorn main:app --reload
```

#### Plugin/rag-backend/hd
Runs the page where you upload the file
```
python3 -m http.server 5500
```

