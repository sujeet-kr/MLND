

- Installing Flask dependencies for using API calls

```
    pip install -r requirements.txt
    export FLASK_APP=api.py
    flask run
```

- Request details
    - Verb - `POST`
    - url - `http://host:port/predict`
    - Request Header - `Content-Type = application/json`
    - Body - Sample body
        ```
            {
	            "question": "What is your name",
	            "id": 1,
	            "dataset_id":2
            }
        ```