# ExelaChatbotPack

ExelaChatbotPack is a Python package that allows easy integration of Exela Chatbot functionalities into your projects. The package provides methods to insert files into the chatbot's index, query the index for responses, and delete the index as needed.

## Dependencies
- Python 3.11
- pip

## How to Use

1. Open a command prompt.

2. Create a virtual environment:
```python
$ virtualenv EnvName
```

3. Install the package from the provided wheel file:
```python
$ pip install ExelaChatbot-0.1.0-py3-none-any.whl
```

4. The `.env` file should be placed in the same directory as your code to access environment variables.

5. Add the following import statement to the file where you want to use the chatbot functionality:
```python
from ExelaChatbot import insert_into_index, query_index, delete_index
```

## Functionalities

### Insert File to Index:
```python
insert_into_index(doc_file_path)
```
- This method takes the path of the document as an argument (doc_file_path of type string).
- It returns True if successful, False otherwise.
- Only .pdf, .docx, and .txt files are allowed.

### Delete Index:
```python
delete_index()
```
- Delete all the files from Database.
- It returns True if successful, False otherwise.

### Query Index:
```python
query_index(query, chat_history=[])
```
- This method takes a mandatory argument query (of type string) and an optional argument chat_history (of type List).
- It returns a string message containing the generated response on success or a fixed message "Internal Error Occurred" on failure.

```python
from ExelaChatbot import insert_into_index, query_index, delete_index

# Example usage of functionalities
if insert_into_index('sample_document.docx'):
    print("Document inserted into the index successfully.")
else:
    print("Failed to insert the document.")

if query_index("What is the weather today?"):
    print("Query successful. Response received.")
else:
    print("Query failed. Internal error occurred.")

if delete_index():
    print("Index deleted successfully.")
else:
    print("Failed to delete the index.")
```


### Note
Please ensure that you have a valid API key and other necessary configurations set up in the .env file before using the ExelaChatbotPack functionalities.