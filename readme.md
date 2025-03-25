### Streamlit App
This README provides instructions for running the Streamlit application.
Getting Started
Follow these steps to run the application:
Prerequisites

Python 3.7 or higher
pip (Python package installer)

### Installation

Clone this repository or download the source code
Navigate to the project directory
Install the required dependencies:

``` bash

 pip install -r requirements.txt

 ```


### Running the Application
The application requires two steps to run properly:

## Step 1: Run the main Python script
First, execute the main.py file which processes the data:
``` bash

python main.py
```

## Step 2: Launch the Streamlit app
After the main script completes, start the Streamlit interface:
```bash
streamlit run app.py
```
The application should open automatically in your default web browser. If not, you can access it at http://localhost:8501.


Troubleshooting

If you encounter any issues with dependencies, ensure all required packages are listed in requirements.txt
For errors related to data processing, check that main.py executed successfully before running the Streamlit app