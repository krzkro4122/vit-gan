# vit-gan

## Running the project

- (Optional) Create a virtual environment (supports only python3.10!)

  ```bash
    virtualenv .venv
  ```

- (Optional) Activate the virtual environment

  ```bash
    source .venv/bin/activate
  ```

- Install the dependencies

  ```bash
    pip install -r requirements.txt
  ```

- Run the first version (complex code)

  ```bash
    SCRATCH=$(pwd) python main-v1.py
  ```

- Run the second version (RECOMMENDED - simpler code)

  ```bash
    python main-v2.py
  ```

  OR

  ```bash
    SCRATCH=$(pwd) python main-v2.py
  ```

> You can specify any dir as the `SCRATCH` variable to save the data and output there instead of your current directory.
