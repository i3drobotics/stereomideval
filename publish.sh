cd stereomideval
python -m pip install --upgrade pip
python -m pip install setuptools wheel twine
python -m pip install --upgrade -r requirements.txt
python -m pip install flake8 pytest

# stop the build if there are Python syntax errors or undefined names
#flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
#flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
# check build with pytest
pytest
pytest_exit_code=$?
if [ $pytest_exit_code != 0 ]; then
    echo "Pytest failed with exit code $pytest_exit_code"
    exit $pytest_exit_code
fi

python setup.py clean
python setup.py sdist bdist_wheel

twine upload --repository-url https://test.pypi.org/legacy/ dist/*
twine upload dist/*