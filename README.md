# Build
```
cd StereoMiddleburyEval
python -m pip install --user --upgrade twine wheel
python setup.py clean --all
python setup.py sdist bdist_wheel
```
# Upload to Test Pip
```
cd StereoMiddleburyEval
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
# Upload to Pip
```
cd StereoMiddleburyEval
python -m twine upload dist/*
```