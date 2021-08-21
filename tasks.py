from invoke import run, task

name = 'hstt'
src = 'hstt'


@task
def isort(c):
    run(f'isort {src}')


@task
def lint(c):
    run(f'flake8 {src} --ignore E501')


@task
def build(c):
    run('python setup.py sdist bdist_wheel')


@task
def clean(c):
    run('rm -rf ./build ./dist ./*.egg-info')


@task
def upload_test(c):
    run('python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*')


@task
def upload_release(c):
    run('python -m twine upload dist/*')


@task
def install_test(c):
    run(f'python -m pip install --index-url https://test.pypi.org/simple/ --no-deps --pre -U {name}')


@task
def install_release(c):
    run(f'python -m pip install -U {name}')


@task
def uninstall(c):
    run(f'python -m pip uninstall {name}')
