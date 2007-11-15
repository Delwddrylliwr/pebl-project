import tempfile
import os, os.path
import inspect

location = tempfile.mkdtemp()
def setup():
    os.chdir(location)

def teardown():
    os.system("rm -rf %s" % location)

def datadir():
    return os.path.join(
            os.path.dirname(inspect.getfile(datadir)),
            'datasets'
    )

